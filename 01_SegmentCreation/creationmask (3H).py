"""
Generate the MASK sequences of the pursuit tracking task.
------------
Each segment is a 3-harmonic Fourier series (centred on 0) of 240 samples.
Random segments are generated and kept if they stay within the position limits
and have exactly `nb_amp` changes of direction.

The search is repeated until `nbseg` segments are found.

Output: segments_3H_MASK.xlsx (one row per segment: Fourier coefficients,
        phase, length, min/max, mean velocity/acceleration/jerk,
        mean amplitude, extrema Dir1..DirN and faded extrema DirFade1..DirFadeN)
        visualisation_MASK.xlsx (trajectories, one column per segment)
-------------
Author: Charlène Truong - Brain, Action and Skill Lab, UCLouvain
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# =============================================================================
# PARAMETERS
# =============================================================================

num_segments = 10000            # number of random segments generated per search
segment_length = 240            # samples per segment (4 s at 60 Hz)
nbseg = 194                     # number of segments to select
alphamax, alphamin = 10, -10    # position limits (amplitude bounds)
count = 0                       # number of selected segments (updated by segmentrecherche)
df_params = None                # parameters of the selected segments (updated by segmentrecherche)

nb_amp = 4                      # required number of changes of direction (peaks + troughs)

fade_len = 20                   # fade-in / fade-out length (samples)

# Output files
fade_filename = "segments_3H_MASK.xlsx"             # parameters + faded extrema (DirFade)
visu_filename = "visualisation_MASK.xlsx"           # trajectories (one column per segment)


# =============================================================================
# FUNCTIONS
# =============================================================================

def generate_segment(length):
    """Generate a segment from a centred 3-harmonic Fourier series.

    The segment is shifted (phase phi) so that it starts at the first
    zero-crossing of the original signal.
    Returns: segment, b0 (offset), a coefficients, b coefficients, phi.
    """
    a_coeffs = np.random.uniform(-5, 5, 3)
    b_coeffs = np.random.uniform(-5, 5, 3)
    i = np.arange(length)
    q = 2 * np.pi * i / length

    # Reference signal (before phase shift)
    segment_ref = np.zeros(length)
    for n in range(3):
        segment_ref += a_coeffs[n] * np.sin((n + 1) * q) + b_coeffs[n] * np.cos((n + 1) * q)

    # Centre the signal on 0
    b0 = -np.mean(segment_ref)
    segment_ref += b0

    # Phase shift = index of the first zero-crossing
    phi = 0
    for idx_seg in range(1, length):
        if (segment_ref[idx_seg - 1] < 0 < segment_ref[idx_seg]) or (segment_ref[idx_seg - 1] > 0 > segment_ref[idx_seg]):
            phi = idx_seg
            break

    # Final segment, shifted so it starts at 0
    q = 2 * np.pi * (i+phi) / length

    segment = np.zeros(length)
    for n in range(3):
        segment += a_coeffs[n] * np.sin((n + 1) * q) + b_coeffs[n] * np.cos((n + 1) * q)
    segment += b0

    return segment, b0, a_coeffs, b_coeffs, phi

def derivative_analytic(x, a, b, order):
    """Analytic derivative of the Fourier series.
    order = 1: velocity, 2: acceleration, 3: jerk.
    """
    y = np.zeros_like(x)
    for i, (ai, bi) in enumerate(zip(a, b)):
        n = i + 1
        if order == 1:  # velocity
            y += ai*n*np.cos(n*x) - bi*n*np.sin(n*x)
        elif order == 2:  # acceleration
            y += -ai*n**2*np.sin(n*x) - bi*n**2*np.cos(n*x)
        elif order == 3:  # jerk
            y += -ai*n**3*np.cos(n*x) + bi*n**3*np.sin(n*x)
    return y


# --- Post-processing: fade and export of the trajectories ---

def fade_in_from_zero(y, fade_len):
    """Raised-cosine fade-in: the first `fade_len` samples go smoothly from 0 to the signal."""
    fade = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len)))
    y[:fade_len] *= fade
    return y

def fade_out_to_zero(y, fade_len):
    """Raised-cosine fade-out: the last `fade_len` samples go smoothly from the signal to 0."""
    fade = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_len)))
    y[-fade_len:] *= fade
    return y

def reconstruct_segment(a0, a_coeffs, b_coeffs, length, phi):
    """Rebuild a segment from its Fourier coefficients and phase shift (no fade)."""
    i = np.arange(length)
    q = 2 * np.pi * (i+phi) / length
    segment = a0
    for n, (a_n, b_n) in enumerate(zip(a_coeffs, b_coeffs)):
        segment += a_n * np.sin((n + 1) * q) + b_n * np.cos((n + 1) * q)
    return segment

def process_segments_excel(df, output_file, fade_len=20):
    """
    Take the table of segment parameters. For each row:
      - rebuild the segment (with its phase shift phi)
      - apply a fade-in and a fade-out
      - detect the peaks and troughs of the faded segment
    The extrema values are added as columns DirFade1..DirFadeN and the
    table is saved to `output_file`.
    """
    dir_columns = []
    dir_values_all = []

    for idx, row in df.iterrows():
        b0 = row['b0']
        phi = row['Phi']
        a_coeffs = [row[f'a{i + 1}'] for i in range(3)]
        b_coeffs = [row[f'b{i + 1}'] for i in range(3)]

        # Rebuild the segment
        i_arr = np.arange(segment_length)
        q = 2 * np.pi * (i_arr+phi) / segment_length
        segment = np.zeros(segment_length)
        for n in range(3):
            segment += a_coeffs[n] * np.sin((n + 1) * q) + b_coeffs[n] * np.cos((n + 1) * q)
        segment += b0

        # Fade-in and fade-out
        segment = fade_in_from_zero(segment, fade_len)
        segment = fade_out_to_zero(segment, fade_len)

        # Peaks and troughs of the faded segment
        peaks, _ = find_peaks(segment)
        troughs, _ = find_peaks(-segment)
        extrema_indices = np.sort(np.concatenate((peaks, troughs)))
        extrema_values = segment[extrema_indices]

        dir_values = list(extrema_values)
        dir_values_all.append(dir_values)

        # Number of DirFade columns = max number of extrema found
        if len(dir_values) > len(dir_columns):
            dir_columns = [f'DirFade{i+1}' for i in range(len(dir_values))]

    # Add the DirFade columns (NaN when a segment has fewer extrema)
    max_dirs = len(dir_columns)
    dir_matrix = np.full((len(df), max_dirs), np.nan)

    for i, dirs in enumerate(dir_values_all):
        dir_matrix[i, :len(dirs)] = dirs

    dir_df = pd.DataFrame(dir_matrix, columns=dir_columns)
    df = pd.concat([df, dir_df], axis=1)

    # Save
    df.to_excel(output_file, index=False)

    print(f"✅ Excel file created: {output_file}")


# =============================================================================
# SEGMENT SELECTION
# =============================================================================

def segmentrecherche():
    global count, df_params

    # --- Generate the pool of random segments ---
    all_segments = []
    all_params = []
    for _ in range(num_segments):
        seg, b0, a_coeffs, b_coeffs, phi = generate_segment(segment_length)
        all_segments.append(seg)
        all_params.append([b0] + list(a_coeffs) + list(b_coeffs) + [phi])

    # Path length of each segment (sum of absolute sample-to-sample differences)
    segment_lengths = [np.sum(np.abs(seg[1:] - seg[:-1])) for seg in all_segments]

    # --- Select the segments ---
    columns = ['b0'] + [f'a{i + 1}' for i in range(3)] + [f'b{i + 1}' for i in range(3)] + ["Phi"] + \
              ['length', 'Max', 'Min', 'Mean_Velocity', 'Mean_Acceleration', 'Mean_Jerk', 'Mean_Amplitude'] + [
                  f'Dir{i + 1}' for i in range(nb_amp)]
    data = []

    count = 0
    i = 0
    while count < nbseg and i < len(all_segments):
        seg = all_segments[i]
        max_val = np.max(seg)
        min_val = np.min(seg)

        params = all_params[i]
        a_i, b_i = params[1:4], params[4:7]  # phi (last value) excluded
        x = 2 * np.pi * np.arange(segment_length) / segment_length

        # Analytic kinematics
        vel = derivative_analytic(x, a_i, b_i, 1)
        acc = derivative_analytic(x, a_i, b_i, 2)
        jerk = derivative_analytic(x, a_i, b_i, 3)

        mean_velocity = np.mean(np.abs(vel))
        mean_acceleration = np.mean(np.abs(acc))
        mean_jerk = np.mean((jerk ** 2))

        # Local maxima and minima (find_peaks only detects maxima -> -seg for minima)
        peaks, _ = find_peaks(seg)
        troughs, _ = find_peaks(-seg)
        nb_len = len(peaks) + len(troughs)  # number of changes of direction

        if nb_len == nb_amp:
            extrema = np.sort(np.concatenate((peaks, troughs)))
            values = seg[extrema]
            amplitudes = np.abs(np.diff(values))
            mean_amplitude = np.mean(amplitudes)

        # --- Filter: within limits and required number of direction changes ---
        if (max_val < alphamax and min_val > alphamin and nb_len == nb_amp
        ):
            row = list(params) + [segment_lengths[i], max_val, min_val,
                                  mean_velocity, mean_acceleration, mean_jerk, mean_amplitude] + list(values)
            data.append(row)
            count += 1
        i += 1

    # --- Keep the parameters of the selected segments ---
    df_params = pd.DataFrame(data, columns=columns)

    print(f"Number of selected segments: {count}")


# =============================================================================
# MAIN: repeat the search until enough segments are found
# =============================================================================

run = True
while run:
    segmentrecherche()
    if count >= nbseg:
        print("✅ Required number of segments reached, stopping.")
        run = False
    else:
        print('continue')


# =============================================================================
# POST-PROCESSING
# =============================================================================

# 1) Parameters + faded extrema values saved to Excel
process_segments_excel(df_params, fade_filename, fade_len)

# 2) Trajectories exported for the task: one column = one segment,
#    one row = one sample (segment_length rows)
df_coeffs = pd.read_excel(fade_filename)

all_segments = []
for idx, row in df_coeffs.iterrows():
    a0 = row['b0']
    phi = row['Phi']
    a_coeffs = row[['a1', 'a2', 'a3']].values
    b_coeffs = row[['b1', 'b2', 'b3']].values
    segment = reconstruct_segment(a0, a_coeffs, b_coeffs, segment_length, phi)
    all_segments.append(segment)

df_segments = pd.DataFrame(all_segments).T
df_segments.to_excel(visu_filename, index=False)
print(f"✅ Excel file created: {visu_filename}")