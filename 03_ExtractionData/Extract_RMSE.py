"""
Compute the tracking error (RMSE) between the cursor position and the target
position for every participant (specific ID 100 to 400), session, sequence type and hand.
-------------
For each Excel file found in DATA_FOLDER (searched recursively):
  - each sheet = one block of the pursuit tracking task
  - the trial is split into 3 segments (see `segments`)
  - the cursor signal is low-pass filtered (Butterworth)
  - the RMSE between the filtered cursor and the target is computed per segment

Output: RMSE_alldata.xlsx (one row per participant x sheet x segment)
-------------
Author: Charlène Truong - Brain, Action and Skill Lab, UCLouvain
"""

import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt


# =============================================================================
# PARAMETERS
# =============================================================================

# Folder containing the raw data (.xlsx), searched recursively -> to adapt
folder_path = r"path/to/All_data"

# Column names in the Excel sheets
column_cursor = "Cursor_X"   # participant's cursor position (x)
column_ref = "Red_X"         # target position (x)

# Segment boundaries (sample indices, inclusive).
# Fade-in/fade-out excluded.
segments = {"segment1": (260, 459),
            "segment2": (500, 699),
            "segment3": (740, 939)}

# Low-pass filter parameters
cutoff_freq = 14      # cutoff frequency (Hz)
sampling_rate = 60    # sampling rate (Hz)
filter_order = 3      # Butterworth filter order


# =============================================================================
# FUNCTIONS
# =============================================================================

def butter_lowpass_filter(data, cutoff, fs, order=3):
            """Zero-phase low-pass Butterworth filter (filtfilt)."""
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, data)

def compute_rmse(signal1, signal2):
            """Root mean square error between two signals of the same length."""
            return np.sqrt(np.mean((signal1 - signal2) ** 2))


# =============================================================================
# MAIN LOOP
# =============================================================================

recap = []

# Browse all Excel files in the data folder
for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        if file_name.endswith((".xlsx", ".xls")):

            # --- Participant ID and group (first digit of the ID) ---
            # 1xx = IMP (implicit), 2xx = EXP (explicit),
            # 3xx = EXPCUE (explicit + cue), 4xx = CTR (control)
            participant = file_name[:3]
            print(participant)
            if participant[0] == '1':
                groupe = 'IMP'
            elif participant[0] == '2':
                groupe = 'EXP'
            elif participant[0] == '3':
                groupe = 'EXPCUE'
            elif participant[0] == '4':
                groupe = 'CTR'


            file_path = os.path.join(root, file_name)
            xls = pd.ExcelFile(file_path)

            for sheet_name in xls.sheet_names:
                # Skip familiarisation and training sheets
                if "familiarisation" not in sheet_name.lower() and "training" not in sheet_name.lower():

                    # Session name = sheet name without its last character.
                    # Files from sessions 4_1 and 5_1 correspond to the
                    # 24-h retention test -> suffix '24' added.
                    if "4_1" in file_path or "5_1" in file_path:
                        session_name = sheet_name[:-1]+'24'
                    else:
                        session_name = sheet_name[:-1]

                    # --- Session: Pre / Post / Post24 ---
                    if "Pre" in session_name:
                        session = 'Pre'
                    elif "Post" in session_name and "24" not in session_name:
                        session = 'Post'
                    elif "Post" in session_name and "24" in session_name:
                        session = 'Post24'

                    # --- Sequence type: Repeated / Mirror / Random ---
                    if "Seg" in session_name:
                        sequence = 'Repete'
                    elif "Mir" in session_name:
                        sequence = 'Mirror'
                    elif "Alea" in session_name:
                        sequence = 'Alea'

                    # --- Hand: trained hand or transfer to the other hand ---
                    if "_T" in session_name:
                        hand = 'Transfer'
                    else:
                        hand = 'Main'


                    df = pd.read_excel(xls, sheet_name=sheet_name)

                    # Skip sheets without the cursor / target columns
                    if column_cursor not in df.columns or column_ref not in df.columns:
                        continue

                    cursor_seg = df[column_cursor].to_numpy(dtype=float)
                    ref_seg = df[column_ref].to_numpy(dtype=float)

                    for seg_name, (start, end) in segments.items():
                        # Extract the segment
                        cursor_x = cursor_seg[start:end + 1]
                        ref_x = ref_seg[start:end + 1]

                        # Filter the cursor signal
                        cursor_x_filt = butter_lowpass_filter(cursor_x, cutoff_freq, sampling_rate, filter_order)

                        # RMSE between filtered cursor and (raw) target
                        rmse_segment = compute_rmse(cursor_x_filt, ref_x)

                        recap.append({
                            "ID": participant,
                            "Group": groupe,
                            "Feuille": session_name,
                            "Session" : session,
                            "Sequence" : sequence,
                            "Hand" : hand,
                            "RMSE": rmse_segment,
                        })

                        print(participant, groupe, session_name )


# =============================================================================
# EXPORT
# =============================================================================

with pd.ExcelWriter('RMSE_alldata.xlsx', engine='openpyxl') as writer:
    recap_df = pd.DataFrame(recap)
    recap_df.to_excel(writer)
print("✅ Recap file 'RMSE_alldata.xlsx' created")