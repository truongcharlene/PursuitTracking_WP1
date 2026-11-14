"""
Pursuit Tracking Task
-------------

One-dimensional pursuit tracking task controlled with a joystick, used to study
motor sequence learning through motor imagery (or observation for the Control group),
with or without explicit knowledge of the repeated sequence.

Each trial is built as:  mask -> segment 1 -> segment 2 -> segment 3 -> mask
Segments and masks are generated from Fourier coefficients stored in Excel files
(folder ``FileTask/``). The participant tracks a red target with a black cursor.

Groups (chosen at start-up)
---------------------------
1  Implicit         no cue during training
2  Explicit         during training: target turns blue + beep at each repeated segment
3  Explicit guided  as Explicit + numbered bars showing where the target changes direction
4  Control          training by eye-tracking only (no movement), with the guided cues

In groups 1-3, training is done by motor imagery: the thumb stays still on the
joystick and the cursor stays locked on the target.

Sessions
--------
 0  Familiarisation   right hand then left hand, random segments
 1  Pre-test          right hand: Repeated / Random / Mirror segments
 2  Pre-transfer      left hand:  Repeated / Random / Mirror segments
 3  Training          repeated segment (imagery / observation)
 4  Post-test         right hand
 5  Post-transfer     left hand
-1  Quit

After a session is completed, the next one starts automatically.
Drawn segments are saved per participant, so the task can be stopped and
resumed without re-using the same random segments.

Participant-facing instructions are in French.

-------------
Author: Charlène Truong - Brain, Action and Skill Lab, UCLouvain
"""

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pygame
from screeninfo import get_monitors

# Windows-only modules (sounds and window focus). The task runs without them.
try:
    import winsound
except ImportError:
    winsound = None
try:
    import pygetwindow as gw
except ImportError:
    gw = None


# =============================================================================
# PARAMETERS
# =============================================================================

# --- Paths ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TASK_DIR = BASE_DIR / "FileTask"      # coefficient files + sounds
DATA_DIR = BASE_DIR / "data"          # output, one folder per participant

SEGMENT_FILE = TASK_DIR / "segments_3H_SEG.xlsx"   # repeated / random segments
MASK_FILE = TASK_DIR / "segments_3H_MASK.xlsx"     # masks (also used for familiarisation)
BIP_SOUND = TASK_DIR / "bip.wav"                   # beep at the start of each repeated segment
SESSION_SOUNDS = {s: TASK_DIR / f"session{s}.wav" for s in range(6)}

# --- Screen and joystick -------------------------------------------------------
SCREEN_DIAGONAL_CM = 14 * 2.54                 # 14-inch screen
SCREEN_WIDTH_CM = SCREEN_DIAGONAL_CM * 0.872   # width of a 16:9 screen (or set in cm)
MONITOR_INDEX = 0          # monitor on which the task is displayed
JOY_RANGE_CM = 8           # max horizontal amplitude of the cursor on screen (cm)
JOY_SCALE = 1              # extra gain of the cursor (1 = none)
JOY_AXIS = 2               # 2 = right stick, 0 = left stick
JOY_START_BUTTONS = (4, 5)  # buttons used to start / continue
MAX_CURVE = 10             # max amplitude of the segments (rescaled to the screen)

# --- Sequence -------------------------------------------------------------------
N_POINTS_SEGMENT = 240     # samples per segment
FPS = 60                   # display / sampling rate (Hz)
FADE_LEN = 20              # samples of fade-in and fade-out (return to zero)
N_SEGMENTS = 3             # segments per trial
N_MASKS = 2                # masks per trial (start + end)

# --- Design ---------------------------------------------------------------------
N_TRIALS_FAM = 4           # familiarisation trials (half right hand, half left hand)
N_BLOCKS_TEST = 3          # blocks of (Repeated, Random, Mirror) per test session
N_TRIALS_TRAINING = 60
PAUSE_SHORT_S = 20
PAUSE_LONG_S = 300         # after training
PAUSE_EVERY = 10           # pause every x training trials
RANDOMIZE_TEST_ORDER = False   # False = the group's default order for every participant

# --- Display --------------------------------------------------------------------
WINDOW_TITLE = "Tracking"
SCREEN_COLOR = (170, 170, 170)
RED = (150, 0, 0)
BLUE = (0, 0, 180)
TARGET_RADIUS = 15               # px
CURSOR_COLOR = (0, 0, 0)
CURSOR_RADIUS = 15
CURSOR_WIDTH = 4
TEXT_COLOR = (0, 0, 0)
TEXT_IMPORTANT = RED
BAR_COLOR = (150, 150, 150)      # numbered bars (Explicit guided, Control)
VAS_RADIUS = 5
NASA_SCALE_CM = 20

# Training cues: the repeated segments are samples [start, end] of the trial
CUE_START = N_POINTS_SEGMENT                       # end of the first mask
CUE_END = N_POINTS_SEGMENT * (N_SEGMENTS + 1)      # start of the last mask
BEEP_FRAMES = [N_POINTS_SEGMENT * (k + 1) for k in range(N_SEGMENTS)]
# As in the original experiment, the beep is played with SND_ALIAS: Windows may
# play its default system sound instead of bip.wav. Use winsound.SND_FILENAME to
# play bip.wav itself.
BEEP_FLAGS = (winsound.SND_ALIAS | winsound.SND_ASYNC) if winsound else 0


# =============================================================================
# INSTRUCTIONS (French) - each line is a list of (text, colour) chunks
# =============================================================================
T, I, B = TEXT_COLOR, TEXT_IMPORTANT, BLUE
N_TEST_TRIALS = N_BLOCKS_TEST * 3
BLANK = [(" ", T)]
CONSIGNE = [("Consigne:", T)]

# --- Familiarisation, pre-test and pre-transfer (same for all groups) ----------
FAM_RIGHT_LINES = [
    CONSIGNE, BLANK,
    [("Utilisez la main ", T), ("DROITE", I), (".", T)],
    [("Essayez de suivre la cible rouge le plus précisément possible", T)],
    [("en BOUGEANT votre pouce, représenté par un cercle noir.", T)],
    BLANK, BLANK,
    [(f"Appuyez sur le bouton du joystick pour essayer ({N_TRIALS_FAM // 2}x)", T)],
]
FAM_LEFT_LINES = [
    CONSIGNE, BLANK,
    [("Utilisez la main ", T), ("GAUCHE", I), (".", T)],
    [("Essayez de suivre la cible rouge le plus précisément possible", T)],
    [("en BOUGEANT votre pouce, représenté par un cercle noir.", T)],
    BLANK, BLANK,
    [(f"Appuyez sur le bouton du joystick pour essayer ({N_TRIALS_FAM // 2}x)", T)],
]

START_TEST = [(f"Appuyez sur le bouton du joystick pour commencer ({N_TEST_TRIALS}x)", T)]
PRE_TEST_LINES = [
    CONSIGNE, BLANK,
    [("Utilisez la main", T), (" DROITE", I), (".", T)],
    [("Suivez la cible rouge le plus précisément possible", T)],
    [("en BOUGEANT votre pouce, représenté par un cercle noir.", T)],
    BLANK, BLANK, START_TEST,
]
PRE_TRANSFER_LINES = [
    CONSIGNE, BLANK,
    [("Utilisez la main ", T), ("GAUCHE", I), (".", T)],
    [("Suivez la cible rouge le plus précisément possible", T)],
    [("en BOUGEANT votre pouce, représenté par un cercle noir.", T)],
    BLANK, BLANK, START_TEST,
]

# --- Post-test and post-transfer (wording depends on the group) ----------------
MOVE_LINE = [("en BOUGEANT", I), (" votre pouce, représenté par un cercle noir.", T)]
FOLLOW_LINE = [("Suivez la cible le plus précisément possible", T)]


def post_lines(hand_chunks, follow_line, note=None):
    lines = [CONSIGNE, BLANK, hand_chunks, follow_line, MOVE_LINE, BLANK]
    if note:
        lines += [[(note, T)], BLANK]
    return lines + [BLANK, START_TEST]


POST_IMPLICIT = {
    "post_test": post_lines([("Utilisez le joystick", T), (" DROIT", I), (".", T)],
                            [("Suivez la cible rouge le plus précisément possible", T)]),
    "post_transfer": post_lines([("Utilisez le joystick ", T), ("GAUCHE", I), (".", T)],
                                [("Suivez le plus précisément possible", T)]),
}
NOTE_EXPLICIT = "La cible restera rouge tout le long des essais et vous n'aurez pas de BIPs sonores"
POST_EXPLICIT = {
    "post_test": post_lines([("Utilisez le joystick", T), (" DROIT", I), (".", T)],
                            FOLLOW_LINE, NOTE_EXPLICIT),
    "post_transfer": post_lines([("Utilisez le joystick ", T), ("GAUCHE", I), (".", T)],
                                FOLLOW_LINE, NOTE_EXPLICIT),
}
POST_GUIDED = {
    "post_test": post_lines([("Utilisez la main", T), (" DROITE", I), (".", T)], FOLLOW_LINE,
                            "La cible restera rouge tout le long des essais, vous n'aurez pas de "
                            "barres numérotées ni de BIPs sonores"),
    "post_transfer": post_lines([("Utilisez la main ", T), ("GAUCHE", I), (".", T)], FOLLOW_LINE,
                  "La cible restera rouge tout le long des essais, vous n'aurez pas de barres "
                  "chiffrées ni de BIPs sonores"),
}

# --- Training instructions (motor imagery; observation for the Control group) --
START_TRAINING = [(f"Appuyez sur le bouton du joystick pour commencer l'entrainement "
                   f"({N_TRIALS_TRAINING}x)", T)]
NO_MOVE = [("Vous NE BOUGEREZ PAS pendant les essais d'entrainement.", I)]
IMAGINE = [("IMAGINEZ VOIR et RESSENTIR votre pouce ", I),
           ("suivre la cible le plus précisément possible", T)]
WITH_JOYSTICK = [("à l'aide de votre joystick, représenté par un cercle noir.", T)]
BLUE_COMMA = [("Lorsque la cible deviendra", T), (" bleue", B), (",", T)]
BLUE_DOT = [("Lorsque la cible deviendra", T), (" bleue.", B)]
SAME_3_TIMES = [("le mouvement de la cible suivra une séquence spécifique 3 fois d'affilé", T)]
SAME_ALL_TRAINING = [("et restera la même tout au long de l'entrainement.", T)]
BEEP_EACH = [("Un BIP sonore vous indiquera le début de chaque séquence.", T)]
BEEP_EACH_AND = [("Un BIP sonore vous indiquera le début de chaque séquence, et", T)]
BARS = [("des barres (numérotées de 1 à 4) vous indiqueront où la cible changera de direction.", T)]

TRAINING_LINES = {
    "Implicit": [
        CONSIGNE, BLANK,
        [("Positionnez votre pouce sur le joystick", T), (" DROIT", I), (".", T)], BLANK,
        NO_MOVE, BLANK, IMAGINE, WITH_JOYSTICK,
        BLANK, BLANK, START_TRAINING],
    "Explicit": [
        CONSIGNE, BLANK,
        [("Positionnez votre pouce sur le joystick", T), (" DROIT", I), (".", T)], BLANK,
        NO_MOVE, BLANK, IMAGINE, WITH_JOYSTICK, BLANK,
        BLUE_COMMA, SAME_3_TIMES, SAME_ALL_TRAINING, BLANK, BEEP_EACH,
        BLANK, BLANK, START_TRAINING],
    "ExplicitGuided": [
        CONSIGNE, BLANK,
        [("Positionnez votre pouce", T), (" DROIT", I), (" sur le joystick.", T)], BLANK,
        NO_MOVE, BLANK, IMAGINE, WITH_JOYSTICK, BLANK,
        BLUE_COMMA, SAME_3_TIMES, SAME_ALL_TRAINING, BLANK, BEEP_EACH_AND, BARS,
        BLANK, BLANK, START_TRAINING],
    "Control": [
        CONSIGNE, BLANK,
        [("Pendant l'entrainement, retirez votre pouce du joystick", I)],
        NO_MOVE,
        [("Suivez la cible avec le simple mouvement de vos yeux", I)], BLANK,
        BLUE_COMMA,
        [("le mouvement de la cible suivra une séquence spécifique 3 fois d'affilée", T)],
        SAME_ALL_TRAINING, BLANK,
        [("Un BIP sonore vous indiquera le début de la séquence, et", T)], BARS,
        BLANK, BLANK, START_TRAINING],
}

# --- Reminders shown after each pause during training (header added later) -----
FOOTER = [("Appuyez sur le bouton du joystick pour continuer l'entrainement", T)]
ALWAYS_RIGHT = [("Toujours avec votre main DROITE,", I)]
KEEP_IMAGINING = [("Continuez à IMAGINER VOIR et RESSENTIR", I)]
THUMB_MOVE = [("votre pouce bouger pour suivre la cible le plus précisément possible", T)]
SAME_AGAIN = [("elle suivra à nouveau la même séquence spécifique.", T)]
BEEP_AGAIN = [("Le BIP sonore vous indiquera le début de chaque séquence.", T)]
BEEP_AGAIN_AND = [("Le BIP sonore vous indiquera le début de chaque séquence, et", T)]
BARS_AGAIN = [("les barres (numérotées de 1 à 4) vous indiqueront où la cible changera de direction.", T)]

REMINDER_LINES = {
    "Implicit": [
        BLANK, ALWAYS_RIGHT, KEEP_IMAGINING, BLANK, THUMB_MOVE, WITH_JOYSTICK,
        BLANK, BLANK, FOOTER],
    "Explicit": [
        BLANK, [("Toujours avec votre main DROITE", I)], KEEP_IMAGINING, THUMB_MOVE, WITH_JOYSTICK,
        BLANK, BLUE_COMMA, SAME_AGAIN, BEEP_AGAIN, BLANK, BLANK, FOOTER],
    "ExplicitGuided": [
        BLANK, ALWAYS_RIGHT, KEEP_IMAGINING, THUMB_MOVE, WITH_JOYSTICK,
        BLANK, BLUE_DOT, SAME_AGAIN, BLANK, BEEP_AGAIN_AND, BARS_AGAIN, BLANK, BLANK, FOOTER],
    "Control": [
        BLANK, [("Continuez à suivre la cible avec vos yeux, sans bouger", I)],
        BLANK, BLUE_DOT, SAME_AGAIN, BLANK, BEEP_AGAIN_AND, BARS_AGAIN, BLANK, BLANK, FOOTER],
}

# --- Imagery quality scales (7-point) and NASA-TLX (0-20) ----------------------
VAS_LABELS = ["Très difficile", "Difficile", "Assez difficile", "Neutre (ni facile",
              "Assez facile", "Facile", "Très facile"]
VAS_LABELS_VISUAL = ["à visualiser"] * 3 + ["ni difficile)"] + ["à visualiser"] * 3
VAS_LABELS_KINESTHETIC = ["à sentir"] * 3 + ["ni difficile)"] + ["à sentir"] * 3

NASA_DIMENSIONS = ["Exigence Mentale", "Exigence Physique", "Exigence Temporelle",
                   "Performance", "Effort", "Frustration"]
NASA_QUESTIONS = ["Comment était la difficulté mentale de la tâche? ",
                  "Comment était la difficulté physique de la tâche? ",
                  "Comment avez-vous ressenti la pression temporelle lors de l'exécution de la tâche?",
                  "Avez-vous réussi à accomplir ce qui vous était demandé ?",
                  "À quel point l'exercice fut-il difficile à accomplir?",
                  "Etiez-vous stressé, découragé, agacé, inconfortable?"]
NASA_LOW = ["Très faible", "Très faible", "Très faible", "Pas du tout", "Très facile", "Pas du tout"]
NASA_HIGH = ["Très élevée", "Très élevée", "Très élevée", "Parfaitement", "Très difficile", "Extremement"]


# =============================================================================
# GROUPS
# =============================================================================
# folder         prefix of the participant output folder
# default_order  order of conditions in test blocks (1 = Repeated, 2 = Random, 3 = Mirror);
#                None = entered by the experimenter for each participant
# question_every questionnaires every x training trials
# blue_beep      target turns blue + beep during the repeated segments (training)
# bars           numbered bars at the direction changes of the repeated segment (training)
GROUPS = {
    "Implicit": dict(folder="GrpImp", default_order=[1, 2, 3], question_every=60,
                     blue_beep=False, bars=False, post_lines=POST_IMPLICIT),
    "Explicit": dict(folder="GrpE", default_order=[3, 1, 2], question_every=20,
                     blue_beep=True, bars=False, post_lines=POST_EXPLICIT),
    "ExplicitGuided": dict(folder="GrpExG", default_order=[1, 2, 3], question_every=20,
                           blue_beep=True, bars=True, post_lines=POST_GUIDED),
    "Control": dict(folder="GrpCtr", default_order=None, question_every=20,
                    blue_beep=True, bars=True, post_lines=POST_GUIDED),
}
GROUP_MENU = {1: "Implicit", 2: "Explicit", 3: "ExplicitGuided", 4: "Control"}

TEST_KINDS = {
    "pre_test": dict(label="Pre-test", prefix="Pre_", export="PRE-TEST"),
    "pre_transfer": dict(label="Transfer-test", prefix="Pre_T", export="TRANSFERT"),
    "post_test": dict(label="Post-test", prefix="Post_", export="POST-TEST"),
    "post_transfer": dict(label="Transfer-test", prefix="Post_T", export="TRANSFERT"),
}

# Which test is run in sessions 1, 2, 4 and 5. The order chosen for the pre-tests
# (test = right hand first, or transfer = left hand first) is also used for the post-tests.
SESSION_ORDERS = {
    "test_first": {1: "pre_test", 2: "pre_transfer", 4: "post_test", 5: "post_transfer"},
    "transfer_first": {1: "pre_transfer", 2: "pre_test", 4: "post_transfer", 5: "post_test"},
}
PAUSE_AFTER_SESSION = {1: True, 2: True, 4: True, 5: False}


def test_lines(kind):
    if kind == "pre_test":
        return PRE_TEST_LINES
    if kind == "pre_transfer":
        return PRE_TRANSFER_LINES
    return GROUPS[group]["post_lines"][kind]


# =============================================================================
# SEGMENT GENERATION
# =============================================================================
def trig_series(n_points, a_coef, b_coef, b0, phi):
    """Fourier series: sum of a_n*sin(n*q) + b_n*cos(n*q) + b0."""
    q = 2 * np.pi * (np.arange(n_points) + phi) / n_points
    y = np.zeros(n_points)
    for n in range(len(a_coef)):
        y += a_coef[n] * np.sin((n + 1) * q) + b_coef[n] * np.cos((n + 1) * q)
    return y + b0


def fade_in(y, n):
    y[:n] *= 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    return y


def fade_out(y, n):
    y[-n:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, n)))
    return y


def make_segment(idx, table):
    """Build segment number `idx` (1-based row of `table`), starting and ending at zero."""
    row = table.loc[idx - 1]
    a = row[["a1", "a2", "a3"]].values.astype(float)
    b = row[["b1", "b2", "b3"]].values.astype(float)
    y = trig_series(N_POINTS_SEGMENT, a, b, float(row["b0"]), float(row["Phi"]))
    return fade_out(fade_in(y, FADE_LEN), FADE_LEN)


def load_ints(path):
    """Read integers saved in a text file. Returns None if the file does not exist."""
    if not path.exists():
        return None
    content = path.read_text().strip()
    return [int(x) for x in content.split()] if content else []


class SegmentPool:
    """Draws segment indices without replacement and saves the used ones to disk,
    so that the same random segment is never presented twice to a participant,
    even across separate runs of the script."""

    def __init__(self, n_total, used_file, excluded=()):
        self.used_file = used_file
        used = load_ints(used_file)
        if used is None:
            print(f"🆕 Pas de fichier {used_file.name} pour ce participant")
        self.used = set(used or [])
        self.available = set(range(1, n_total + 1)) - self.used - set(excluded)

    def draw(self, n):
        if len(self.available) < n:
            raise ValueError(f"Pas assez de segments restants pour en tirer {n}.")
        picked = random.sample(sorted(self.available), n)
        self.available -= set(picked)
        self.used.update(picked)
        self.used_file.write_text(" ".join(map(str, sorted(self.used))))
        return picked


# =============================================================================
# CONSOLE INPUT (experimenter)
# =============================================================================
def ask_int(prompt, valid=None):
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("Entrée invalide, veuillez entrer un nombre entier.")
            continue
        if valid is None or value in valid:
            return value
        print(f"Veuillez entrer un nombre entre {min(valid)} et {max(valid)}.")


def ask_session():
    screen.fill(SCREEN_COLOR)
    pygame.display.flip()
    names = {"pre_test": "PRE", "pre_transfer": "PRETRANSFER",
             "post_test": "POST", "post_transfer": "le TRANSFERT"}
    s = {n: names[k] for n, k in SESSION_ORDERS[session_order].items()}
    print(f"CHOIX DE SESSION:  0 pour FAM; 1 pour {s[1]}; 2 pour {s[2]}; 3 pour TRAINING; "
          f"4 pour {s[4]}; 5 pour {s[5]}; -1 pour stop")
    session = ask_int("Numéro de session : ", range(-1, 6))
    print(f"Vous avez choisi la session {session}.")
    return session


# =============================================================================
# DISPLAY HELPERS
# =============================================================================
def quit_task():
    pygame.quit()
    sys.exit()


def play_session_sound(session):
    path = SESSION_SOUNDS[session]
    if winsound is not None and path.exists():
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    time.sleep(0.5)


def play_beep():
    if winsound is not None:
        winsound.PlaySound(str(BIP_SOUND), BEEP_FLAGS)


def focus_window():
    """Bring the task window to the front (Windows only)."""
    if gw is None:
        return
    windows = gw.getWindowsWithTitle(WINDOW_TITLE)
    if not windows:
        return
    window = windows[monitor_index] if monitor_index < len(windows) else windows[0]
    try:
        window.activate()
    except gw.PyGetWindowException as e:
        if "Error code from Windows: 0" not in str(e):
            raise


def draw_centered(text_font, text, y, color=TEXT_COLOR):
    surface = text_font.render(text, True, color)
    screen.blit(surface, (mid_w - surface.get_width() / 2, y))


def show_lines(lines):
    """Display instruction lines, each made of coloured chunks, centred on screen."""
    screen.fill(SCREEN_COLOR)
    y = HEIGHT // 3
    for line in lines:
        surfaces = [font_small.render(text, True, color) for text, color in line]
        x = WIDTH // 2 - sum(s.get_width() for s in surfaces) // 2
        for s in surfaces:
            screen.blit(s, (x, y))
            x += s.get_width()
        y += font_small.get_height() + 5
    pygame.display.flip()


def wait_for_button(lines, on_escape=None):
    """Show instructions until a joystick button is pressed.
    on_escape: None (Escape ignored), "back" (return False) or "quit" (close the task)."""
    while True:
        show_lines(lines)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_task()
            elif event.type == pygame.JOYBUTTONDOWN and event.button in JOY_START_BUTTONS:
                return True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if on_escape == "back":
                    return False
                if on_escape == "quit":
                    quit_task()


def pause(seconds):
    screen.fill(SCREEN_COLOR)
    if seconds >= 60:
        text = f"Pause de {seconds / 60:g} minutes"
    else:
        text = f"Pause de {seconds} secondes"
    draw_centered(font_small, text, HEIGHT // 3)
    pygame.display.flip()
    start = time.time()
    while time.time() - start < seconds:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return


def read_cursor_x():
    if joystick:
        axis = joystick.get_axis(JOY_AXIS)
        return int(cursor_left + (axis + 1) / 2 * (cursor_right - cursor_left))
    return pygame.mouse.get_pos()[0]


def draw_target_and_cursor(target_x, y, lock_cursor, color=RED):
    """Draw target and cursor. If lock_cursor (imagery, observation), the cursor stays
    on the target. Returns the real cursor position (recorded in the data)."""
    pygame.draw.circle(screen, color, (target_x, y), TARGET_RADIUS)
    cursor_x = read_cursor_x()
    shown_x = target_x if lock_cursor else cursor_x
    pygame.draw.circle(screen, CURSOR_COLOR, (shown_x, y), CURSOR_RADIUS, CURSOR_WIDTH)
    return cursor_x


def draw_direction_bars(segment_idx):
    """Numbered bars (1-4, left to right) at the positions where the repeated
    segment changes direction (columns DirFade1-4 of the segment file)."""
    values = sorted(segment_table.loc[segment_idx - 1, ["DirFade1", "DirFade2", "DirFade3",
                                                        "DirFade4"]].values.astype(float))
    for number, val in enumerate(values, start=1):
        x = int(mid_w + val * coef_scale - 10)
        y = int(mid_h - 140)
        pygame.draw.rect(screen, BAR_COLOR, (x + 10, y + 45, 4, 200))
        screen.blit(font_number.render(str(number), True, TEXT_COLOR), (x, y))


# =============================================================================
# TRIAL
# =============================================================================
def run_trial(segment_ids, mask_ids, table, mirror=False, lock_cursor=False, cues=False):
    """One tracking trial: mask - segments - mask. Returns the frame-by-frame data.
    cues=True (training of the Explicit, Explicit guided and Control groups) adds the
    blue target, the beeps and/or the numbered bars, depending on the group."""
    segments = [make_segment(i, table) for i in segment_ids]
    masks = [make_segment(i, mask_table) for i in mask_ids]
    y_all = np.concatenate([masks[0], *segments, masks[-1]]) * coef_scale
    if mirror:
        y_all = -y_all

    cfg = GROUPS[group]
    started = False
    countdown_active = False
    countdown_value = 3
    countdown_start = 0
    start_time = None
    seq_pos = 0
    data = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button in JOY_START_BUTTONS and not started and not countdown_active:
                    countdown_active = True
                    countdown_start = time.time()
                    countdown_value = 3
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.fill(SCREEN_COLOR)

        if countdown_active:                       # 3-2-1 countdown
            if time.time() - countdown_start >= 1:
                if countdown_value > 1:
                    countdown_value -= 1
                    countdown_start = time.time()
                else:
                    countdown_active = False
                    started = True
                    seq_pos = 0
                    data.clear()
                    start_time = pygame.time.get_ticks()
            draw_target_and_cursor(mid_w, mid_h, lock_cursor)
            draw_centered(font_big, str(countdown_value), HEIGHT // 3)

        elif started and seq_pos < len(y_all):     # tracking
            color = RED
            if cues:
                if cfg["bars"]:
                    draw_direction_bars(segment_ids[1])
                if cfg["blue_beep"]:
                    if seq_pos in BEEP_FRAMES:
                        play_beep()
                    if CUE_START <= seq_pos <= CUE_END:
                        color = BLUE
            target_x = mid_w + int(y_all[seq_pos])
            cursor_x = draw_target_and_cursor(target_x, mid_h, lock_cursor, color)
            time_ms = pygame.time.get_ticks() - start_time
            data.append([seq_pos, time_ms, cursor_x, mid_h, target_x, mid_h, y_all[seq_pos]])
            seq_pos += 1

        elif started:                              # sequence finished
            time.sleep(0.5)
            running = False

        else:                                      # waiting for the start button
            draw_centered(font_small, "Appuie sur le bouton du joystick pour démarrer", HEIGHT // 3)
            draw_target_and_cursor(mid_w, mid_h, lock_cursor)

        pygame.display.flip()
        clock.tick(FPS)

    return pd.DataFrame(data, columns=["Frame", "Time_ms", "Cursor_X", "Cursor_Y",
                                       "Red_X", "Red_Y", "Seq_Value"])


# =============================================================================
# QUESTIONNAIRES
# =============================================================================
def number_entry(draw_screen, allowed_chars, is_valid):
    """Generic keyboard entry: type a number, validate with SPACE."""
    value = ""
    while True:
        screen.fill(SCREEN_COLOR)
        draw_screen()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_task()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if is_valid(value.strip()):
                        print("✅ Valeur saisie :", value)
                        return int(value)
                    print("⚠️ pas de chiffre avant de valider.")
                elif event.key == pygame.K_BACKSPACE:
                    value = value[:-1]
                elif event.key == pygame.K_ESCAPE:
                    quit_task()
                elif event.unicode and event.unicode in allowed_chars:
                    value += event.unicode
        draw_centered(font_big, value, HEIGHT * 2 / 3)
        pygame.display.flip()


def imagery_scale(keyword, sublabels):
    """7-point scale of visual or kinesthetic imagery difficulty."""
    def draw():
        pygame.draw.line(screen, "white", (WIDTH * 2 / 10, mid_h), (WIDTH * 8 / 10, mid_h), 3)
        draw_centered(font, "Estimez votre difficulté à", HEIGHT / 4)
        draw_centered(font_small, keyword, HEIGHT / 4 + 40, TEXT_IMPORTANT)
        draw_centered(font, "votre pouce bouger pendant l'imagination", HEIGHT / 4 + 80)
        draw_centered(font, "Entrez un chiffre entre 1 et 7 puis appuyez sur ESPACE", HEIGHT / 4 + 140)
        for i in range(7):
            x = WIDTH * (i + 2) / 10
            pygame.draw.circle(screen, TEXT_COLOR, (x, mid_h), VAS_RADIUS)
            pygame.draw.line(screen, TEXT_COLOR, (x - 1, mid_h), (x - 1, mid_h + 30), 2)
            for text, dy in ((str(i + 1), -40), (VAS_LABELS[i], 40), (sublabels[i], 60)):
                surface = font.render(text, True, TEXT_COLOR)
                screen.blit(surface, (x - surface.get_width() / 2, mid_h + dy))

    return number_entry(draw, "1234567", lambda v: v in list("1234567"))


def imagery_ratings():
    visual = imagery_scale("VOIR", VAS_LABELS_VISUAL)
    kinesthetic = imagery_scale("SENTIR", VAS_LABELS_KINESTHETIC)
    return visual, kinesthetic


def nasa_tlx():
    """NASA-TLX, 6 dimensions rated from 0 to 20."""
    scores = np.zeros(6)
    for i in range(6):
        def draw(i=i):
            pygame.draw.line(screen, "white", (nasa_start, mid_h), (nasa_end, mid_h), 3)
            draw_centered(font_small, NASA_DIMENSIONS[i], HEIGHT / 4 - 40, RED)
            draw_centered(font, NASA_QUESTIONS[i], HEIGHT / 4)
            draw_centered(font, "Entrez un chiffre entre 0 et 20 puis appuyez sur ESPACE", HEIGHT / 4 + 60)
            for place in range(21):
                x = nasa_start + place * nasa_step
                pygame.draw.line(screen, "white", (x - 1, mid_h), (x - 1, mid_h - 30), 2)
                surface = font.render(str(place), True, TEXT_COLOR)
                screen.blit(surface, (x - surface.get_width() / 2, mid_h + 20))
            for x in (nasa_start, mid_w, nasa_end):
                pygame.draw.line(screen, "white", (x - 1, mid_h), (x - 1, mid_h - 60), 2)
            low = font.render(NASA_LOW[i], True, TEXT_COLOR)
            screen.blit(low, (nasa_start - 5, mid_h + 50))
            high = font.render(NASA_HIGH[i], True, TEXT_COLOR)
            screen.blit(high, (nasa_end - high.get_width() + 5, mid_h + 50))

        scores[i] = number_entry(draw, "0123456789",
                                 lambda v: v.isdigit() and 0 <= int(v) <= 20)
    return scores


# =============================================================================
# SAVING
# =============================================================================
ORDER_COLUMNS = ["Essai", "Mask1", "Mask2", "S1", "S2", "S3", "Condition"]
NASA_COLUMNS = ["Essai", "EM", "EP", "ET", "Performance", "Effort", "Frustration"]


def new_output_file(session):
    """Excel file of the session; adds _1, _2... instead of overwriting an existing file."""
    path = participant_dir / f"{file_prefix}Session_{session}.xlsx"
    i = 1
    while path.exists():
        path = participant_dir / f"{file_prefix}Session_{session}_{i}.xlsx"
        i += 1
    return path


def order_row(trial, masks, segments, condition):
    return dict(zip(ORDER_COLUMNS, [trial, masks[0], masks[1], *segments[:3], condition]))


# =============================================================================
# SESSIONS - each returns True when completed, False if cancelled with Escape
# =============================================================================
def run_familiarisation():
    play_session_sound(0)
    focus_window()
    if not wait_for_button(FAM_RIGHT_LINES, on_escape="back"):
        return False

    path = new_output_file(0)
    writer = pd.ExcelWriter(path, engine="openpyxl")
    order = []
    try:
        for trial in range(1, N_TRIALS_FAM + 1):
            masks = mask_pool.draw(N_MASKS)
            segments = mask_pool.draw(N_SEGMENTS)   # familiarisation uses the mask bank only
            run_trial(segments, masks, mask_table).to_excel(
                writer, sheet_name=f"Familiarisation_rep{trial}", index=False)
            print(f"Essai : {trial} / {N_TRIALS_FAM}")
            order.append(order_row(trial, masks, segments, "Fam"))
            if trial == N_TRIALS_FAM // 2:          # switch to the left hand
                wait_for_button(FAM_LEFT_LINES)
    finally:
        pd.DataFrame(order, columns=ORDER_COLUMNS).to_excel(writer, sheet_name="Ordre", index=False)
        writer.close()
    print(f"✅ Export FAM terminé : {path}")
    return True


def run_test_session(session):
    """Pre-test, pre-transfer, post-test and post-transfer (which one depends on the
    session order): N_BLOCKS_TEST blocks of Repeated (Seg), Random (Alea) and Mirror (Mir)."""
    kind = SESSION_ORDERS[session_order][session]
    cfg = TEST_KINDS[kind]
    play_session_sound(session)
    focus_window()
    if not wait_for_button(test_lines(kind), on_escape="back"):
        return False

    path = new_output_file(session)
    writer = pd.ExcelWriter(path, engine="openpyxl")
    order = []
    trial = 1
    try:
        for _ in range(N_BLOCKS_TEST):
            for step in test_order:
                masks = mask_pool.draw(N_MASKS)
                if step == 1:
                    segments, mirror, cond = repeated_segments, False, "Seg"
                elif step == 2:
                    segments, mirror, cond = segment_pool.draw(N_SEGMENTS), False, "Alea"
                else:
                    segments, mirror, cond = repeated_segments, True, "Mir"
                condition = cfg["prefix"] + cond

                run_trial(segments, masks, segment_table, mirror=mirror).to_excel(
                    writer, sheet_name=f"{condition}{trial}", index=False)
                print(condition, f"{cfg['label']}: {trial} / {N_TEST_TRIALS}")
                order.append(order_row(trial, masks, segments, condition))
                trial += 1
    finally:
        pd.DataFrame(order, columns=ORDER_COLUMNS).to_excel(writer, sheet_name="Ordre", index=False)
        writer.close()
    print(f"✅ Export {cfg['export']} terminé : {path}")
    if PAUSE_AFTER_SESSION[session]:
        pause(PAUSE_SHORT_S)
    return True


def run_training():
    """Training on the repeated segment, by motor imagery (Implicit, Explicit,
    Explicit guided) or by observation (Control). The thumb does not move and the
    cursor stays on the target."""
    cfg = GROUPS[group]
    control = group == "Control"
    cues = cfg["blue_beep"] or cfg["bars"]

    play_session_sound(3)
    focus_window()
    if not wait_for_button(TRAINING_LINES[group], on_escape="back"):
        return False

    def reminder(trial):
        header = [[(f"Vous allez reprendre l'entrainement ({trial}/{N_TRIALS_TRAINING})", T)]]
        wait_for_button(header + REMINDER_LINES[group], on_escape="quit")

    path = new_output_file(3)
    writer = pd.ExcelWriter(path, engine="openpyxl")
    order, ratings, nasa = [], [], []
    try:
        for trial in range(1, N_TRIALS_TRAINING + 1):
            last_trial = trial == N_TRIALS_TRAINING
            masks = mask_pool.draw(N_MASKS)
            run_trial(repeated_segments, masks, segment_table,
                      lock_cursor=True, cues=cues).to_excel(
                writer, sheet_name=f"Training_{trial}", index=False)
            print(f"Essai : {trial} / {N_TRIALS_TRAINING}")

            questions = trial % cfg["question_every"] == 0
            if questions:
                if not control:            # imagery ratings
                    visual, kinesthetic = imagery_ratings()
                    ratings.append({"Essai": trial, "QualiteV": visual, "QualiteK": kinesthetic})
                scores = nasa_tlx()
                nasa.append(dict(zip(NASA_COLUMNS, [trial, *scores])))
                if control and not last_trial:
                    pause(PAUSE_SHORT_S)
                    reminder(trial)

            # Regular pause (Control: replaced by the questionnaire pause on that trial)
            if not last_trial and trial % PAUSE_EVERY == 0 and not (control and questions):
                pause(PAUSE_SHORT_S)
                reminder(trial)

            order.append(order_row(trial, masks, repeated_segments, "Training"))
    finally:
        if not control:
            pd.DataFrame(ratings, columns=["Essai", "QualiteV", "QualiteK"]).to_excel(
                writer, sheet_name="QV", index=False)
        pd.DataFrame(order, columns=ORDER_COLUMNS).to_excel(writer, sheet_name="Ordre", index=False)
        pd.DataFrame(nasa, columns=NASA_COLUMNS).to_excel(writer, sheet_name="NASA", index=False)
        writer.close()
    print(f"✅ Export ENTRAINEMENT terminé : {path}")
    pause(PAUSE_LONG_S)
    return True


def run_session(session):
    if session == 0:
        return run_familiarisation()
    if session == 3:
        return run_training()
    return run_test_session(session)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    segment_table = pd.read_excel(SEGMENT_FILE)
    mask_table = pd.read_excel(MASK_FILE)

    # --- Participant, group and training type ---
    participant_id = input("Entrez l'identifiant du participant : ").strip()
    print("GROUPE : 1 pour IMPLICIT; 2 pour EXPLICIT; 3 pour EXPLICIT GUIDED; 4 pour CONTROL")
    group = GROUP_MENU[ask_int("Numéro du groupe : ", range(1, 5))]
    print(f"Vous avez choisi le groupe {group}.")

    if group == "Control":
        participant_dir = DATA_DIR / f"{GROUPS[group]['folder']}_{participant_id}"
        file_prefix = participant_id + "Ctr"
    else:
        training_name = "GImagery"   # training by motor imagery only
        participant_dir = DATA_DIR / f"{GROUPS[group]['folder']}_{training_name}_{participant_id}"
        file_prefix = participant_id + training_name
    participant_dir.mkdir(parents=True, exist_ok=True)

    # Repeated segment: asked once, then reloaded at each run
    repeated_file = participant_dir / f"{file_prefix}_repete_segment.txt"
    saved = load_ints(repeated_file)
    if saved:
        repeated_idx = saved[0]
    else:
        print("🆕 Pas de fichier repete_segment pour ce participant")
        repeated_idx = ask_int("Numéro de séquence répété à utiliser : ")
        repeated_file.write_text(str(repeated_idx))
        print(f"✅ Séquence {repeated_idx} enregistrée comme segment répété.")
    repeated_segments = [repeated_idx] * N_SEGMENTS

    segment_pool = SegmentPool(len(segment_table),
                               participant_dir / f"{file_prefix}_used_segments.txt",
                               excluded=[repeated_idx])
    mask_pool = SegmentPool(len(mask_table), participant_dir / f"{file_prefix}_mask_segments")

    # Order of conditions in test blocks (1 = Repeated, 2 = Random, 3 = Mirror)
    order_file = participant_dir / f"{file_prefix}_ordre_test"
    test_order = load_ints(order_file)
    if test_order:
        print("ordre chargé: ", test_order)
    else:
        default_order = GROUPS[group]["default_order"]
        if default_order is None:
            test_order = [ask_int(f"Entrez l'ordre chiffre {k}:") for k in (1, 2, 3)]
        else:
            test_order = list(default_order)
            if RANDOMIZE_TEST_ORDER:
                random.shuffle(test_order)
        order_file.write_text(" ".join(map(str, test_order)))
        print("🆕 Pas de fichier de randomisation pour ce participant, "
              "création du fichier avec l'ordre : ", test_order)

    # Session order: test (right hand) or transfer (left hand) first.
    # Chosen once, then reloaded; the same order is used for pre- and post-tests.
    session_order_file = participant_dir / f"{file_prefix}_ordre_sessions"
    saved = load_ints(session_order_file)
    if saved:
        session_order = "test_first" if saved[0] == 0 else "transfer_first"
        print("ordre des sessions chargé : ", session_order)
    else:
        print("ORDRE DES SESSIONS : 0 pour commencer par le TEST (main droite); "
              "1 pour commencer par le TRANSFERT (main gauche)")
        choice = ask_int("Ordre des sessions : ", range(0, 2))
        session_order = "test_first" if choice == 0 else "transfer_first"
        session_order_file.write_text(str(choice))
    print(", ".join(f"Session {n} = {k}" for n, k in SESSION_ORDERS[session_order].items()))

    # --- Window on the chosen monitor ---
    monitors = get_monitors()
    monitor_index = MONITOR_INDEX if len(monitors) > 1 else 0
    monitor = monitors[monitor_index]
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{monitor.x},{monitor.y}"

    pygame.init()
    WIDTH, HEIGHT = monitor.width, monitor.height
    mid_w, mid_h = WIDTH / 2, HEIGHT / 2
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)
    clock = pygame.time.Clock()
    pygame.mouse.set_visible(False)

    font_big = pygame.font.SysFont(None, 100)
    font_small = pygame.font.SysFont(None, 40)
    font_number = pygame.font.SysFont(None, 60)
    font = pygame.font.Font("freesansbold.ttf", 20)

    px_per_cm = WIDTH / SCREEN_WIDTH_CM
    nasa_start = mid_w - NASA_SCALE_CM / 2 * px_per_cm
    nasa_end = mid_w + NASA_SCALE_CM / 2 * px_per_cm
    nasa_step = px_per_cm

    # --- Joystick (mouse used as fallback) ---
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"✅ Joystick détecté : {joystick.get_name()}")
    else:
        print("⚠️ Aucun joystick détecté !!! La souris devient le curseur!!")
        joystick = None

    # Cursor range on screen (JOY_RANGE_CM) and scaling of the segments to the screen
    half_range = WIDTH * (JOY_RANGE_CM * JOY_SCALE) / SCREEN_WIDTH_CM / 2
    cursor_left, cursor_right = mid_w - half_range, mid_w + half_range
    coef_scale = (WIDTH * (JOY_RANGE_CM * JOY_SCALE - 1) / SCREEN_WIDTH_CM / 2) / MAX_CURVE

    # --- Session loop ---
    while True:
        session = ask_session()
        if session == -1:
            print("Fin du programme.")
            break
        # Run the chosen session, then the following ones, until session 5 or Escape
        while session <= 5 and run_session(session):
            session += 1

    pygame.quit()