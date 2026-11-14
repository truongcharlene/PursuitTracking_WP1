"""
Recognition Test
-------------

Explicit recognition test run after the Pursuit Tracking Task (the day after training).

Three single segments are presented in random order:
  - the participant's repeated segment,
  - the mirror of the repeated segment,
  - one random segment (never the repeated one).
After each segment, the participant rates on a 1-9 scale whether it corresponds to the
sequence repeated during training (1 = "Pas du tout", 9 = "Tout à fait").

Sessions
--------
 0  Visual recognition     the participant only watches the target
 1  Physical recognition   the participant tracks the target with the joystick (right hand)
-1  Quit

Session 0 is followed by session 1. The same three segments are used in both sessions,
in a new random order.

The participant's repeated segment is read from the folder created by
pursuit_tracking_task.py, and all display parameters are imported from it, so both
scripts always use the same screen settings.

Participant-facing instructions are in French.

-------------
Author: Charlène Truong - Brain, Action and Skill Lab, UCLouvain
"""

import os
import random
import sys
import time

import numpy as np
import pandas as pd
import pygame
from screeninfo import get_monitors

# Shared parameters and functions (same folder)
from PursuitTrackingTask import (
    CURSOR_COLOR, CURSOR_RADIUS, CURSOR_WIDTH, DATA_DIR, FPS, GROUP_MENU, GROUPS,
    JOY_AXIS, JOY_RANGE_CM, JOY_SCALE, JOY_START_BUTTONS, MAX_CURVE, MONITOR_INDEX,
    RED, SCREEN_COLOR, SCREEN_WIDTH_CM, SEGMENT_FILE, TARGET_RADIUS, TEXT_COLOR,
    TEXT_IMPORTANT, WINDOW_TITLE, ask_int, load_ints, make_segment,
)

try:
    import pygetwindow as gw
except ImportError:
    gw = None


# =============================================================================
# PARAMETERS
# =============================================================================
N_RANDOM_SEGMENTS = 1      # number of new (random) segments presented
SCALE_MAX = 9              # rating scale from 1 to SCALE_MAX


# =============================================================================
# INSTRUCTIONS (French)
# =============================================================================
T, I = TEXT_COLOR, TEXT_IMPORTANT
BLANK = [(" ", T)]

SESSION_NAMES = {0: "recognition visuel", 1: "recognition physique"}

INSTRUCTIONS = {
    0: [
        [("Consigne:", T)], BLANK,
        [("Vous allez voir trois séquences s'afficher.", T)], BLANK,
        [("Observez", I), (" attentivement chaque séquence.", T)],
        [("A la fin de chaque présentation, indiquez sur une échelle de 1 à 9,", T)],
        [("si elle correspond à la séquence répétée lors de l’entraînement d’hier.", T)],
        BLANK, BLANK,
        [("Appuyez sur le bouton du joystick pour commencer", T)],
    ],
    1: [
        [("Consigne:", T)], BLANK,
        [("Vous allez voir trois séquences s'afficher.", T)], BLANK,
        [("Utilisez la main ", T), ("DROITE", I), (" pour suivre la cible rouge", T)],
        [("en BOUGEANT votre pouce, représenté par un cercle noir.", T)], BLANK,
        [("A la fin de chaque présentation, indiquez sur une échelle de 1 à 9,", T)],
        [("si elle correspond à la séquence répétée lors de l’entraînement d’hier.", T)],
        BLANK, BLANK,
        [("Appuyez sur le bouton du joystick pour commencer", T)],
    ],
}
WAITING_TEXT = {
    0: "Appuie sur le bouton du joystick pour voir la séquence",
    1: "Appuie sur le bouton du joystick pour realiser la séquence",
}


# =============================================================================
# DISPLAY
# =============================================================================
def quit_task():
    pygame.quit()
    sys.exit()


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


def wait_for_button(lines):
    """Show instructions until a joystick button is pressed (Escape = back to menu)."""
    while True:
        show_lines(lines)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_task()
            elif event.type == pygame.JOYBUTTONDOWN and event.button in JOY_START_BUTTONS:
                return True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False


def draw_cursor(y):
    if joystick:
        axis = joystick.get_axis(JOY_AXIS)
        x = int(cursor_left + (axis + 1) / 2 * (cursor_right - cursor_left))
    else:
        x = pygame.mouse.get_pos()[0]
    pygame.draw.circle(screen, CURSOR_COLOR, (x, y), CURSOR_RADIUS, CURSOR_WIDTH)


# =============================================================================
# TRIAL AND RATING
# =============================================================================
def run_segment(idx, mirror, session):
    """Present one segment. The cursor is shown only in the physical session (1)."""
    y_all = make_segment(idx, segment_table) * coef_scale
    if mirror:
        y_all = -y_all
    show_cursor = session == 1

    started = False
    countdown_active = False
    countdown_value = 3
    countdown_start = 0
    seq_pos = 0

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
            pygame.draw.circle(screen, RED, (mid_w, mid_h), TARGET_RADIUS)
            draw_centered(font_big, str(countdown_value), HEIGHT // 3)
            if show_cursor:
                draw_cursor(mid_h)

        elif started and seq_pos < len(y_all):     # segment
            pygame.draw.circle(screen, RED, (mid_w + int(y_all[seq_pos]), mid_h), TARGET_RADIUS)
            if show_cursor:
                draw_cursor(mid_h)
            seq_pos += 1

        elif started:                              # segment finished
            time.sleep(0.5)
            running = False

        else:                                      # waiting for the start button
            draw_centered(font_small, WAITING_TEXT[session], HEIGHT // 3)
            pygame.draw.circle(screen, RED, (mid_w, mid_h), TARGET_RADIUS)
            if show_cursor:
                draw_cursor(mid_h)

        pygame.display.flip()
        clock.tick(FPS)


def recognition_rating():
    """1-9 rating: does this segment correspond to the repeated sequence?"""
    valid = [str(v) for v in range(1, SCALE_MAX + 1)]
    value = ""
    while True:
        screen.fill(SCREEN_COLOR)
        draw_centered(font, "Estimez si cette séquence correspond à la séquence répétée", HEIGHT / 4)
        draw_centered(font, f"Entrez un chiffre entre 1 et {SCALE_MAX}. ", HEIGHT / 4 + 40, TEXT_IMPORTANT)

        left, right = WIDTH * 2 / 12, WIDTH * 10 / 12
        pygame.draw.line(screen, "white", (left, mid_h), (right, mid_h), 3)
        for k in range(SCALE_MAX):
            x = WIDTH * (k + 2) / 12
            pygame.draw.circle(screen, "white", (x, mid_h), 5)
            label = font.render(str(k + 1), True, TEXT_COLOR)
            screen.blit(label, (x - label.get_width() / 2, mid_h + 20))
        screen.blit(font.render("Pas du tout", True, TEXT_COLOR), (left, mid_h - 50))
        high = font.render("Tout à fait", True, TEXT_COLOR)
        screen.blit(high, (right - high.get_width(), mid_h - 50))
        draw_centered(font, "Appuyez sur ESPACE pour valider votre réponse", HEIGHT * 3 / 4)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_task()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if value.strip() in valid:
                        print("✅ Valeur saisie :", value)
                        return int(value)
                    print("⚠️ pas de chiffre avant de valider.")
                elif event.key == pygame.K_BACKSPACE:
                    value = value[:-1]
                elif event.key == pygame.K_ESCAPE:
                    quit_task()
                elif event.unicode and event.unicode in "".join(valid):
                    value += event.unicode

        draw_centered(font_big, value, HEIGHT * 2 / 3)
        pygame.display.flip()


def run_session(session, segments, writer):
    """Present the three segments in random order and collect the ratings.
    Returns False if cancelled with Escape on the instruction screen."""
    if not wait_for_button(INSTRUCTIONS[session]):
        return False
    random.shuffle(segments)
    print(segments)
    rows = []
    for idx, mirror in segments:
        run_segment(idx, mirror, session)
        response = recognition_rating()
        rows.append({"S": idx, "mirror": "yes" if mirror else "no",
                     "RS": repeated_idx, "Response": response})
        print(idx, "mirror" if mirror else "", repeated_idx, response)
    pd.DataFrame(rows, columns=["S", "mirror", "RS", "Response"]).to_excel(
        writer, sheet_name=f"Recognitiontest{session}", index=False)
    return True


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    segment_table = pd.read_excel(SEGMENT_FILE)

    # --- Participant (same folder as the tracking task) ---
    participant_id = input("Entrez l'identifiant du participant : ").strip()
    print("GROUPE : 1 pour IMPLICIT; 2 pour EXPLICIT; 3 pour EXPLICIT GUIDED; 4 pour CONTROL")
    group = GROUP_MENU[ask_int("Numéro du groupe : ", range(1, 5))]
    print(f"Vous avez choisi le groupe {group}.")

    if group == "Control":
        participant_dir = DATA_DIR / f"{GROUPS[group]['folder']}_{participant_id}"
        file_prefix = participant_id + "Ctr"
        output_prefix = participant_id
    else:
        participant_dir = DATA_DIR / f"{GROUPS[group]['folder']}_GImagery_{participant_id}"
        file_prefix = output_prefix = participant_id + "GImagery"

    saved = load_ints(participant_dir / f"{file_prefix}_repete_segment.txt")
    if not saved:
        print(f"❌ Pas de segment répété trouvé dans {participant_dir}.")
        print("Vérifiez l'identifiant et le groupe (le participant doit avoir fait la tâche de tracking).")
        sys.exit()
    repeated_idx = saved[0]
    print(f"Segment répété : {repeated_idx}")

    print("CHOIX DE SESSION:  0 pour recognition visuel; 1 pour recognition physique; -1 pour stop")
    session = ask_int("Numéro de session : ", range(-1, 2))
    if session == -1:
        print("Fin du programme.")
        sys.exit()
    print(f"Vous avez choisi la session {session}.")

    # One random segment (never the repeated one) + repeated + mirror of the repeated
    candidates = sorted(set(range(1, len(segment_table) + 1)) - {repeated_idx})
    segments = [(idx, False) for idx in random.sample(candidates, N_RANDOM_SEGMENTS)]
    segments += [(repeated_idx, False), (repeated_idx, True)]

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
    font = pygame.font.Font("freesansbold.ttf", 20)

    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"✅ Joystick détecté : {joystick.get_name()}")
    else:
        print("⚠️ Aucun joystick détecté !!! La souris devient le curseur!!")
        joystick = None

    half_range = WIDTH * (JOY_RANGE_CM * JOY_SCALE) / SCREEN_WIDTH_CM / 2
    cursor_left, cursor_right = mid_w - half_range, mid_w + half_range
    coef_scale = (WIDTH * (JOY_RANGE_CM * JOY_SCALE - 1) / SCREEN_WIDTH_CM / 2) / MAX_CURVE

    # --- Output file (never overwritten) ---
    path = participant_dir / f"{output_prefix}_Recognitiontest.xlsx"
    i = 1
    while path.exists():
        path = participant_dir / f"{output_prefix}_Recognitiontest_{i}.xlsx"
        i += 1

    time.sleep(0.5)
    focus_window()

    # --- Sessions: visual (0) then physical (1) ---
    writer = pd.ExcelWriter(path, engine="openpyxl")
    done = 0
    try:
        while session <= 1 and run_session(session, segments, writer):
            done += 1
            session += 1
    finally:
        if done:
            writer.close()
            print(f"✅ Export Recognitiontest terminé : {path}")
        else:
            print("Aucune session réalisée, pas de fichier créé.")
    pygame.quit()