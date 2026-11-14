# PursuitTracking_WP1
Article : Augmenting explicit sequence awareness during motor imagery practice promotes effector-independent consolidation 

Code and data of a study on motor sequence learning through motor imagery, using a pursuit tracking task.

Participants track a target moving horizontally on the screen with a joystick. Each trial contains three 4-s segments: the participant's repeated segment, its mirror, or random segments. Performance is measured as the tracking error (RMSE) between the cursor and the target, before training, after training and 24 h later, with the trained (right) hand and the untrained (left) hand. An explicit recognition test is run after training.

Main experiment: training by motor imagery in three groups: implicit (IMP), explicit (EXP) and explicit guided (EXP+ / EXPCUE).
Control experiment: explicit guided imagery (EXP+) vs observation of the target without movement (CTR).

60 participants, 15 per group.

--------
Repository structure
├── 01_SegmentCreation/      Generation of the target trajectories (stimuli)
├── 02_Task/                 Pursuit tracking task (experiment script)
├── 03_ExtractionData/       Extraction of the variables from the raw files
├── 04_AllData/              Anonymised data used for the statistics
├── 05_StatisticalAnalysis/  Statistical analyses
└── README.md

--------
Requirements
Python ≥ 3.8
numpy, pandas, scipy, openpyxl
Task: pygame, screeninfo (+ optional pygetwindow, Windows) – see 02_Task
R ≥ 4.0 with readxl, tidyverse (dplyr, tidyr, ggplot2), lme4, lmerTest, emmeans, performance, effectsize, permuco, rstatix, openxlsx

==========
01_SegmentCreation
==========
File	Content:
creationsegment (3H).py:	Generates the 55 matched segments (repeated and random segments of the task)
creationmask (3H).py:	Generates the masks (start/end of each trial and familiarisation)
segments_3H_SEG.xlsx:	Parameters of the 55 segments used in the experiment
segments_3H_MASK.xlsx:	Parameters of the 350 masks used in the experiment

How the segments are built:
Each segment is 240 samples (4 s at 60 Hz) built from a 3-harmonic Fourier series (coefficients drawn uniformly in [−5, 5]), centred on 0 and phase-shifted to start at its first zero-crossing. Velocity, acceleration and jerk are computed analytically. All kept segments stay within [−10, 10] and have exactly 4 changes of direction.

Segments (creationsegment (3H).py): from a pool of 1,000,000 segments, a target segment is chosen (path length closest to half the maximum length, extrema values at least 2 units apart). The segments kept match the target: path length within 1 %, mean velocity and mean amplitude within 5 %.
Masks (creationmask (3H).py): segments meeting the limit and direction-change criteria only.

After selection, each segment is rebuilt, a 20-sample raised-cosine fade-in/fade-out is applied, and the extrema of the faded segment are stored (DirFade1..4).

Parameter files (segments_3H_SEG.xlsx, segments_3H_MASK.xlsx)
One row per segment:
b0:	Offset (centres the segment on 0)
a1–a3, b1–b3:	Sine / cosine Fourier coefficients (harmonics 1–3)
Phi:	Phase shift (samples)
length:	Path length of the segment
Max, Min:	Maximum / minimum position
Mean_Velocity, Mean_Acceleration:	Mean absolute velocity / acceleration
Mean_Jerk:	Mean squared jerk
Mean_Amplitude:	Mean amplitude between successive extrema
Dir1–Dir4:	Position of the 4 extrema (changes of direction)
DirFade1–DirFade4:	Position of the extrema after fade-in/fade-out

A segment can be rebuilt as:
x(t) = b0 + Σₙ₌₁³ [aₙ sin(n·q) + bₙ cos(n·q)], q = 2π (t + Phi) / 240, t = 0…239

Each script writes the parameter file and a visualisation_*.xlsx file (trajectories, one column per segment, 240 rows). Segments are drawn at random (no fixed seed), so a new run produces new sequences: the .xlsx files in this folder are the ones used in the experiment (a copy is in 02_Task/FileTask/).

==========
02_Task
==========
File	Content
PursuitTrackingTask.py:	Pursuit tracking task (all groups and sessions)
Recognitiontest.py:	Explicit recognition test (run the day after training)
FileTask/segments_3H_SEG.xlsx:	Segments (copy of 01_SegmentCreation)
FileTask/segments_3H_MASK.xlsx:	Masks (copy of 01_SegmentCreation)
FileTask/bip.wav:	Beep at the start of each repeated segment (training)
FileTask/session0.wav … session5.wav:	Sound played at the start of each session

Additional requirements: pygame, screeninfo; optional, Windows only: pygetwindow (window focus) and winsound (sounds, built into Python on Windows). The task runs without them, but without sounds. A joystick is expected; for testing without a joystick, the mouse controls the cursor and the SPACE key replaces the joystick button.

Screen, joystick, timing and design parameters are set at the top of PursuitTrackingTask.py; Recognitiontest.py imports them, so both scripts always use the same settings. Participant-facing instructions are in French.

-----------
Pursuit tracking task (PursuitTrackingTask.py)
-----------
The participant tracks a red target moving horizontally with a black cursor controlled by the thumb on a joystick. Each trial (60 Hz, 1200 samples = 20 s) is built as:

mask (0–239) → segment 1 (240–479) → segment 2 (480–719) → segment 3 (720–959) → mask (960–1199)

Each segment / mask starts and ends at 0 (20-sample raised-cosine fade-in/fade-out).

Groups (chosen at start-up):
1	Implicit:	Motor imagery, no cue
2	Explicit:	Motor imagery; target turns blue + beep at each repeated segment
3	Explicit+:	As Explicit + numbered bars at the direction changes of the repeated segment
4	Control:	Observation (eye tracking of the target, no movement), with the guided cues

During training (groups 1–3: motor imagery; group 4: observation) the thumb does not move and the cursor stays locked on the target.

Sessions:
0 :	Familiarisation :	4 trials (2 right hand, 2 left hand), masks only
1	: Pre-test : Right hand, 3 blocks of Repeated / Random / Mirror (9 trials)
2	: Pre-transfer	: Left hand, same structure
3	: Training : 60 trials of the repeated segment, pause every 10 trials, questionnaires
4 :	Post-test :	Right hand, as session 1
5	: Post-transfer	: Left hand, as session 2

Repeated: the participant's repeated segment, 3 times; 
Mirror: the same, mirrored (−x); 
Random: 3 segments drawn from segments_3H_SEG.xlsx, never re-used for the same participant and never the repeated one. 
Masks are drawn from segments_3H_MASK.xlsx, also without re-use.
The order of Repeated / Random / Mirror within blocks and whether the right- or the left-hand test comes first are set once per participant (and re-used in the post-tests).

Questionnaires at the end of training: 
imagery quality (visual and kinesthetic, 7-point scales; groups 1–3) and NASA-TLX (6 dimensions, 0–20).

After a session, the next one starts automatically. The task can be stopped and resumed: the repeated segment, the segments already used and the orders are saved in the participant's folder.

Start-up (console): 
participant ID, group, number of the repeated segment (first run only), session order (first run only), then the session to start with.
**Note that participant ID was between 100-199 for Implicit group, 200-299 for Explicit group, 300-399 for Explicit+ group, and 400-499 for Control group (this is used to be extract later).**

Output (02_Task/data/<GroupFolder>_<ID>/): one Excel file per session, <ID>…Session_<n>.xlsx (a new run adds _1, _2… instead of overwriting), with:
- one sheet per trial (e.g. Familiarisation_rep1, Pre_Seg1, Pre_TAlea2, Post_Mir3, Training_1) with columns Frame, Time_ms, Cursor_X, Cursor_Y, Red_X, Red_Y, Seq_Value;
- Ordre: masks and segments used in each trial;
- QV (imagery quality) and NASA (NASA-TLX), training session only.

--------
Recognition test (Recognitiontest.py)
--------
Three segments are presented in random order: the participant's repeated segment, its mirror, and one random segment. After each one, the participant rates from 1 ("Not at all") to 9 ("Absolutely") whether it corresponds to the sequence repeated during training. Session 0 = visual (the participant only watches), then session 1 = physical (the participant tracks the target with the right hand), with the same segments in a new order.

Output <ID>…_Recognitiontest.xlsx in the participant's folder:
sheets Recognitiontest0 and Recognitiontest1, columns S (segment shown), mirror (yes/no), RS (repeated segment), Response (1–9).

=================
03_ExtractionData
=================
Scripts that read the raw files created by 02_Task and gather the variables of all participants in one table. Before running, set folder_path at the top of each script to the folder containing the raw data. Outputs are saved in the current working directory.

Extract_RMSE.py	: Tracking error between cursor and target, per segment	: RMSE_alldata.xlsx
Extract_RecognitionTest.py	: Ratings of the recognition test :	Recognition_data.xlsx
Extract_Questionnaire.py : Questionnaires of the training session (imagery quality + NASA-TLX) :	Questionnaires_data.xlsx

Participant ID: the first 3 characters of each file name; the first digit gives the group: 1 = IMP (implicit), 2 = EXP (explicit), 3 = EXPCUE / EXP+ (explicit guided), 4 = CTR (control).

------------
RMSE (Extract_RMSE.py)
------------
Reads every sheet of the test sessions (familiarisation and training sheets are ignored).
Files whose path contains 4_1 or 5_1 (second run of sessions 4 and 5, the next day) are coded as the 24-h retention test (Post24).
Condition coded from the sheet name: Pre / Post → Session; Seg / Mir / Alea → Sequence (Repete, Mirror, Alea); _T → Hand = Transfer (otherwise Main).
The 3 segments of each trial are extracted, without the fade-in/fade-out: samples 260–459, 500–699, 740–939 (columns Cursor_X and Red_X).
Cursor low-pass filtered: Butterworth, 3rd order, 14 Hz cutoff, zero-phase (filtfilt).
RMSE = √mean((cursor_filtered − target)²), in pixels.
Output: one row per participant × trial × segment, columns ID, Group, Feuille (sheet), Session, Sequence, Hand, RMSE.

------------
Recognition test (Extract_RecognitionTest.py)
------------
Reads the Recognitiontest files (columns S, mirror, RS, Response) and returns, for each file and sheet (Recognitiontest0 = visual, Recognitiontest1 = visuo-motor), the rating given to:

Rsequence	: Repeated	: S == RS and mirror == no
Rmirror	: Mirror	: S == RS and mirror == yes
Ralea	: Random	: S ≠ RS and mirror == no

------------
Questionnaires (Extract_Questionnaire.py)
------------
Gathers the QV (imagery quality) and NASA (NASA-TLX) sheets of all training files (Session_3) into one Excel file (one sheet each), with a Fichier column giving the source file.

=================
04_AllData
=================
Anonymised data of the 60 participants (15 per group), used for the statistical analyses.

File	Content
01_RMSE_alldata.xlsx : RMSE per participant × trial × segment (9,720 rows)
02_Recognition_data.xlsx	: Recognition ratings, one row per participant
03_Questionnaires_data.csv	: NASA-TLX and imagery quality, one row per participant (; separator)

------------
01_RMSE_alldata.xlsx
------------
Output of Extract_RMSE.py: 60 participants × 3 sessions × 2 hands × 9 trials (3 Repeated, 3 Mirror, 3 Random) × 3 segments.


ID	Participant : (101–115, 201–215, 301–315, 401–415)
Group	: IMP, EXP, EXPCUE, CTR
Feuille	: Condition of the trial (sheet name)
Session	: 01Pre, 02Post, 03Post24 (24-h retention)
Sequence : Repete (repeated), Mirror, Alea (random)
Hand	: Main (trained, right hand), Transfer (left hand)
RMSE :	Tracking error (pixels)

------------
02_Recognition_data.xlsx
------------
Two sheets: Recognition main (IMP, EXP, EXP+; 45 participants) and Recognition control (EXP+, CTR; 30 participants).


ID, Grp	: Participant and group
V_sequence, V_mirror, V_random	: Visual recognition, rating (1–9) of the repeated, mirror and random segments
VM_sequence, VM_mirror, VM_random	: Visuo-motor recognition, same, while tracking the target

------------
03_Questionnaires_data.csv
------------

Grp, ID	: Group and participant
Mental demand … Frustration :	NASA-TLX dimensions (0–100)
IMQuality_Visual, IMQuality_Kinesthetic	: Imagery quality (1–7); empty for CTR (no imagery)

======================
05_StatisticalAnalysis
======================
R scripts using the files of 04_AllData. Set the path to the data file at the top of each script (section 2) before running.

The main experiment compares the three imagery groups (IMP, EXP, EXP+); the control experiment compares the explicit guided imagery group (EXP+) with the observation group (CTR).

Script	Data	Analysis
LMM - RMSE - MainExperiment.R :	01_RMSE_alldata.xlsx (CTR excluded) :	Linear mixed model
LMM - RMSE - Control.R	: 01_RMSE_alldata.xlsx (IMP and EXP excluded)	: Linear mixed model
PermutANOVA - RecognitionTest.R	: 02_Recognition_data.xlsx	: Permutation ANOVA + non-parametric post-hocs

------------
RMSE: linear mixed models
------------
Outliers removed: trials beyond ±2.5 SD of the participant's mean in each Session × Hand × Sequence cell (cleaned data saved as RMSE_withoutoutlier.xlsx).
RMSE log-transformed (distribution checked with histograms, Q-Q plots and performance::check_distribution).
Model: log_RMSE ~ Group * Hand * Sequence * Session + (1 | ID) (lme4, REML, bobyqa optimizer); ANOVA table with Satterthwaite degrees of freedom (lmerTest).
Post-hocs (emmeans, Bonferroni): groups, sessions and sequences compared within each level of the other factors (Group × Session × Sequence); main experiment only: Hand × Session. For each contrast: estimate (log scale), 95 % CI, adjusted p-value, RMSE ratio (back-transformed) with CI, and Cohen's d with CI.
Output: Results_MainAnalyses.xlsx / Results_ControlAnalyses.xlsx.

------------
Recognition test: permutation ANOVA
------------
The script runs the visual recognition test of the main experiment. For the visuo-motor test, replace V_ by VM_ in sections 3 and 4; for the control experiment, use the Recognition control sheet (section 2).

Permutation ANOVA (permuco::aovperm, 10,000 permutations): Score ~ Grp * Temps + Error(ID / Temps), where Temps = repeated / mirror / random segment; partial η² computed from F and degrees of freedom.
Post-hocs (rstatix, Bonferroni): Mann-Whitney tests between groups for each segment type; Wilcoxon signed-rank tests between segment types within each group and with all groups pooled.
Effect sizes: rank-biserial correlation with 95 % percentile bootstrap CI (5,000 resamples).
Output: Resultats_Permut.xlsx (ANOVA, post-hocs and effect sizes, one sheet each).
