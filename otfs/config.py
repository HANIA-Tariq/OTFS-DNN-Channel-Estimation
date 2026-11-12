"""System parameters (Section IV of the paper)."""
import numpy as np

# Step 1: OTFS frame
M = 64                      # delay bins (subcarriers)
N = 8                       # Doppler bins (OTFS symbols)
MN = M * N                  # samples per frame
DF = 15e3                   # subcarrier spacing [Hz]
FC = 3e9                    # carrier frequency [Hz]
C = 3e8                     # speed of light [m/s]

# Step 2: channel limits
LM = 4                      # max delay tap: 4 us / (1 / (M * DF)) = 3.84 -> 4
L = LM + 1                  # possible delay taps 0..LM
MAX_SPEED_KMH = 1000        # max terminal speed

# Step 3: pilot and guard layout (Fig. 2)
LZP = 2 * LM + 1            # zero-padding (guard) rows = 9
MD = M - LZP                # data rows 0..54
LP = M - 1 - LM             # pilot delay row = 59
KP = N // 2                 # pilot Doppler column = 4

# Step 4: modulation (4-QAM, unit energy, Gray mapped)
QAM = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)

# Step 5: DNN training
TRAIN_SNR_DB = 10           # paper: best training SNR = 10 dB
HIDDEN = (256, 128)         # Q1, Q2 (Fig. 3)
