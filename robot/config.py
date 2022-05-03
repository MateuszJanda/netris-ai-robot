#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

# Server settings
HOST = "127.0.0.1"
PORT = 9800

# Netris/environment parameters
BOARD_HEIGHT = 20
BOARD_WIDTH = 10
NUM_OF_PIECES = 7
ACTION_SPACE_SIZE = 4 * 10
SHFIT_OFFSET = 5

SCREEN_ID = 0
TOP_LINE = 19
BOTTOM_LINE = 0
EMPTY_BLOCK = 0
FULL_BLOCK = 1
EMPTY_LINE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Snapshot settings
SNAPSHOT_MODULO = 50
PRINT_BOARD_MODULO = 100
MODEL_SNAPSHOT = "%05d_model.h5"
DATA_SNAPSHOT = "%05d_data.pickle"

# Stats settings
STATS_FILE = "stats.txt"

# DQN parameters
DISCOUNT = 0.95                               # Gamma (𝛾) parameter for Bellman equation
MINIBATCH_SIZE = 128                          # How many steps (samples) to use for training
REPLAY_MEMORY_SIZE = 20_000                   # Last steps kept for model training
MIN_REPLAY_MEMORY_SIZE = 10 * MINIBATCH_SIZE  # Minimum number of steps in a memory to start training
EPISODES = 50_000                             # Episodes == full games


# Custom reward settings
MAX_PIECE_HEIGHT = 4
MAX_ALLOWED_GAPS = 5
MAX_CLEARED_LINES = 4
REWAD_THRESHOLDS = MAX_CLEARED_LINES + (MAX_CLEARED_LINES - 1)
