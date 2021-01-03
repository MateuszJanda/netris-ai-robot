#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

HOST = "127.0.0.1"
PORT = 9800

# Netris/environment parameters
BOARD_HEIGHT = 20
BOARD_WIDTH = 10
ACTION_SPACE_SIZE = 4 * 10
SHFIT_OFFSET = 5

SCREEN_ID = 0

# Snapshot settings
SNAPSHOT_MODULO = 50
MODEL_SNAPSHOT = "%05d_model.h5"
DATA_SNAPSHOT = "%05d_data.pickle"
STATS_FILE = "stats.txt"


# DQN parameters
DISCOUNT = 0.95                               # Gamma (𝛾) parameter from Bellman equation
MINIBATCH_SIZE = 128                          # How many steps (samples) to use for training
REPLAY_MEMORY_SIZE = 40_000                   # Last steps kept for model training
MIN_REPLAY_MEMORY_SIZE = 10 * MINIBATCH_SIZE  # Minimum number of steps in a memory to start training
EPISODES = 20_000                             # Episodes == full games

# Exploration settings - try/explore random action with probability epsilon
EPSILON_DECAY = 0.99995                       # Decay epsilon. Smarter NN is, then less random action should be taken
MIN_EPSILON = 0.02                            # Epsilon shouldn't less than this. We always want to check something new
