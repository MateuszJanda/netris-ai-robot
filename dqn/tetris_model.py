#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""


import numpy as np
import config


class TetrisModel:
    COLOR_TO_PIECE = {
        -1: 4,      # Piece Id: 11
        -2: 0,      # Piece Id: 0
        -3: 1,      # Piece Id: 2
        -4: 2,      # Piece Id: 3
        -5: 3,      # Piece Id: 7
        -6: 5,      # Piece Id: 15
        -7: 6,      # Piece Id: 17
    }

    PIECE_TO_PIECE_ID = {
        4: 11,
        0: 0,
        1: 2,
        2: 3,
        3: 7,
        5: 15,
        6: 17,
    }

    PIECE_ID_TO_NAME = {
        11: "white pyramid",
         0: "blue log",
         2: "violet square",
         3: "azure L",
         7: "yellow mirror L",
        15: "green S",
        17: "red Z",
    }

    def __init__(self):
        self._last_round = False
        self._reward = 0
        self._new_piece = 0

        self._board = np.zeros(shape=(config.BOARD_HEIGHT, config.BOARD_WIDTH), dtype=int)


    def parse(self, msg):
        last_round, reward, new_piece, *state = msg

        self._last_round = True if int(last_round) else False
        self._reward = float(reward)
        self._board = np.array([float(val) for val in state])
