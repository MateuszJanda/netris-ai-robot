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
        self._lines_cleared = 0
        self._new_piece = 0

        self._board = None


    def parse(self, msg):
        """
        Parse message, extract if this is last round, lines cleared, and current
        board state.
        """
        last_round, lines_cleared, new_piece, *board = msg

        self._last_round = True if int(last_round) else False
        self._lines_cleared = float(lines_cleared)
        self._board = np.array([float(block_color) for block_color in board])

    def last_round(self):
        """
        Return if game is over.
        """
        self._last_round

    def reward(self):
        """
        Return reward, lines_cleared**2.
        """
        self._lines_cleared**2

    def board(self):
        """
        Create flat board with four blocks representing pieces.
        """
        out_board = np.copy(self.model._board)

        # Normalize board
        out_board = out_board / len(COLOR_TO_PIECE)

        # In top row set four middle block as new piece, and erase all other
        for x in range(config.BOARD_WIDTH):
            if 3 < x < 8:
                out_board[0][x] = self._new_piece / len(COLOR_TO_PIECE)
            else:
                out_board[0][x] = config.EMPTY_BLOCK

        return out_board
