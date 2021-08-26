#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""


import numpy as np
from robot import config


class TetrisData:
    NUM_OF_COLORS = 7

    def __init__(self):
        self._last_round = False
        self._lines_cleared = 0
        self._new_piece = 0

        self._board = None

    def parse(self, msg):
        """
        Parse message, extract info about last round, lines cleared, and current
        board state.
        """
        last_round, lines_cleared, new_piece, *board = msg.split()

        self._last_round = True if int(last_round) else False
        self._lines_cleared = float(lines_cleared)
        self._new_piece = int(new_piece)
        self._board = np.array([float(block_color) for block_color in board]).reshape(config.BOARD_HEIGHT, config.BOARD_WIDTH)

    def last_round(self):
        """
        Return if game is over.
        """
        return self._last_round

    def reward(self):
        """
        Return reward (lines_cleared)
        """
        return self._lines_cleared

    def piece(self):
        """
        Current piece.
        """
        return self._new_piece

    def raw_board(self):
        """
        Create flat raw (without merged piece) board.
        """
        # Normalize board, all blocks are set to 1
        out_board = (self._board > 0).astype(float)

        return out_board

    def board(self):
        """
        Create flat board with four blocks representing pieces.
        """
        # Normalize board, all blocks are set to 1
        out_board = (self._board > 0).astype(float)

        # In top row set four middle block as new piece, and erase all other
        for x in range(config.BOARD_WIDTH):
            if 3 < x < 8:
                out_board[0][x] = self._new_piece / TetrisData.NUM_OF_COLORS
            else:
                out_board[0][x] = config.EMPTY_BLOCK

        return out_board
