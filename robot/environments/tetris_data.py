#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""


import numpy as np
from robot import config


class TetrisData:
    def __init__(self):
        self._last_round = False
        self._lines_cleared = 0
        self._new_piece_index = 0

        self._board = None

    def parse(self, msg):
        """
        Parse message, extract info about last round, cleared lines, and current
        board state.
        """
        last_round, lines_cleared, new_piece_index, *board = msg.split()

        self._last_round = True if int(last_round) else False
        self._lines_cleared = float(lines_cleared)
        self._new_piece_index = int(new_piece_index)
        self._board = np.array([float(block_color) for block_color in board]).reshape(config.BOARD_HEIGHT, config.BOARD_WIDTH)

    def last_round(self):
        """
        Return if game is over.
        """
        return self._last_round

    def reward(self):
        """
        Return reward (cleared lines)
        """
        return self._lines_cleared

    def piece_index(self):
        """
        Current piece index.
        """
        return self._new_piece_index

    def raw_board(self):
        """
        Create raw (without merged piece) board.
        """
        # Normalize board, all blocks are set to 1.0
        out_board = (self._board > 0).astype(float)

        return out_board

    def board(self):
        """
        Create board with four blocks representing pieces.
        """
        # Normalize board, all blocks are set to 1
        out_board = (self._board > 0).astype(float)

        # In top row set four middle block as new piece and erase all others
        for x in range(config.BOARD_WIDTH):
            if 3 < x < 8:
                out_board[0][x] = self._new_piece_index / config.NUM_OF_PIECES
            else:
                out_board[0][x] = config.EMPTY_BLOCK

        return out_board
