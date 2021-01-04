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

    def __init__(self, log_file):
        self._log_file = log_file

        self._board = np.zeros(shape=(config.BOARD_HEIGHT, config.BOARD_WIDTH), dtype=int)
        self.sequence_num = None
        self._fresh_piece = False
        self._round = 0

        self._lines_cleared = 0

    def update_lines_cleared(self, params):
        """Handle Ext:LinesCleared - available only in netris-env."""
        scr, lines_cleared = [int(p) for p in params]

        # Skip data if they don't belong to robot
        if scr != config.SCREEN_ID:
            return

        self._lines_cleared = lines_cleared

    def update_new_piece(self, params):
        """
        Handle NewPiece from netris. Unfortunately game provide only
        sequence number not real piece id.
        """
        self.sequence_num = params[0]
        self._fresh_piece = True
        self._round += 1

    def update_row(self, params):
        """
        Handle RowUpdate command from netris. Update board.
        """
        scr, y, *row = [int(p) for p in params]

        # Skip data if they don't belong to robot
        if scr != config.SCREEN_ID:
            return None

        # Netris inform about switch from "piece block" to "fixed block" starting
        # from second RowUpdate command after NewPiece command. This is to late,
        # for prediction, so better is assume that first line is always empty.
        if y != config.TOP_LINE:
            for x, val in enumerate(row):
                self._board[config.BOARD_HEIGHT - 1 - y][x] = config.FULL_BLOCK if val else config.EMPTY_BLOCK

        # Create board status if this is new piece
        if self._fresh_piece and y == config.TOP_LINE:
            self._fresh_piece = False
            return row

        return None

    def log(self, *args, **kwargs):
        """Print log to other terminal or file."""
        if self._log_file:
            print(*args, **kwargs, file=self._log_file)
