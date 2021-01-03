#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import numpy as np
import config


class TetrisModel:
    def __init__(self, loop, future_stop, queue):
        self._board = np.zeros(shape=(config.BORAD_HEIGHT, config.BOARD_WIDTH), dtype=int)
        self._sequence_num = None
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
        Handle NewPiece from server. Unfortunately server provide here only
        sequence number not real piece id.
        """
        self._sequence_num = params[0]
        self._fresh_piece = True
        self._round += 1

    def validate_board_size(self, params):
        """
        Handle BoardSize command from server. Just to validate height and
        width.
        """
        scr_id, height, width = [int(p) for p in params]

        if width != BOARD_WIDTH and height != BORAD_HEIGHT:
            raise Exception("Invalid board size fail %d %d %d %d" % (width, BOARD_WIDTH, height, BORAD_HEIGHT))
            return

        return

    def update_row(self, params):
        """
        Handle RowUpdate command from server. Update board. This is the moment
        when action can be taken for new piece.
        """
        scr, y, *row = [int(p) for p in params]

        # Skip data if they don't belong to robot
        if scr != config.SCREEN_ID:
            return

        # Netris inform about switch from "piece block" to "fixed block" starting
        # from second RowUpdate command after NewPiece command. This is to late,
        # for prediction, so better is assume that first line is always empty.
        if y != TOP_LINE:
            for x, val in enumerate(row):
                self._board[config.BORAD_HEIGHT - 1 - y][x] = config.FULL_BLOCK if val else config.EMPTY_BLOCK

        # Send board to agent if this is new piece
        if self._fresh_piece and y == config.TOP_LINE:
            self._fresh_piece = False

        return
