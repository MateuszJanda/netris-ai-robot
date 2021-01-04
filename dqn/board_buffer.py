#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import numpy as np
import config


class BoardBuffer:
    def __init__(self, log_file):
        self._log_file = log_file

        self._board = np.zeros(shape=(config.BOARD_HEIGHT, config.BOARD_WIDTH), dtype=int)
        self.sequence_num = None # TODO: getter
        self._new_piece = 0
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
            return False

        # Netris inform about switch from "piece block" to "fixed block" starting
        # from second RowUpdate command after NewPiece command. This is to late,
        # for prediction, so better is assume that first line is always empty.
        if y != config.TOP_LINE:
            for x, color in enumerate(row):
                assert(color >= 0) # TODO: check
                self._board[config.BOARD_HEIGHT - 1 - y][x] = color

        # Create board status if this is new piece
        if self._fresh_piece and y == config.TOP_LINE:
            for color in row:
                if color < 0:
                    self._new_piece = -color
                    self._fresh_piece = False
                    return True

        return False

    def flush_as_msg(self, game_is_over=False):
        """Create status message for agent."""
        new_board = np.copy(self.model._board).flatten()
        flat_board = "".join([("%d " % val) for val in new_board])

        game_is_over = int(game_is_over)

        # Format status message
        status = str(game_is_over) + " " + str(self._lines_cleared) + " " + \
            str(self._new_piece) +  self._ flat_board + "\n"

        self._reset()
        return status

    def _reset(self):
        self._lines_cleared = 0
        self._new_piece = 0

    def log(self, *args, **kwargs):
        """Print log to other terminal or file."""
        if self._log_file:
            print(*args, **kwargs, file=self._log_file)
