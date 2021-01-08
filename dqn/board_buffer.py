#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import numpy as np
import config
import utils


class BoardBuffer:
    def __init__(self, log_file):
        self._log_file = log_file

        self._board = np.zeros(shape=(config.BOARD_HEIGHT, config.BOARD_WIDTH), dtype=int)
        self._sequence_num = None
        self._piece_color = 0
        self._read_piece = False
        self._round = 0

        self._lines_cleared = 0

    @property
    def sequence_num(self):
        return self._sequence_num

    def update_lines_cleared(self, params):
        """
        Handle Ext:LinesCleared - available only in netris-env.

        Format:
        Ext:LinesCleared <screen-id> <lines-cleared>

        Example:
        Ext:LinesCleared 0 0
        """
        scr, lines_cleared = [int(p) for p in params]

        # Skip data if they don't belong to robot
        if scr != config.SCREEN_ID:
            return

        self._lines_cleared = lines_cleared

    def update_new_piece(self, params):
        """
        Handle NewPiece from netris. Unfortunately game provide only
        sequence number not real piece id.

        Format:
        NewPiece <sequence-number>

        Example:
        NewPiece 26
        """
        self._sequence_num = params[0]
        self._read_piece = True
        self._round += 1

    def update_row(self, params):
        """
        Handle RowUpdate command from Netris. Update board. This is the moment
        when action can be taken for new piece.

        Format:
        RowUpdate <screen-id> <line> <color> * width
        Note:
        - negative color value describe moving, positive fixed blocks.
        - top line has high number, bottom line is 0

        Example:
        RowUpdate 0 19 0 0 0 -3 -3 0 0 0 0 0
        """
        scr, y, *row = [int(p) for p in params]

        # Skip data if they don't belong to robot
        if scr != config.SCREEN_ID:
            return False

        # Netris inform about switch from "piece block" to "fixed block" starting
        # from second RowUpdate after NewPiece command. This is to late
        # for prediction, so better is assume that first line is always empty,
        # and mark "piece (moving) block" as "fixed".
        if y != config.TOP_LINE:
            for x, color in enumerate(row):
                self._board[config.BOARD_HEIGHT - 1 - y][x] = abs(color)

        # Board ready to send if new piece in top line
        if self._read_piece and y == config.TOP_LINE:
            for color in row:
                if color < 0:
                    self._piece_color = abs(color)
                    self._read_piece = False
                    return True

        return False

    def flush_as_msg(self, game_is_over=False):
        """
        Create status message for agent.
        """
        flat_board = self._board.flatten()
        flat_values = "".join([("%d " % val) for val in flat_board])

        game_is_over = int(game_is_over)

        # Format status message
        status = str(game_is_over) + " " + str(self._lines_cleared) + " " + \
            str(self._piece_color) + " " + flat_values + "\n"

        self._reset()
        return status

    def _reset(self):
        """
        Reset lines cleared, and new piece.
        """
        self._lines_cleared = 0
        self._new_piece = 0

    def _log(self, *args, **kwargs):
        """
        Print log to other terminal or file.
        """
        if self._log_file:
            print(*args, **kwargs, file=self._log_file)
