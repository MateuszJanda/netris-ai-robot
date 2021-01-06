#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import numpy as np
import config

class HeuristicSolver:

    PIECE = {
        1: [[1, 1, 1],
            [0, 1, 0]],
        2: [[1, 1, 1, 1]],
        3: [[1, 1],
            [1, 1]],
        4: [[1, 1, 1],
            [1, 0, 0]],
        5: [[1, 1, 1],
            [0, 0, 1]],
        6: [[0, 1, 1],
            [1, 1, 0]],
        7: [[1, 1, 0],
            [0, 1, 1]],
    }

    def action(self, piece_index, board):
        best_action = 0
        piece = np.array(PIECE[piece_index])

        board = board > 0
        board = board.astype(int)

        for rot in range(4):
            for column in range(config.BOARD_WIDTH):
                row = self._fit(column, piece, board)

                if not row:
                    continue

                lines_cleared = self._lines_cleared()

            piece = np.rot90(piece)

        return best_action

    def _fit(self, column, piece, board):
        if column + piece.shape[1] >= config.BOARD_WIDTH:
            return None

        last_row = None
        for row in range(config.BOARD_HEIGHT):
            sub_board = board[row: row + piece.shape[0]][column: column + piece.shape[1]]

            if np.any(piece + sub_board == 2):
                return last_row

            last_row = row

        return last_row

    def _lines_cleared(self):
        return 0
