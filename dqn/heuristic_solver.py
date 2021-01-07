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
        piece_blocks = np.array(HeuristicSolver.PIECE[piece_index])

        board = board.reshape(config.BOARD_HEIGHT, config.BOARD_WIDTH) > 0
        board = board.astype(int)

        min_score = None
        for rot in range(4):
            for col in range(config.BOARD_WIDTH):
                row, merged_board = self._fit(col, piece_blocks, board)

                if not row:
                    continue

                lines_cleared = self._lines_cleared(merged_board)
                score = self._score(row, merged_board)

                if not min_score or min_score < score:
                    min_score = score
                    best_action = rot * config.BOARD_WIDTH + col

            piece_blocks = np.rot90(piece_blocks)

        return best_action

    def _fit(self, col, piece_blocks, board):
        if col + piece_blocks.shape[1] >= config.BOARD_WIDTH:
            return None

        last_row = None
        last_board = None
        for row in range(config.BOARD_HEIGHT - piece_blocks.shape[0]):
            merged_board = board[row: row + piece_blocks.shape[0], col: col + piece_blocks.shape[1]] + piece_blocks

            if np.any(merged_board == 2):
                return last_row, last_board

            last_row = row

        return last_row, last_board

    def _lines_cleared(self, merged_board):
        return np.sum(np.sum(merged_board, axis=1) == config.BOARD_WIDTH)

    def _score(self, current_row, lines_cleared, board):
        max_height = 0

        height = np.zeros((config.BOARD_WIDTH), dtype=int)
        for col in range(config.BOARD_WIDTH):
            for row in range(config.BOARD_HEIGHT):
                if board[row][col]:
                    height[col] = row + 1
            if max_height < height[col]:
                max_height = height[col]

        cover = np.zeros((config.BOARD_WIDTH), dtype=int)
        depend = np.zeros((config.BOARD_HEIGHT), dtype=int)
        for row in reversed(range(max_height)):
            for col in range(config.BOARD_WIDTH):
                if board[row][col]:
                    cover[col] |= 1 << row
                else:
                    depend[row] |= cover[col]

            for i in range(row + 1, max_height):
                if depend[row] & (1 << 1):
                    depend[row] |= depend[i]


        hard_fit = np.full((config.BOARD_HEIGHT), 5, dtype=int)
        space = 0
        delta_left = 0
        delta_right = 0
        fit_probs = 0
        for row in reversed(range(max_height)):
            count = 0
            for col in range(config.BOARD_WIDTH):
                if board[row][col]:
                    space += 0.5
                else:
                    count += 1
                    space += 1
                    hard_fit[row] += 1

                    if height[col] < row:
                        hard_fit[row] += row - height[col]

                    if col > 0:
                        delta_left = height[col - 1] - row
                    else:
                        delta_left = config.BOARD_HEIGHT
                    if col < config.BOARD_HEIGHT - 1:
                        delta_right = height[col + 1] - row
                    else:
                        delta_right = config.BOARD_HEIGHT
                    if delta_left > 2 and delta_right > 2:
                        hard_fit[row] += 7
                    elif delta_left > 2 or delta_right > 2:
                        hard_fit[row] += 2
                    elif abs(delta_left) == 2 and abs(delta_right) == 2:
                        hard_fit[row] += 2
                    elif abs(delta_left) == 2 or abs(delta_right) == 2:
                        hard_fit[row] += 3

            max_hard = 0
            for i in range(min(row + 5, max_height)):
                if depend[row] & (1 << i):
                    if max_hard < hard_fit[i]:
                        max_hard = hard_fit[i]
            fit_probs += max_hard * count

        top_shape = 0
        for col in range(config.BOARD_WIDTH):
            if col > 0:
                delta_left = height[col - 1] - height[col]
            else:
                delta_left = config.BOARD_HEIGHT
            if col < config.BOARD_WIDTH - 1:
                delta_right = height[col + 1] - height[col]
            else:
                delta_right = config.BOARD_HEIGHT
            if delta_left > 2 and delta_right > 2:
                top_shape += 15 + 15 * (min(delta_left, delta_right) / 4)
            elif delta_left > 2 or delta_right > 2:
                top_shape += 2
            elif abs(delta_left) == 2 and abs(delta_right) == 2:
                top_shape += 2
            elif abs(delta_left) == 2 or abs(delta_right) == 2:
                top_shape += 3

        close_to_top = current_row / config.BOARD_HEIGHT
        close_to_top *= close_to_top

        close_to_top *= 200
        score = space + close_to_top + top_shape + fit_probs - lines_cleared * 10

        return score
