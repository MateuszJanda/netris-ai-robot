#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import numpy as np
from dqn import config


class HeuristicSolver:

    # Map piece color to his basic block representation
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

    # Starting column for piece
    START_COL = {
        1: [4, 5, 4, 4],
        2: [4, 5, 4, 5],
        3: [4, 4, 4, 4],
        4: [4, 5, 4, 4],
        5: [4, 5, 4, 4],
        6: [4, 5, 4, 5],
        7: [4, 4, 4, 4],
    }

    MAX_BOARD_WIDTH = 32
    MAX_BOARD_HEIGHT = 64

    def action(self, piece_index, board):
        """
        Choose action from ACTION_SPACE_SIZE.
        """
        best_action = 0
        piece_blocks = np.array(HeuristicSolver.PIECE[piece_index])

        board = board.reshape(config.BOARD_HEIGHT, config.BOARD_WIDTH) > 0
        board = board.astype(int)

        min_score = None
        for rot in range(4):
            for col in range(config.BOARD_WIDTH):
                last_row, merged_board = self._fit(col, piece_blocks, board)

                if last_row is None:
                    continue

                lines_cleared, final_board = self._clear_full_lines(merged_board)
                score = self._score(last_row, lines_cleared, final_board)

                if min_score is None or score < min_score:
                    min_score = score

                    shift = col - HeuristicSolver.START_COL[piece_index][rot]
                    best_action = rot * config.BOARD_WIDTH + shift + config.SHFIT_OFFSET

            piece_blocks = np.rot90(piece_blocks)

        return best_action

    def _fit(self, col, piece_blocks, board):
        """
        Fit piece blocks with board.
        """
        if col + piece_blocks.shape[1] > config.BOARD_WIDTH:
            return None, None

        top_row = None
        merged_board = None
        for row in range(config.BOARD_HEIGHT - piece_blocks.shape[0] + 1):
            sub_board = board[row: row + piece_blocks.shape[0], col: col + piece_blocks.shape[1]]

            if np.any(sub_board + piece_blocks == 2):
                break

            top_row = row

        if top_row is None:
            return None, None

        merged_board = np.copy(board)
        merged_board[top_row: top_row + piece_blocks.shape[0], col: col + piece_blocks.shape[1]] += piece_blocks
        return top_row + piece_blocks.shape[0] - 1, merged_board

    def _clear_full_lines(self, merged_board):
        """
        Clear full lines from merged board (board with piece blocks).
        Return number of cleared lines and final board.
        """
        lines_cleared = 0
        dst = merged_board.shape[0] - 1
        for src in reversed(range(merged_board.shape[0])):
            if np.all(merged_board[src]):
                merged_board[src] = np.zeros(merged_board.shape[1])
                lines_cleared += 1
            else:
                if src != dst:
                    merged_board[dst] = merged_board[src]
                    merged_board[src] = np.zeros(merged_board.shape[1])
                dst -= 1

        return lines_cleared, merged_board

    def _score(self, last_row, lines_cleared, final_board):
        """
        Calculate score. This is recreation of Netris simple robot algorithm.
        """
        # In Netris row indexing is flipped.
        final_board = np.flip(final_board, axis=0)
        last_row = config.BOARD_HEIGHT - last_row - 1

        max_height = 0
        height = np.zeros((HeuristicSolver.MAX_BOARD_WIDTH), dtype=int)
        for col in range(config.BOARD_WIDTH):
            for row in range(config.BOARD_HEIGHT):
                if final_board[row][col]:
                    height[col] = row + 1
            if max_height < height[col]:
                max_height = height[col]

        # Calculate dependencies
        cover = np.zeros((config.BOARD_WIDTH), dtype=int)
        depend = np.zeros((max_height), dtype=int)
        for row in reversed(range(max_height)):
            for col in range(config.BOARD_WIDTH):
                if final_board[row][col]:
                    cover[col] |= 1 << row
                else:
                    depend[row] |= cover[col]

            for i in range(row + 1, max_height):
                if depend[row] & (1 << 1):
                    depend[row] |= depend[i]

        # Calculate hardness of fit
        hard_fit = np.full((max_height), 5)
        space = 0
        delta_left = 0
        delta_right = 0
        fit_probs = 0
        for row in reversed(range(max_height)):
            count = 0
            for col in range(config.BOARD_WIDTH):
                if final_board[row][col]:
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
                        delta_left = HeuristicSolver.MAX_BOARD_HEIGHT

                    if col < config.BOARD_HEIGHT - 1:
                        delta_right = height[col + 1] - row
                    else:
                        delta_right = HeuristicSolver.MAX_BOARD_HEIGHT

                    if delta_left > 2 and delta_right > 2:
                        hard_fit[row] += 7
                    elif delta_left > 2 or delta_right > 2:
                        hard_fit[row] += 2
                    elif abs(delta_left) == 2 and abs(delta_right) == 2:
                        hard_fit[row] += 2
                    elif abs(delta_left) == 2 or abs(delta_right) == 2:
                        hard_fit[row] += 3

            max_hard = 0
            for i in range(row + 1, min(row + 5, max_height)):
                if depend[row] & (1 << i):
                    if max_hard < hard_fit[i]:
                        max_hard = hard_fit[i]
            fit_probs += max_hard * count

        # Calculate score based on top shape
        top_shape = 0
        for col in range(config.BOARD_WIDTH):
            if col > 0:
                delta_left = height[col - 1] - height[col]
            else:
                delta_left = HeuristicSolver.MAX_BOARD_HEIGHT

            if col < config.BOARD_WIDTH - 1:
                delta_right = height[col + 1] - height[col]
            else:
                delta_right = HeuristicSolver.MAX_BOARD_HEIGHT

            if delta_left > 2 and delta_right > 2:
                top_shape += 15 + 15 * (min(delta_left, delta_right) // 4)
            elif delta_left > 2 or delta_right > 2:
                top_shape += 2
            elif abs(delta_left) == 2 and abs(delta_right) == 2:
                top_shape += 2
            elif abs(delta_left) == 2 or abs(delta_right) == 2:
                top_shape += 3

        close_to_top = last_row / config.BOARD_HEIGHT
        close_to_top *= close_to_top

        close_to_top *= 200
        space /= 2
        score = space + close_to_top + top_shape + fit_probs - lines_cleared * 10

        return score
