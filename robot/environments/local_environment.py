#!/usr/bin/env python3

# Author: Mateusz Janda <mateusz janda at gmail com>
# Site: github.com/MateuszJanda/netris-ai-robot
# Ad maiorem Dei gloriam

import time
import random
import numpy as np
from robot import config


# Piece index and it representation. Counterclockwise rotation.
INDEX_TO_PIECE = {
    1: [
        [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    ],
    2: [
        [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    ],
    3: [
        [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]
    ],
    4: [
        [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    ],
    5: [
        [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]],
    ],
    6: [
        [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    ],
    7: [
        [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    ]
}

class LocalEnvironment:
    """
    Local environment with build in *etris logic.
    """

    def __init__(self):
        self._board = None
        self._piece_index = 0

        self._game_tic = time.time()
        self._start_tic = time.time()
        self._handle_times = []

    def reset(self):
        """Reset game, clear board."""
        self._game_tic = time.time()
        self._handle_times = []

        self._board = np.zeros(shape=(config.BOARD_HEIGHT, config.BOARD_WIDTH), dtype=int)

        last_round = False
        reward = 0
        # self._piece_index = random.randrange(1, config.NUM_OF_PIECES + 1)
        self._piece_index = random.choice([2, 3, 4])

        raw_board = (self._board > 0).astype(float)

        self._start_tic = time.time()
        return last_round, reward, self._piece_index, raw_board, self._board_with_flat_piece(self._piece_index)

    def step(self, action):
        """Apply action from agent and return current game state."""
        self._handle_times.append(time.time() - self._start_tic)
        self._start_tic = time.time()

        if action >= config.ACTION_SPACE_SIZE:
            raise Exception("Action not in action space:", action)

        last_round, reward = self._apply_action(action)

        # self._piece_index = random.randrange(1, config.NUM_OF_PIECES + 1)
        self._piece_index = random.choice([2, 3, 4])
        raw_board = (self._board > 0).astype(float)

        return last_round, reward, self._piece_index, raw_board, self._board_with_flat_piece(self._piece_index)

    def _apply_action(self, action):
        """Apply agent action."""
        shift = action % config.BOARD_WIDTH - config.SHFIT_OFFSET
        rotate = (action // config.BOARD_WIDTH) % len(INDEX_TO_PIECE[self._piece_index])

        piece, piece_height = self._position_piece_before_drop(self._piece_index, rotate, shift)
        self._board = self._board_with_dropped_piece(piece, piece_height)

        reward = self._count_full_lines()
        self._remove_full_lines()

        return self._is_last_round(), reward

    def _position_piece_before_drop(self, piece_index, rotate, shift):
        """Position (roate and shift) piece before drop."""
        piece = np.array(INDEX_TO_PIECE[piece_index][rotate])

        piece_height = len(INDEX_TO_PIECE[piece_index][rotate])
        piece = np.pad(piece, pad_width=[(0, config.BOARD_HEIGHT - piece_height), (0, 0)], mode="constant", constant_values=0)

        for _ in range(abs(shift)):
            if shift < 0:
                shifted_piece = np.roll(piece, shift=-1)
            else:
                shifted_piece = np.roll(piece, shift=1)

            if self._is_collision(shifted_piece):
                break

            piece = shifted_piece

            if self._is_touching_wall(piece):
                break

        return piece, piece_height

    def _board_with_dropped_piece(self, piece, piece_height):
        """Position piece during drop and return with merged piece."""
        for _ in range(config.BOARD_HEIGHT - piece_height):
            dropped_piece = np.roll(piece, shift=1, axis=0)

            if self._is_collision(dropped_piece):
                break

            piece = dropped_piece

        return self._board + piece

    def _count_full_lines(self):
        """Count full lines (agent reward)."""
        return np.sum(np.sum(self._board, axis=1) == config.BOARD_WIDTH)

    def _remove_full_lines(self):
        """Remove full lines from board."""
        for y, count in enumerate(np.sum(self._board, axis=1)):
            if count != config.BOARD_WIDTH:
                continue

            # Copy previous rows except first
            self._board[1:y+1] = self._board[:y]

            # Zero out first row
            self._board[:1] = 0

    def _is_touching_wall(self, piece):
        """Check if piece is touching board walls."""
        return np.any(piece[:,0]) or np.any(piece[:,-1])

    def _is_collision(self, piece):
        """Check for collision with blocks on current board."""
        return np.any(self._board + piece >= 2)

    def _is_last_round(self):
        """True if there is any block in first line."""
        return np.any(self._board[0,:])

    def _board_with_flat_piece(self, new_piece_index):
        """Create board with four blocks representing pieces."""
        # Normalize board, all blocks are set to 1.0
        out_board = (self._board > 0).astype(float)

        # In top row set four middle block as new piece, and erase all others
        for x in range(config.BOARD_WIDTH):
            if 3 < x < 8:
                out_board[0][x] = new_piece_index / config.NUM_OF_PIECES
            else:
                out_board[0][x] = config.EMPTY_BLOCK

        return out_board

    def raw_board(self):
        """
        Return 2D raw board (without piece).
        """
        return self._board > 0

    def num_of_steps(self):
        """Return current numer of steps (pieces)."""
        return len(self._handle_times)

    def game_duration(self):
        """Return current game duration."""
        return time.time() - self._game_tic

    def step_duration(self):
        """Return one step duration."""
        return sum(self._handle_times) / len(self._handle_times)

    def close(self):
        """Only to fulfill API requirements."""
        pass
