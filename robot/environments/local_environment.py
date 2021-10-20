#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import time
import random
import numpy as np
from robot import config


class LocalEnvironment:
    def __init__(self):
        self.piece = 0
        self.shift = 0
        self.rotate = 0
        self.points = 0

        self._board = None

        self._step_tic = time.time()
        self.game_tic = time.time()
        self.handling_time = []

    def reset(self):
        """
        Reset game, clear board.
        """
        self._board = np.zeros(shape=(config.BOARD_HEIGHT, config.BOARD_WIDTH), dtype=int)

        last_round = False
        reward = 0
        piece = random.randint(1, config.NUM_OF_PIECES + 1)

        raw_board = (self._board > 0).astype(float)

        return last_round, reward, piece, raw_board, self._merge_piece_with_board(piece, piece)

    def _merge_piece_with_board(self, new_piece):
        """
        Create board with four blocks representing pieces.
        """
        # Normalize board, all blocks are set to 1
        out_board = (self._board > 0).astype(float)

        # In top row set four middle block as new piece, and erase all others
        for x in range(config.BOARD_WIDTH):
            if 3 < x < 8:
                out_board[0][x] = new_piece / config.NUM_OF_COLORS
            else:
                out_board[0][x] = config.EMPTY_BLOCK

        return out_board

    def step(self, action):
        """Send action to robot and receive new feedback."""
        if action >= config.ACTION_SPACE_SIZE:
            raise Exception("Action not in action space:", action)

        self.handling_time.append(time.time() - self._step_tic)
        shift = action % config.BOARD_WIDTH - config.SHFIT_OFFSET
        rotate = action // config.BOARD_WIDTH

        message = str(shift) + ' ' + str(rotate) + '\n'
        self._conn.sendall(message.encode())

        last_round, reward, piece, raw_board, board = self._update_model()

        return last_round, reward, piece, raw_board, board

    def close(self):
        """Only to fulfill API requirements."""
        pass

    def _update_model(self):
        """Receive data from robot."""
        if not self._conn:
            raise Exception('Connection not established')

        # Ensure that new full data is received (single line with \n at the end)
        while True:
            self._buffer += self._conn.recv(BUFFER_SIZE)

            if b'\n' in self._buffer:
                break

        self._step_tic = time.time()

        msg_status = self._buffer[:self._buffer.find(b'\n')]
        self._buffer = self._buffer[self._buffer.find(b'\n') + 1:]

        # Parse status from robot
        self._tetris_data.parse(msg_status.decode())

        return self._tetris_data.last_round(), self._tetris_data.reward(), \
            self._tetris_data.piece(), self._tetris_data.raw_board(), \
            self._tetris_data.board()
