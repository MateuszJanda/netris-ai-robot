#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import time
import config
from tetris_model import TetrisModel


BUFFER_SIZE = 1024


class Environment:
    def __init__(self, sock):
        self._sock = sock
        self._conn = None
        self._buffer = bytes()

        self._model = TetrisModel()

        self._step_tic = time.time()
        self.game_tic = time.time()
        self.handling_time = []

    def reset(self):
        """
        Reset game. Close connection with old robot and establish
        connection with new one.
        """
        self.game_tic = time.time()
        self.handling_time = []
        self._conn, addr = self._sock.accept()
        last_round, reward, state = self._receive_data()

        return state

    def step(self, action):
        """Send action to robot and receive new feedback."""
        if action >= config.ACTION_SPACE_SIZE:
            raise Exception("Action not in action space:", action)

        self.handling_time.append(time.time() - self._step_tic)
        shift = action % config.BOARD_WIDTH - config.SHFIT_OFFSET
        rotate = action // config.BOARD_WIDTH

        message = str(shift) + ' ' + str(rotate) + '\n'
        self._conn.sendall(message.encode())

        last_round, reward, board_state = self._receive_data()
        print("Reward:", reward)

        return last_round, reward, board_state

    def close(self):
        """Close connection with robot."""
        if self._conn:
            self._conn.close()

    def _receive_data(self):
        """Receive data from robot."""
        if not self._conn:
            raise Exception('Connection not established')

        # Ensure that new full data is received (single line with \n at the end)
        while True:
            self._buffer += self._conn.recv(BUFFER_SIZE)

            if b'\n' in self._buffer:
                break

        self._step_tic = time.time()

        status = self._buffer[:self._buffer.find(b'\n')]
        self._buffer = self._buffer[self._buffer.find(b'\n') + 1:]

        # Parse status from robot
        self._model.parse(status.decode())

        return self.last_round(), self.reward(), self.board()