#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import numpy as np
import time
import config
import utils


class Environment:
    def __init__(self, sock):
        self._sock = sock
        self._conn = None
        self._buffer = bytes()

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

        last_round, reward, state = self._receive_data()
        utils.log("Reward:", reward)

        return last_round, reward, state

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
            self._buffer += self._conn.recv(1024)

            if b'\n' in self._buffer:
                break

        self._step_tic = time.time()

        msg = self._buffer[:self._buffer.find(b'\n')]
        self._buffer = self._buffer[self._buffer.find(b'\n') + 1:]

        # Parse msg from robot
        last_round, reward, *state = msg.decode().split()

        last_round = True if int(last_round) else False
        reward = float(reward)
        state = np.array([float(val) for val in state])

        return last_round, reward, state
