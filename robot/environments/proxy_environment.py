#!/usr/bin/env python3

# Author: Mateusz Janda <mateusz janda at gmail com>
# Site: github.com/MateuszJanda/netris-ai-robot
# Ad maiorem Dei gloriam


import time
from robot import config
from robot.environments.tetris_data import TetrisData


BUFFER_SIZE = 1024


def print_board(board, descripiton):
    """Print current board state. For debug purpose."""
    print("Board: ", descripiton)
    for row in board:
        line = "".join(["1" if b else "0" for b in row])
        print(line)
    print('')



class ProxyEnvironment:
    def __init__(self, sock):
        self._sock = sock
        self._conn = None
        self._buffer = bytes()

        self._tetris_data = TetrisData()

        self._start_tic = time.time()
        self._game_tic = time.time()
        self._handle_times = []

    def reset(self):
        """
        Reset game. Close connection with old robot and establish
        connection with new one.
        """
        self._game_tic = time.time()
        self._handle_times = []

        self._conn, _ = self._sock.accept()
        last_round, reward, piece_index, raw_board, board = self._receive_data()

        self._start_tic = time.time()
        return last_round, reward, piece_index, raw_board, board

    def step(self, action):
        """Send action to robot and receive new feedback."""
        self._handle_times.append(time.time() - self._start_tic)
        self._start_tic = time.time()

        if action >= config.ACTION_SPACE_SIZE:
            raise Exception("Action not in action space:", action)

        shift = action % config.BOARD_WIDTH - config.SHFIT_OFFSET
        rotate = action // config.BOARD_WIDTH

        message = str(shift) + ' ' + str(rotate) + '\n'
        self._conn.sendall(message.encode())

        last_round, reward, piece_index, raw_board, board = self._receive_data()

        return last_round, reward, piece_index, raw_board, board

    def _receive_data(self):
        """Receive data from robot."""
        if not self._conn:
            raise Exception("Connection not established")

        # Ensure that new full data is received (single line with \n at the end)
        while True:
            self._buffer += self._conn.recv(BUFFER_SIZE)

            if b'\n' in self._buffer:
                break

        msg_status = self._buffer[:self._buffer.find(b'\n')]
        self._buffer = self._buffer[self._buffer.find(b'\n') + 1:]

        # Parse status from robot
        self._tetris_data.parse(msg_status.decode())

        return self._tetris_data.last_round(), self._tetris_data.reward(), \
            self._tetris_data.piece_index(), self._tetris_data.raw_board(), \
            self._tetris_data.board()

    def raw_board(self):
        """
        Return 2D raw board (without piece).
        """
        return self._tetris_data.raw_board()

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
        """Close connection with robot."""
        if self._conn:
            self._conn.close()
