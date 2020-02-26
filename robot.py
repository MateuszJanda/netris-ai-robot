#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import traceback
import time
import datetime
import numpy as np
import os
# Disable info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf


BOARD_WIDTH = 10
BORAD_HEIGHT = 20
SCR_ID = 0

EMPTY_BLOCK = 0
FULL_BLOCK = 1


class Robot:
    def __init__(self):
        self.file = open('/dev/pts/2', 'w')
        self.board = np.zeros(shape=(BORAD_HEIGHT, BOARD_WIDTH), dtype=int)

    def set_log_file(self, file):
        # self.file = file
        pass

    def new_pice(self, params):
        piece_id = params[0]
        out = ['Right ' + piece_id, 'Right ' + piece_id, 'Right ' + piece_id, 'Message Disp asdf']
        self._print_board()
        return True, out

    def board_size(self, params):
        scr_id, height, width = [int(p) for p in params]

        if width != BOARD_WIDTH and height != BORAD_HEIGHT:
            self._log('[!] Validation board size fail %d %d %d %d' % (width, BOARD_WIDTH, height, BORAD_HEIGHT))
            return False, ['Exit']

        return True, []

    def row_update(self, params):
        params = [int(p) for p in params]

        if params[0] != SCR_ID:
            return True, []

        y = params[1]
        for x, val in enumerate(params[2:]):
            self.board[BORAD_HEIGHT - 1 - y][x] = FULL_BLOCK if val != 0 else EMPTY_BLOCK

        return True, []

    def exit(self, params):
        return True, []

    def _print_board(self):
        self._log('Board')
        for line in self.board:
            l = ''.join(['1' if b else ' ' for b in line])
            self._log(l)

    def _log(self, *args, **kwargs):
        if self.file:
            print(*args, **kwargs, file=self.file)


class RobotML:
    def __init__(self):
        self.file = open('/dev/pts/2', 'w')
        sys.stderr = self.file
        self.board = np.zeros(shape=(BORAD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.model = tf.keras.models.load_model('only_wins.h5')

    def set_log_file(self, file):
        # self.file = file
        pass

    def new_pice(self, params):
        piece_id = params[0]
        x_data = np.array([np.concatenate(([int(piece_id)/6], self.board.flatten()))])
        self._log('here 3')
        self._log(x_data.shape)

        y_shift, y_rotate = self.model.predict(x_data)
        self._log('here 4')
        shift = np.argmax(y_shift) - 5
        rotate = np.argmax(y_rotate)

        out = []
        if shift < 0:
            while shift != 0:
                out.append('Left ' + piece_id)
                shift += 1
        elif shift > 0:
            while shift != 0:
                out.append('Right ' + piece_id)
                shift -= 1

        while rotate != 0:
            out.append('Rotate ' + piece_id)
            rotate -= 1

        out.append('Drop ' + piece_id)
        return True, out

    def board_size(self, params):
        scr_id, height, width = [int(p) for p in params]

        if width != BOARD_WIDTH and height != BORAD_HEIGHT:
            self._log('[!] Validation board size fail %d %d %d %d' % (width, BOARD_WIDTH, height, BORAD_HEIGHT))
            return False, ['Exit']

        return True, []

    def row_update(self, params):
        params = [int(p) for p in params]

        if params[0] != SCR_ID:
            return True, []

        y = params[1]
        for x, val in enumerate(params[2:]):
            self.board[BORAD_HEIGHT - 1 - y][x] = FULL_BLOCK if val != 0 else EMPTY_BLOCK

        return True, []

    def exit(self, params):
        return True, []

    def _print_board(self):
        self._log('Board')
        for line in self.board:
            l = ''.join(['1' if b else ' ' for b in line])
            self._log(l)

    def _log(self, *args, **kwargs):
        if self.file:
            print(*args, **kwargs, file=self.file)


def game(robot):
    log_name = parse_args()

    # try:
    if log_name:
        with open(log_name, 'w') as file:
            loop(file, robot)
    else:
        loop(None, robot)
    # except:
    #     if log_name:
    #         with open(log_name, 'w') as file:
    #             traceback.print_exc(file=file)


def loop(file, robot):
    robot.set_log_file(file)
    out_cmd(file, 'Version 1')

    handler = {
        'NewPiece' : robot.new_pice,
        'BoardSize' : robot.board_size,
        'RowUpdate' : robot.row_update,
        'Exit' : robot.exit
    }

    while True:
        cmd = input()
        log(file, '[>] ' + cmd)

        name = cmd.split(' ')[0]
        if name not in handler:
            continue

        params = cmd.split(' ')[1:]
        result, cmds = handler[name](params)

        for c in cmds:
            out_cmd(file, c)

        if not result:
            break


def parse_args():
    if len(sys.argv) == 2 and sys.argv[1] == '-l':
        ts = time.time()
        return datetime.datetime.fromtimestamp(ts).strftime('robot_%Y%m%d%H%M%S.txt')
    elif len(sys.argv) == 3 and sys.argv[1] == '-t':
        return sys.argv[2]

    return None


def out_cmd(file, cmd):
    log(file, '[<] ' + cmd)
    print(cmd)


def log(file, msg):
    if file:
        file.write(msg + '\n')
        file.flush()


if __name__ == '__main__':
    robot = RobotML()
    game(robot)
