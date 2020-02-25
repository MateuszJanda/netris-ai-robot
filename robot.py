#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import sys
import traceback
import time
import datetime
import numpy as np


BOARD_WIDTH = 10
BORAD_HEIGHT = 20
SCR_ID = 0

EMPTY_LINE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
EMPTY_BLOCK = 0
FULL_BLOCK = 1


class Robot:
    def __init__(self):
        # self.file = None
        self.file = open('/dev/pts/2', 'w')

        self.board = np.zeros(shape=(BORAD_HEIGHT, BOARD_WIDTH), dtype=int)


    def set_log_file(self, file):
        # self.file = file
        pass

    def new_pice(self, params):
        piece_id = params[0]
        out = ['Right ' + piece_id, 'Right ' + piece_id, 'Right ' + piece_id, 'Message Disp asdf']
        # self._print_board()
        return True, out

    def board_size(self, params):
        scr_id, height, width = [int(p) for p in params]

        if width != BOARD_WIDTH and height != BORAD_HEIGHT:
            self._log2('[!] Validation board size fail %d %d %d %d' % (width, BOARD_WIDTH, height, BORAD_HEIGHT))
            return False, ['Exit']

        return True, []

    def row_update(self, params):
        params = [int(p) for p in params]

        if params[0] != SCR_ID:
            return True, []

        y = params[1]
        # self._log2(y)
        for x, val in enumerate(params[2:]):
            self._log2('----')
            self._log2('asdf %d' % 45)
            self._log2(x, y, val)
            self._log2(BORAD_HEIGHT - 1 - y)
            self._log2(FULL_BLOCK if val != 0 else EMPTY_BLOCK)
            self.board[BORAD_HEIGHT - 1 - y][x] = FULL_BLOCK if val != 0 else EMPTY_BLOCK
            # board[0][0] = FULL_BLOCK if val != 0 else EMPTY_BLOCK
            # self.board[0][0] = 1
            self._log2('end')


        return True, []

    def exit(self, params):
        return True, []

    def _print_board(self):
        for line in self.board:
            l = ''.join(['1' if b else ' ' for b in line])
            # print(l, file=self.stderr)
            self.stderr.write(l)
            # print(l)

    def _log2(self, *args, **kwargs):
        if self.file:
            print(*args, **kwargs, file=self.file)



def game(robot):
    log_name = parse_args()

    try:
        if log_name:
            with open(log_name, 'w') as file:
                loop(file, robot)
        else:
            loop(None, robot)
    except:
        if log_name:
            with open(log_name, 'w') as file:
                traceback.print_exc(file=file)


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
    robot = Robot()
    game(robot)
