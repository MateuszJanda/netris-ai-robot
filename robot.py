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

BOARD_WIDTH = 10
BORAD_HEIGHT = 20
SCR_ID = 0

EMPTY_LINE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
EMPTY_BLOCK = 0
FULL_BLOCK = 1


class Robot:
    def __init__(self):
        self.file = None
        self.board = [EMPTY_LINE for _ in range(BORAD_HEIGHT)]

    def set_log_file(self, file):
        self.file = file

    def new_pice(self, params):
        piece_id = params[0]
        out = ['Right ' + piece_id, 'Right ' + piece_id, 'Right ' + piece_id]
        return True, out

    def board_size(self, params):
        scr_id, height, width = [int(p) for p in params]

        if width != BOARD_WIDTH and height != BORAD_HEIGHT:
            self._log('[!] Validation board size fail %d %d %d %d' % (width, BOARD_WIDTH, height, BORAD_HEIGHT))
            return False, ['Exit']

        return True, []

    def row_update(self, params):
        params = [int(p) for p in params]

        if params[0] == SCR_ID:
            y = params[1]
            for x, val in enumerate(params[2:]):
                board[BORAD_HEIGHT - 1 - y][x] = FULL_BLOCK if val != 0 else EMPTY_BLOCK

        return True, []

    def exit(self, params):
        return False, []

    def _log(self, msg):
        if self.file:
            log(self.file, msg)


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
