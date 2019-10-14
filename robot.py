#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""
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
    def __init__(self, f):
        self.f = f
        self.board = [EMPTY_LINE for _ in range(BORAD_HEIGHT)]
        self.piece_id = None

    def new_pice(self, params):
        self.piece_id = params[0]
        return True, ['Message Lorem ipsum']

    def board_size(self, params):
        scr_id, height, width = [int(p) for p in params]

        if width != BOARD_WIDTH and height != BORAD_HEIGHT:
            log(self.f, '[!] Validation board size fail %d %d %d %d' % (width, BOARD_WIDTH, height, BORAD_HEIGHT))
            return False, ['Exit']

        return True, []

    def row_update(self, params):
        params = [int(p) for p in params]

        if params[0] == SCR_ID:
            y = params[1]
            for x, val in enumerate(params[2:]):
                board[BORAD_HEIGHT - 1 - y][x] = FULL_BLOCK if val != 0 else EMPTY_BLOCK

        return True, []

    def time_stamp(self, params):
        if not self.piece_id:
            return True, []

        out = ['Right ' + self.piece_id, 'Right ' + self.piece_id, 'Right ' + self.piece_id]
        self.piece_id = None
        return True, out


    def exit(self, params):
        return False, []


def main():
    ts = time.time()
    # log_name = datetime.datetime.fromtimestamp(ts).strftime('robot_%Y%m%d%H%M%S.txt')
    log_name = '/dev/pts/3'

    try:
        with open(log_name, 'w') as f:
            fun(f)
    except:
        with open(log_name, 'w') as f:
            traceback.print_exc(file=f)

    return 0


def fun(f):
    robot = Robot(f)
    out_cmd(f, 'Version 1')

    handler = {
        'NewPiece' : robot.new_pice,
        'BoardSize' : robot.board_size,
        'TimeStamp' : robot.time_stamp,
        'Exit' : robot.exit
    }

    while True:
        cmd = input()
        log(f, '[>] ' + cmd)

        name = cmd.split(' ')[0]
        if name not in handler:
            continue

        params = cmd.split(' ')[1:]
        result, cmds = handler[name](params)

        for c in cmds:
            out_cmd(f, c)

        if not result:
            break


def out_cmd(file, cmd):
    log(file, '[<] ' + cmd)
    print(cmd)


def log(file, msg):
    file.write(msg + '\n')
    file.flush()


if __name__ == '__main__':
    main()
