#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import traceback
import time
import datetime
import nn_model
import numpy as np
import tensorflow as tf


DEBUG_OUT = "/dev/pts/1"
# DEBUG_OUT = None

BOARD_WIDTH = 10
BORAD_HEIGHT = 20
PIECE_TYPES = 6
SHFIT_OFFSET = 5

SCREEN_ID = 0
FIRST_LINE = 19
EMPTY_BLOCK = 0
FULL_BLOCK = 1

# Disable TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class BlockTypeA:
    BT_none = 0
    BT_white = 1,
    BT_blue = 2,
    BT_magenta = 3,

                            BT_cyan, BT_yellow, BT_green, BT_red,
                            BT_wall, BT_len } BlockTypeA;


class RobotML:
    def __init__(self):
        if DEBUG_OUT:
            self.file = open(DEBUG_OUT, 'w')
            sys.stderr = self.file
        else:
            self.file = None

        self.model = nn_model.create_model()
        self.model.load_weights("only_wins_checkpoint/cp.cpkt")

        self.board = np.zeros(shape=(BORAD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.sequence_num = None
        self.fresh_piece = False
        self.piece = None

    def set_log_file(self, file):
        # self.file = file
        pass

    def new_pice(self, params):
        """
        Handle NewPiece from server. Unfortunately server provide here only
        sequence number not real piece id.
        """
        self._log("[ ] NewPiece", params)
        self.sequence_num = params[0]
        self.fresh_piece = True

    def board_size(self, params):
        """Handle BoardSize command from server. Validate height and width."""
        scr_id, height, width = [int(p) for p in params]

        if width != BOARD_WIDTH and height != BORAD_HEIGHT:
            self._log('[!] Validation board size fail %d %d %d %d' % (width, BOARD_WIDTH, height, BORAD_HEIGHT))
            return False, ['Exit']

        return True, []

    def row_update(self, params):
        """
        Handle RowUpdate command from server. Update board and take action for
        new piece.
        """
        self._log("[ ] RowUpdate", params)
        params = [int(p) for p in params]

        # Analyze data (board) that belongs only to this robot
        if params[0] != SCREEN_ID:
            return True, []

        # Update board
        y = params[1]
        for x, val in enumerate(params[2:]):
            # New piece blocks have negative values, board blocks positive
            self.board[BORAD_HEIGHT - 1 - y][x] = FULL_BLOCK if val > 0 else EMPTY_BLOCK

        # Take action if this is new piece
        cmd_out = []
        if self.fresh_piece and params[1] = FIRST_LINE:
            piece = self._extract_piece(params)
            cmd_out = self._action(piece)
            self.fresh_piece = False

        return True, cmd_out

    def _extract_piece(self, params):
        """Extract new piece from row."""
        for piece in params[2:]:
            # Block of moving piece have negative values
            if piece < 0:
                return -piece

        self._log("Missing new piece")
        raise Exception("Missing new piece.")
        return None

    def _action(self, piece):
        normalized_piece = int(piece) / PIECE_TYPES
        x_data = np.array([np.concatenate(([normalized_piece], self.board.flatten()))])

        y_shift, y_rotate = self.model.predict(x_data)
        shift = np.argmax(y_shift) - SHFIT_OFFSET
        rotate = np.argmax(y_rotate)

        cmd_out = []
        if shift < 0:
            while shift != 0:
                cmd_out.append('Left ' + self.sequence_num)
                shift += 1
        elif shift > 0:
            while shift != 0:
                cmd_out.append('Right ' + self.sequence_num)
                shift -= 1

        while rotate != 0:
            cmd_out.append('Rotate ' + self.sequence_num)
            rotate -= 1

        cmd_out.append('Drop ' + self.sequence_num)
        return cmd_out

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
