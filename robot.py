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

LOG_FILE = None

BOARD_WIDTH = 10
BORAD_HEIGHT = 20
SHFIT_OFFSET = 5

SCREEN_ID = 0
TOP_LINE = 19
EMPTY_BLOCK = 0
FULL_BLOCK = 1

# Disable TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class Robot:
    COLOR_TO_PIECE = {
        1: 4,      # 11
        2: 0,      # 0
        3: 1,      # 2
        4: 2,      # 3
        5: 3,      # 7
        6: 5,      # 15
        7: 6,      # 17
    }

    PIECE_TO_PIECE_ID = {
        4: 11,
        0: 0,
        1: 2,
        2: 3,
        3: 7,
        5: 15,
        6: 17,
    }

    PIECE_ID_TO_NAME = {
        11: "white pyramid",
        0 : "blue log",
        2 : "violet square",
        3 : "azure L",
        7 : "yellow mirror L",
        15: "green S",
        17: "red Z",
    }

    def __init__(self):
        self.model = nn_model.create_model()
        self.model.load_weights("only_wins_checkpoint/cp.cpkt")

        self.board = np.zeros(shape=(BORAD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.sequence_num = None
        self.fresh_piece = False
        self.piece = None

    def exit(self, params):
        """Handle Exit command."""
        return True, []

    def new_pice(self, params):
        """
        Handle NewPiece from server. Unfortunately server provide here only
        sequence number not real piece id.
        """
        self.sequence_num = params[0]
        self.fresh_piece = True

        return True, []

    def board_size(self, params):
        """Handle BoardSize command from server. Validate height and width."""
        scr_id, height, width = [int(p) for p in params]

        if width != BOARD_WIDTH and height != BORAD_HEIGHT:
            log('[!] Validation board size fail %d %d %d %d' % (width, BOARD_WIDTH, height, BORAD_HEIGHT))
            return False, ['Exit']

        return True, []

    def row_update(self, params):
        """
        Handle RowUpdate command from server. Update board and take action for
        new piece.
        """
        params = [int(p) for p in params]

        # Analyze data (board) that belongs only to this robot
        if params[0] != SCREEN_ID:
            return True, []

        y = params[1]
        # Server inform about switch from piece block to fixed block starting
        # from second RowUpdate command after NewPiece command. This is to late,
        # for prediction, so better is assume that first line is always empty.
        if y != TOP_LINE:
            for x, val in enumerate(params[2:]):
                self.board[BORAD_HEIGHT - 1 - y][x] = FULL_BLOCK if val else EMPTY_BLOCK

        # Take action if this is new piece
        cmd_out = []
        if self.fresh_piece and y == TOP_LINE:
            piece = self._extract_piece(params)
            cmd_out = self._action_commands(piece)
            self.fresh_piece = False
            # self._print_board()

        return True, cmd_out

    def _extract_piece(self, params):
        """Extract new piece order number from row."""
        for color_type in params[2:]:
            # Block of moving piece have negative values
            if color_type < 0:
                color_type = -color_type
                log("Extracted piece:", Robot.PIECE_ID_TO_NAME[Robot.PIECE_TO_PIECE_ID[Robot.COLOR_TO_PIECE[color_type]]])
                return Robot.COLOR_TO_PIECE[color_type]

        log("Missing new piece")
        raise Exception("Missing new piece.")
        return None

    def _action_commands(self, piece):
        """Determine next robot move."""
        shift, rotate = self._predict_action(piece)

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

    def _predict_action(self, piece):
        """Prediction action by piece and current board state."""
        normalized_piece = piece / (len(Robot.COLOR_TO_PIECE) - 1)
        x_data = np.array([np.concatenate(([normalized_piece], self.board.flatten()))])

        y_shift, y_rotate = self.model.predict(x_data)
        shift = np.argmax(y_shift) - SHFIT_OFFSET
        rotate = np.argmax(y_rotate)

        return shift, rotate

    def _print_board(self):
        """Print current board state. For debug purpose."""
        log('Board')
        for line in self.board:
            l = ''.join(['1' if b else ' ' for b in line])
            log(l)


def main():
    log_name = parse_args()
    robot = Robot()

    global LOG_FILE

    if log_name:
        LOG_FILE = open(log_name, 'w')
        sys.stderr = LOG_FILE
    elif DEBUG_OUT:
        LOG_FILE = open(DEBUG_OUT, 'w')
        sys.stderr = LOG_FILE

    try:
        command_loop(robot)
    finally:
        if log_name:
            LOG_FILE.close()


def command_loop(robot):
    """Handle command from server."""
    send_command('Version 1')

    handler = {
        'Exit' : robot.exit,
        'NewPiece' : robot.new_pice,
        'BoardSize' : robot.board_size,
        'RowUpdate' : robot.row_update,
    }

    while True:
        cmd = input()
        # log('[>] ' + cmd)

        name = cmd.split(' ')[0]
        if name not in handler:
            continue

        params = cmd.split(' ')[1:]
        result, cmds = handler[name](params)

        for c in cmds:
            send_command(c)

        if not result:
            break


def parse_args():
    """Parse command line arguments."""
    if len(sys.argv) == 2 and sys.argv[1] == '-l':
        return create_log_name()
    elif len(sys.argv) == 3 and sys.argv[1] == '-t':
        return sys.argv[2]

    return None


def create_log_name():
    """Generate log file name."""
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('robot_%Y%m%d%H%M%S.txt')


def send_command(cmd):
    """Send command to server."""
    # log('[<] ' + cmd)
    print(cmd)


def log(*args, **kwargs):
    """Print log to other terminal or file."""
    if LOG_FILE:
        print(*args, **kwargs, file=LOG_FILE)


if __name__ == '__main__':
    main()
