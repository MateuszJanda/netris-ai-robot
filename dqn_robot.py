#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import os
import sys
import time
import asyncio
import datetime
import argparse
import numpy as np


BOARD_WIDTH = 10
BORAD_HEIGHT = 20
SHFIT_OFFSET = 5

SCREEN_ID = 0
TOP_LINE = 19
EMPTY_BLOCK = 0
FULL_BLOCK = 1

LOG_FILE = None


def main():
    args = parse_args()
    setup_logging(args)

    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    # Start monitoring the fd file descriptor for read availability and invoke
    # callback with the specified arguments once fd is available for reading
    loop.add_reader(sys.stdin, got_netris_data, loop, queue)

    coroutine = loop.create_server(lambda: RobotProxy(loop, queue), '127.0.0.1', 9898)
    server = loop.run_until_complete(coroutine)

    # CTRL+C to quit
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    # Close the server
    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()

    LOG_FILE.close()


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Netris proxy robot for reinforcement learning (DQN)\n'
                'Mateusz Janda (c) <mateusz janda at gmail com>\n'
                'netris-ai-robot project github.com/MateuszJanda/sloper\n',
        usage='Please try to use -h, --help for more informations',
        epilog='',
        formatter_class=CustomFormatter)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-l', '--log-to-file', required=False, action='store_true', dest='log_file',
                       help='Log to file - robot_%%Y%%m%%d%%H%%M%%S.txt')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-t', '--log-in-terminal', required=False, metavar='<pts>', dest='log_terminal',
                       help='Log in terminal - e.g. /dev/pts/1')

    args = parser.parse_args()
    if args.log_file:
        ts = time.time()
        args.log_name = datetime.datetime.fromtimestamp(ts).strftime("robot_%Y%m%d%H%M%S.txt")
    elif args.log_terminal:
        args.log_name = args.log_terminal
    else:
        args.log_name = None

    return args


def setup_logging(args):
    """Setup logging."""
    global LOG_FILE

    if args.log_name:
        LOG_FILE = open(args.log_name, "w")
        sys.stderr = LOG_FILE


def got_netris_data(loop, queue):
    """Setup task waiting for Netris commands."""
    loop.create_task(queue.put(sys.stdin.readline()))


class RobotProxy(asyncio.Protocol):
    COLOR_TO_PIECE = {
        -1: 4,      # Piece Id: 11
        -2: 0,      # Piece Id: 0
        -3: 1,      # Piece Id: 2
        -4: 2,      # Piece Id: 3
        -5: 3,      # Piece Id: 7
        -6: 5,      # Piece Id: 15
        -7: 6,      # Piece Id: 17
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

    def __init__(self, loop, queue):
        self.loop = loop
        self.queue = queue

        self.board = np.zeros(shape=(BORAD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.sequence_num = None
        self.fresh_piece = False
        self.piece = None

        self.transport = None

    def connection_made(self, transport):
        """DQN agent established connection with robot."""
        log('Connection from DQN agent')
        self.transport = transport

        # Init game, send firt command (version)
        self._send_netris_cmd("Version 1")
        self.loop.create_task(self._wait_for_netrs_cmd())

    def _send_netris_cmd(self, cmd):
        """Send command to server."""
        # log("[<] " + cmd)
        sys.stdout.write(cmd + "\n")

    def data_received(self, data):
        """Data received from DQN agent."""
        message = data.decode()
        log('Data received: {!r}'.format(message))

    async def _wait_for_netrs_cmd(self):
        """Wait for command from stdin."""
        command = await self.queue.get()
        self._handle_command(command)

    def _handle_command(self, command):
        handlers = {
            "Exit" : self._handle_cmd_exit,
            "NewPiece" : self._hanle_cmd_new_pice,
            "BoardSize" : self._handle_cmd_board_size,
            "RowUpdate" : self._handle_cmd_row_update,
        }

        name = cmd.split(" ")[0]
        if name not in handlers:
            self.loop.create_task(self._wait_for_netrs_cmd())
            return

        params = cmd.split(" ")[1:]
        continue_loop, cmd_reponses = handlers[name](params)

        for c in cmd_reponses:
            self._send_netris_cmd(c)

        if not continue_loop:
            return

        self.loop.create_task(self._wait_for_netrs_cmd())


    def _handle_cmd_exit(self, params):
        """Handle Exit command."""
        return True, []

    def _hanle_cmd_new_pice(self, params):
        """
        Handle NewPiece from server. Unfortunately server provide here only
        sequence number not real piece id.
        """
        self.sequence_num = params[0]
        self.fresh_piece = True

        return True, []

    def _handle_cmd_board_size(self, params):
        """
        Handle BoardSize command from server. Just to validate height and
        width.
        """
        scr_id, height, width = [int(p) for p in params]

        if width != BOARD_WIDTH and height != BORAD_HEIGHT:
            log("[!] Validation board size fail %d %d %d %d" % (width, BOARD_WIDTH, height, BORAD_HEIGHT))
            return False, ["Exit"]

        return True, []

    def _handle_cmd_row_update(self, params):
        """
        Handle RowUpdate command from server. Update board. This is the moment
        when action can be taken for new piece.
        """
        params = [int(p) for p in params]

        # Analyze data (board) that belongs only to this robot
        if params[0] != SCREEN_ID:
            return True, []

        y = params[1]
        # Server inform about switch from "piece block" to "fixed block" starting
        # from second RowUpdate command after NewPiece command. This is to late,
        # for prediction, so better is assume that first line is always empty.
        if y != TOP_LINE:
            for x, val in enumerate(params[2:]):
                self.board[BORAD_HEIGHT - 1 - y][x] = FULL_BLOCK if val else EMPTY_BLOCK

        # Send board to agent if this is new piece
        cmd_out = []
        if self.fresh_piece and y == TOP_LINE:
            self.send_update_to_agent(params[2:])
            self.fresh_piece = False
            # self._print_board()

        return True, []

    def send_update_to_agent(self, top_row):
        norm_board = self._normalized_board(top_row)

        score = ""
        game_is_over = ""
        board = "".join([("%0.2f " % val) for val in norm_board])

        report = score + " " game_is_over + " " + board

        self.transport.write(report.encode())

    def _normalized_board(self, top_row):
        norm_piece = self._normalized_piece(top_row)

        # Combine board with normalized piece
        norm_board = np.copy(self.board)
        for x, color_type in enumerate(top_row):
            if color_type < 0:
                norm_board[0][x] = norm_piece

        return norm_board.flatten()

    def _normalized_piece(self, top_row):
        """Extract new piece (order number) from row."""
        for color_type in top_row:
            # Block of moving piece have negative values
            if color_type < 0:
                log("Extracted piece:", self._piece_name_by_color(color_type))
                piece = self.COLOR_TO_PIECE[color_type]
                # All pieces with full block
                all_pieces = len(self.PIECE_TO_PIECE_ID) + 1
                # Piece +1 because 0 is for "empty block"
                return (piece + 1) / all_pieces

        raise Exception("Missing new piece.")
        return None

    def _piece_name_by_color(color_type):
        piece = self.COLOR_TO_PIECE[color_type]
        piece_id = self.PIECE_TO_PIECE_ID[piece]
        return self.PIECE_ID_TO_NAME[piece_id]



    # def _action_commands(self, piece):
    #     """Determine next robot move."""
    #     shift, rotate = self._predict_action(piece)

    #     cmd_out = []
    #     if shift < 0:
    #         while shift != 0:
    #             cmd_out.append("Left " + self.sequence_num)
    #             shift += 1
    #     elif shift > 0:
    #         while shift != 0:
    #             cmd_out.append("Right " + self.sequence_num)
    #             shift -= 1

    #     while rotate != 0:
    #         cmd_out.append("Rotate " + self.sequence_num)
    #         rotate -= 1

    #     cmd_out.append("Drop " + self.sequence_num)
    #     return cmd_out

    # def _predict_action(self, piece):
    #     """Prediction action by piece and current board state."""
    #     normalized_piece = piece / (len(self.COLOR_TO_PIECE) - 1)
    #     x_data = np.array([np.concatenate(([normalized_piece], self.board.flatten()))])

    #     y_shift, y_rotate = self.model.predict(x_data)
    #     shift = np.argmax(y_shift) - SHFIT_OFFSET
    #     rotate = np.argmax(y_rotate)

    #     return shift, rotate

    def _print_board(self):
        """Print current board state. For debug only."""
        log("Board")
        for line in self.board:
            l = "".join(["1" if b else " " for b in line])
            log(l)

def log(*args, **kwargs):
    """Print log to other terminal or file."""
    if LOG_FILE:
        print(*args, **kwargs, file=LOG_FILE)


if __name__ == "__main__":
    main()