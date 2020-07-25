#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import os
import sys
import signal
import time
import asyncio
import datetime
import argparse
import numpy as np


HOST = "127.0.0.1"
PORT = 9800

BOARD_WIDTH = 10
BORAD_HEIGHT = 20
SHFIT_OFFSET = 5

SCREEN_ID = 0
TOP_LINE = 19
EMPTY_BLOCK = 0
FULL_BLOCK = 1
EMPTY_LINE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

LOG_FILE = None


def main():
    # Should prevent BrokenPipeError
    signal.getsignal(signal.SIGPIPE)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    args = parse_args()
    setup_logging(args)
    log("Start robot, PID: %d. Connection to agent at %s:%d" % (os.getpid(), HOST, args.port))

    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    # Start monitoring the fd file descriptor for read availability and invoke
    # callback with the specified arguments once fd is available for reading
    loop.add_reader(sys.stdin, got_robot_cmd, queue)

    future_stop = loop.create_future()
    future_stop.add_done_callback(cancel_all_task)

    coroutine = loop.create_connection(lambda: RobotProxy(loop, future_stop, queue), HOST, args.port)
    client = loop.run_until_complete(coroutine)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        cancel_all_task()

    log("Stop robot, PID:", os.getpid())
    if LOG_FILE:
        LOG_FILE.close()


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Netris proxy robot for reinforcement learning (DQN)\n'
                'Mateusz Janda (c) <mateusz janda at gmail com>\n'
                'netris-ai-robot project github.com/MateuszJanda/sloper\n'
                '\n'
                'Robot will try to connect with DQN agent at ' + HOST + ':' + str(PORT) + '\n',
        usage='Please try to use -h, --help for more informations',
        epilog=' \n',
        formatter_class=CustomFormatter)

    parser.add_argument('-p', '--port', required=False, action='store', default=PORT, dest='port',
                        help='Connect to DQN server port')

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

    args.port = int(args.port)

    return args


def setup_logging(args):
    """Setup logging."""
    global LOG_FILE

    if args.log_name:
        LOG_FILE = open(args.log_name, "w")
        sys.stderr = LOG_FILE


def got_robot_cmd(queue):
    """Setup task waiting for Netris/Robot commands."""
    loop = asyncio.get_event_loop()
    loop.create_task(queue.put(sys.stdin.readline()))


def cancel_all_task(result=None):
    log("Cancel all tasks")
    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks():
        task.cancel()
    loop.create_task(stop_loop())


async def stop_loop():
    log("Stop loop")
    loop = asyncio.get_event_loop()
    loop.stop()


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

    def __init__(self, loop, future_stop, queue):
        self.loop = loop
        self.future_stop = future_stop
        self.queue = queue

        self.board = np.zeros(shape=(BORAD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.sequence_num = None
        self.fresh_piece = False
        self.piece = None
        self.round = 0
        self.lines_cleared = 0

        self.transport = None
        self.buffer = bytes()

        self.tic = time.time()

    def connection_made(self, transport):
        """DQN agent established connection with robot."""
        log("Connection from DQN agent")
        self.transport = transport

        # Initialize game, send first command (version)
        # self._send_robot_cmd("Version 1")
        self.loop.create_task(self._wait_for_robot_cmd())

    def connection_lost(self, exc):
        log("[!] Connection lost. Should be handled?")

    def _send_robot_cmd(self, cmd):
        """Send command to server."""
        # log("[<] " + cmd.strip())
        try:
            sys.stdout.write(cmd + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, IOError):
            # Reading process terminates and closes its end of the pipe while
            # script still tries to write.
            log("[!] BrokenPipeError. Probably command was not sent.")

    def data_received(self, data):
        """Data received from DQN agent, determine next robot move."""
        self.buffer += data

        if b'\n' not in self.buffer:
            return

        msg = self.buffer[:self.buffer.find(b'\n')]
        self.buffer = self.buffer[self.buffer.find(b'\n') + 1:]
        shift, rotate = [int(d) for d in msg.decode().split()]
        log("Data received: shift: %d, rotate: %d" % (shift, rotate))

        cmd_out = []
        if shift < 0:
            while shift != 0:
                cmd_out.append("Left " + self.sequence_num)
                shift += 1
        elif shift > 0:
            while shift != 0:
                cmd_out.append("Right " + self.sequence_num)
                shift -= 1

        while rotate != 0:
            cmd_out.append("Rotate " + self.sequence_num)
            rotate -= 1

        cmd_out.append("Drop " + self.sequence_num)
        for c in cmd_out:
            self._send_robot_cmd(c)
        self.tic = time.time()

    async def _wait_for_robot_cmd(self):
        """Wait for command from stdin."""
        command = await self.queue.get()
        self._handle_command(command)

    def _handle_command(self, command):
        """Handle Netris (RobotCmd) commands."""
        # log("[>] " + command.strip())

        handlers = {
            "Ext:LinesCleared" : self._handle_cmd_lines_cleared,
            "Exit" : self._handle_cmd_exit,
            "Version" : self._handle_cmd_version,
            "NewPiece" : self._handle_cmd_new_piece,
            "BoardSize" : self._handle_cmd_board_size,
            "RowUpdate" : self._handle_cmd_row_update,
        }

        name = command.strip().split(" ")[0]
        if name == "":
            log("[!] Empty command. Should not happen.")
            return
        if name not in handlers:
            self.loop.create_task(self._wait_for_robot_cmd())
            return

        # log("Time:", time.time() - self.tic, name)

        params = command.strip().split(" ")[1:]
        continue_loop = handlers[name](params)

        if not continue_loop:
            return

        self.loop.create_task(self._wait_for_robot_cmd())

    def _handle_cmd_lines_cleared(self, params):
        """Handle Ext:LinesCleared - available only in modified Netris."""
        scr, lines_cleared = [int(p) for p in params]

        # Check if data belongs to this robot
        if scr != SCREEN_ID:
            return True

        if lines_cleared:
            log("LinesCleared:", lines_cleared)

        self.lines_cleared = lines_cleared
        return True

    def _handle_cmd_exit(self, params):
        """Handle Exit command."""
        log("Exit command received")
        self._send_update_to_agent(top_row=EMPTY_LINE, game_is_over=True)
        self.future_stop.set_result(True)

        return False

    def _handle_cmd_version(self, params):
        """Handle Version command. Send to Netris same command to start game."""
        log("Version command received")
        self._send_robot_cmd("Version 1")

        return True

    def _handle_cmd_new_piece(self, params):
        """
        Handle NewPiece from server. Unfortunately server provide here only
        sequence number not real piece id.
        """
        self.sequence_num = params[0]
        self.fresh_piece = True
        self.round += 1

        return True

    def _handle_cmd_board_size(self, params):
        """
        Handle BoardSize command from server. Just to validate height and
        width.
        """
        scr_id, height, width = [int(p) for p in params]

        if width != BOARD_WIDTH and height != BORAD_HEIGHT:
            log("[!] Validation board size fail %d %d %d %d" % (width, BOARD_WIDTH, height, BORAD_HEIGHT))
            self._send_robot_cmd("Exit")
            return False

        return True

    def _handle_cmd_row_update(self, params):
        """
        Handle RowUpdate command from server. Update board. This is the moment
        when action can be taken for new piece.
        """
        scr, y, *row = [int(p) for p in params]

        # Analyze data (board) that belongs only to this robot
        if scr != SCREEN_ID:
            return True

        # Server inform about switch from "piece block" to "fixed block" starting
        # from second RowUpdate command after NewPiece command. This is to late,
        # for prediction, so better is assume that first line is always empty.
        if y != TOP_LINE:
            for x, val in enumerate(row):
                self.board[BORAD_HEIGHT - 1 - y][x] = FULL_BLOCK if val else EMPTY_BLOCK

        # Send board to agent if this is new piece
        if self.fresh_piece and y == TOP_LINE:
            self._send_update_to_agent(top_row=row, game_is_over=False)
            self.fresh_piece = False

        return True

    def _send_update_to_agent(self, top_row, game_is_over):
        """Send update to DQN agent."""
        norm_board = self._normalized_board(top_row)
        flat_board = "".join([("%0.2f " % val) for val in norm_board])

        if game_is_over:
            score = -5
        else:
            score = self.lines_cleared**2 * 100 + 1
        self.lines_cleared = 0

        game_is_over = int(game_is_over)

        report = str(game_is_over) + " " + str(score) + " " + flat_board + "\n"
        self.transport.write(report.encode())

    def _normalized_board(self, top_row):
        """Create flat board with normalized values."""
        norm_piece = self._normalized_piece(top_row)

        # Combine board with normalized piece
        norm_board = np.copy(self.board).astype('float')
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

        return None

    def _piece_name_by_color(self, color_type):
        """Convert color id to piece name."""
        piece = self.COLOR_TO_PIECE[color_type]
        piece_id = self.PIECE_TO_PIECE_ID[piece]
        return self.PIECE_ID_TO_NAME[piece_id]


def print_board(board):
    """Print board state. For debug only."""
    log("Board")
    for line in board:
        l = "".join(["1" if b else " " for b in line])
        log(l)


def log(*args, **kwargs):
    """Print log to other terminal or file."""
    if LOG_FILE:
        print(*args, **kwargs, file=LOG_FILE)


if __name__ == "__main__":
    main()
