#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
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
MAX_PIECE_HEIGHT = 4
MAX_ALLOWED_GAPS = 5
MAX_CLEARED_LINES = 4
REWAD_THRESHOLDS = MAX_CLEARED_LINES + (MAX_CLEARED_LINES - 1)

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
    loop.run_until_complete(coroutine)

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
                'netris-ai-robot project github.com/MateuszJanda/netris-ai-robot\n'
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
         0: "blue log",
         2: "violet square",
         3: "azure L",
         7: "yellow mirror L",
        15: "green S",
        17: "red Z",
    }

    def __init__(self, loop, future_stop, queue):
        self._loop = loop
        self._future_stop = future_stop
        self._queue = queue

        self._board = np.zeros(shape=(BORAD_HEIGHT, BOARD_WIDTH), dtype=int)
        self._sequence_num = None
        self._fresh_piece = False
        self._piece = None
        self._round = 0
        self._lines_cleared = 0
        self._board_gaps_count = 0
        self._board_max_height = 0

        self._transport = None
        self._buffer = bytes()

        self._tic = time.time()

    def connection_made(self, transport):
        """DQN agent established connection with robot."""
        log("Connection from DQN agent")
        self._transport = transport

        # Initialize game, send first command (version)
        self._loop.create_task(self._wait_for_robot_cmd())

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
        self._buffer += data

        if b'\n' not in self._buffer:
            return

        msg = self._buffer[:self._buffer.find(b'\n')]
        self._buffer = self._buffer[self._buffer.find(b'\n') + 1:]
        shift, rotate = [int(d) for d in msg.decode().split()]
        log("Data received: shift: %d, rotate: %d" % (shift, rotate))

        cmd_out = []

        while rotate != 0:
            cmd_out.append("Rotate " + self._sequence_num)
            rotate -= 1

        if shift < 0:
            while shift != 0:
                cmd_out.append("Left " + self._sequence_num)
                shift += 1
        elif shift > 0:
            while shift != 0:
                cmd_out.append("Right " + self._sequence_num)
                shift -= 1

        cmd_out.append("Drop " + self._sequence_num)
        for c in cmd_out:
            self._send_robot_cmd(c)
        self._tic = time.time()

    async def _wait_for_robot_cmd(self):
        """Wait for command from stdin."""
        command = await self._queue.get()
        self._handle_command(command)

    def _handle_command(self, command):
        """Handle Netris (RobotCmd) commands."""
        # log("[>] " + command.strip())

        handlers = {
            "Ext:LinesCleared": self._handle_cmd_lines_cleared,
            "Exit": self._handle_cmd_exit,
            "Version": self._handle_cmd_version,
            "NewPiece": self._handle_cmd_new_piece,
            "BoardSize": self._handle_cmd_board_size,
            "RowUpdate": self._handle_cmd_row_update,
        }

        name = command.strip().split(" ")[0]
        if name == "":
            log("[!] Empty command. Should not happen.")
            return
        if name not in handlers:
            self._loop.create_task(self._wait_for_robot_cmd())
            return

        # log("Time:", time.time() - self._tic, name)

        params = command.strip().split(" ")[1:]
        continue_loop = handlers[name](params)

        if not continue_loop:
            return

        self._loop.create_task(self._wait_for_robot_cmd())

    def _handle_cmd_lines_cleared(self, params):
        """Handle Ext:LinesCleared - available only in modified Netris."""
        scr, lines_cleared = [int(p) for p in params]

        # Check if data belongs to this robot
        if scr != SCREEN_ID:
            return True

        if lines_cleared:
            log("LinesCleared:", lines_cleared)

        self._lines_cleared = lines_cleared
        return True

    def _handle_cmd_exit(self, params):
        """Handle Exit command."""
        log("Exit command received")
        self._send_update_to_agent(top_row=EMPTY_LINE, game_is_over=True)
        self._future_stop.set_result(True)

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
        self._sequence_num = params[0]
        self._fresh_piece = True
        self._round += 1

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

        # Netris inform about switch from "piece block" to "fixed block" starting
        # from second RowUpdate command after NewPiece command. This is to late,
        # for prediction, so better is assume that first line is always empty.
        if y != TOP_LINE:
            for x, val in enumerate(row):
                self._board[BORAD_HEIGHT - 1 - y][x] = FULL_BLOCK if val else EMPTY_BLOCK

        # Send board to agent if this is new piece
        if self._fresh_piece and y == TOP_LINE:
            self._send_update_to_agent(top_row=row, game_is_over=False)
            self._fresh_piece = False

        return True

    def _send_update_to_agent(self, top_row, game_is_over):
        """Send update to DQN agent."""
        new_board = self._board_with_piece_bits(top_row)
        flat_board = "".join([("%0.2f " % val) for val in new_board])
        reward = self._reward(game_is_over)
        game_is_over = int(game_is_over)

        log("reward:", reward)

        # Format message and send
        report = str(game_is_over) + " " + str(reward) + " " + flat_board + "\n"
        self._transport.write(report.encode())

    def _reward(self, game_is_over):
        """
        Calculate final reward. In competitive game this values, where extra
        lines are added by enemy, will be invalid.

        Return value in range [0, 1]
        """

        # Punish for not filling valleys
        if self._lines_cleared == 0:
            top, bottom = self._calc_valley()
            penalty_valley = (top - bottom) / (BORAD_HEIGHT - 1)
            if penalty_valley < 0 or penalty_valley > 1:
                raise Exception("Penalty out of range: %f, (top, bottom) = (%d, %d)"
                    % (penalty_valley, top, bottom))
        # When cleared lines reveal new bottom then skip
        else:
            penalty_valley = 0

        # Punish for creating gaps
        gaps_count = self._calc_gaps()
        penalty_gaps = min(max(0, gaps_count - self._board_gaps_count), MAX_ALLOWED_GAPS) / MAX_ALLOWED_GAPS
        if penalty_gaps < 0 or penalty_gaps > 1:
            raise Exception("Penalty out of range: %f, gaps = %d, board_gaps = %d"
                % (penalty_gaps, gaps_count, self._board_gaps_count))
        self._board_gaps_count = gaps_count

        # # Punish for increasing the height
        # max_height = self._calc_height()
        # penalty_height = max(0, max_height - self._board_max_height) / MAX_PIECE_HEIGHT
        # if penalty_height < 0 or penalty_height > 1:
        #     raise Exception("Penalty out of range: %f, max_height = %d, board_height = %d"
        #         % (penalty_height, max_height, self._board_max_height))
        # self._board_max_height = max_height

        # Punish for ending the game
        if game_is_over:
            score = 0
        # Reward for adding piece
        elif self._lines_cleared == 0:
            score = 0.08
        # Reward for clearing lines
        else:
            score = (self._lines_cleared + (self._lines_cleared - 1)) / REWAD_THRESHOLDS

        # Reset counter
        self._lines_cleared = 0

        # Normalize reward with punishment
        reward = score * (0.5 * (1 - penalty_valley) + 0.5 * (1 - penalty_gaps))

        if reward < 0 or reward > 1:
            raise Exception("Reward out of range: %f" % (reward))

        return reward

    def _calc_gaps(self):
        """Count all gaps (blocks that can't be reached in next tour)."""
        gaps_count = 0
        for x in range(BOARD_WIDTH):
            is_roof = False
            for y in range(BORAD_HEIGHT):
                if self._board[y][x] == FULL_BLOCK and not is_roof:
                    is_roof = True
                elif self._board[y][x] == EMPTY_BLOCK and is_roof:
                    gaps_count += 1

        return gaps_count

    def _calc_valley(self):
        """Get max and min height."""
        max_height = 0
        min_height = BORAD_HEIGHT - 1
        for x in range(BOARD_WIDTH):
            for y in range(BORAD_HEIGHT):
                if self._board[y][x] == FULL_BLOCK:
                    max_height = max(max_height, BORAD_HEIGHT - y)
                    min_height = min(min_height, BORAD_HEIGHT - y)
                    break
                elif y == BORAD_HEIGHT - 1 and self._board[y][x] == EMPTY_BLOCK:
                    max_height = max(max_height, 0)
                    min_height = min(min_height, 0)
                    break

        return max_height, min_height

    def _calc_height(self):
        """Calculate max height."""
        max_height = 0
        for x in range(BOARD_WIDTH):
            for y in range(BORAD_HEIGHT):
                if self._board[y][x] == FULL_BLOCK:
                    max_height = max(max_height, BORAD_HEIGHT - y)
                    break

        return max_height

    def _board_with_piece_bits(self, top_row):
        """Create flat board with four blocks representing piece."""
        piece_bits = self._piece_as_bits(top_row)

        # Combine board with piece bits
        new_board = np.copy(self._board).astype('float')
        for x in range(BOARD_WIDTH):
            if 3 < x < 8:
                new_board[0][x] = piece_bits[x - 4]

        return new_board.flatten()

    def _piece_as_bits(self, top_row):
        """Extract new piece and create his four bits representation."""
        for color_type in top_row:
            # Block of moving piece have negative values
            if color_type < 0:
                log("Extracted piece:", self._piece_name_by_color(color_type))
                piece = self.COLOR_TO_PIECE[color_type]
                # Piece +1 because 0 is reserved for "empty block"
                return [float(ch) for ch in list('{0:04b}'.format(piece + 1))]

        return [0.0, 0.0, 0.0, 0.0]

    def _normalized_board(self, top_row):
        """Create flat board with normalized values."""
        norm_piece = self._normalized_piece(top_row)

        # Combine board with normalized piece
        norm_board = np.copy(self._board).astype('float')
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
                # Piece +1 because 0 is reserved for "empty block"
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
        blocks = "".join(["1" if b else " " for b in line])
        log(blocks)


def log(*args, **kwargs):
    """Print log to other terminal or file."""
    if LOG_FILE:
        # print(*args, **kwargs, file=LOG_FILE)
        pass


if __name__ == "__main__":
    main()
