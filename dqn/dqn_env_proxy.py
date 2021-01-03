#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import numpy as np
import os
import sys
import signal
import time
import asyncio
import datetime
import argparse
from tetris_model import TetrisModel
import config


LOG_FILE = None


def main():
    # Should prevent BrokenPipeError
    signal.getsignal(signal.SIGPIPE)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    args = parse_args()
    setup_logging(args)
    log("Start robot, PID: %d. Connection to agent at %s:%d" % (os.getpid(), config.HOST, args.port))

    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    # Start monitoring the fd file descriptor for read availability and invoke
    # callback with the specified arguments once fd is available for reading
    loop.add_reader(sys.stdin, got_robot_cmd, queue)

    future_stop = loop.create_future()

    coroutine = loop.create_connection(lambda: RobotProxy(loop, future_stop, queue), config.HOST, args.port)
    loop.run_until_complete(coroutine)

    try:
        loop.run_until_complete(future_stop)
    except KeyboardInterrupt:
        pass

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
                'Robot will try to connect with DQN agent at ' + config.HOST + ':' + str(config.PORT) + '\n',
        usage='Please try to use -h, --help for more informations',
        epilog=' \n',
        formatter_class=CustomFormatter)

    parser.add_argument('-p', '--port', required=False, action='store', default=config.PORT, dest='port',
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
    for task in asyncio.all_tasks(loop):
        task.cancel()


class RobotProxy(asyncio.Protocol):
    def __init__(self, loop, future_stop, queue):
        self._loop = loop
        self._future_stop = future_stop
        self._queue = queue

        self.model = TetrisModel(LOG_FILE)

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
        log("[!] Connection lost")
        self._future_stop.set_result(True)

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
            cmd_out.append("Rotate " + self.model.sequence_num)
            rotate -= 1

        if shift < 0:
            while shift != 0:
                cmd_out.append("Left " + self.model.sequence_num)
                shift += 1
        elif shift > 0:
            while shift != 0:
                cmd_out.append("Right " + self.model.sequence_num)
                shift -= 1

        cmd_out.append("Drop " + self.model.sequence_num)

        for cmd in cmd_out:
            self._send_to_game(cmd)

        self._tic = time.time()

    async def _wait_for_robot_cmd(self):
        """Wait for command from stdin."""
        command = await self._queue.get()
        self._handle_command(command)
        self._queue.task_done()

    def _handle_command(self, command):
        """
        Handle Netris (RobotCmd) commands.
        """
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
        self.model.update_lines_cleared(params)

        return True

    def _handle_cmd_exit(self, params):
        """Handle Exit command."""
        log("Exit command received")
        msg = self.model.create_status_message(top_row=config.EMPTY_LINE, game_is_over=True)
        self._send_to_agent(msg)
        self._future_stop.set_result(True)

        return False

    def _handle_cmd_version(self, params):
        """Handle Version command. Send to netris same command to start game."""
        log("Version command received")
        self._send_to_game("Version 1")

        return True

    def _handle_cmd_new_piece(self, params):
        """
        Handle NewPiece from netris.
        """
        self.model.update_new_piece(params)

        return True

    def _handle_cmd_board_size(self, params):
        """
        Handle BoardSize command from netris - validate height and width.
        """
        scr_id, height, width = [int(p) for p in params]

        if width != config.BOARD_WIDTH and height != config.BOARD_HEIGHT:
            raise Exception("Invalid board size fail %d %d %d %d" %
                (width, config.BOARD_WIDTH, height, config.BOARD_HEIGHT))

        return True

    def _handle_cmd_row_update(self, params):
        """
        Handle RowUpdate command from netris. Update board. This is the moment
        when action can be taken for new piece.
        """
        msg = self.model.update_row(params)
        if msg:
            self._send_to_agent(msg)

        return True

    def _send_to_game(self, cmd):
        """
        Send command to netris.
        """
        # log("[<] " + cmd.strip())
        try:
            sys.stdout.write(cmd + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, IOError):
            # Reading process terminates and closes its end of the pipe while
            # script still tries to write.
            log("[!] BrokenPipeError. Probably command was not sent.")

    def _send_to_agent(self, msg):
        """
        Send status to agent.
        """
        self._transport.write(msg.encode())


def print_board(board):
    """Print board state. For debug only."""
    log("Board")
    for line in board:
        blocks = "".join(["[]" if b else "  " for b in line])
        log(blocks)


def log(*args, **kwargs):
    """
    Print log to other terminal or file.
    """
    if LOG_FILE:
        print(*args, **kwargs, file=LOG_FILE)


if __name__ == "__main__":
    main()
