#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import sys
import asyncio
from tetris_model import TetrisModel
from converter import Converter
import config


class ProxyRobot(asyncio.Protocol):
    def __init__(self, loop, future_stop, queue, log_file=None):
        self._loop = loop
        self._future_stop = future_stop
        self._queue = queue
        self._log_file = log_file

        self.model = TetrisModel(self._log_file)
        self.converter = Converter(self.model, self._log_file)

        self._transport = None
        self._buffer = bytes()

    def connection_made(self, transport):
        """DQN agent established connection with robot."""
        self._log("Connection from DQN agent")
        self._transport = transport

        # Initialize game, send first command (version)
        self._loop.create_task(self._wait_for_robot_cmd())

    def connection_lost(self, exc):
        self._log("[!] Connection lost")
        self._future_stop.set_result(True)

    def data_received(self, data):
        """Data received from DQN agent, determine next robot move."""
        self._buffer += data

        if b'\n' not in self._buffer:
            return

        msg = self._buffer[:self._buffer.find(b'\n')]
        self._buffer = self._buffer[self._buffer.find(b'\n') + 1:]
        shift, rotate = [int(d) for d in msg.decode().split()]
        self._log("Data received: shift: %d, rotate: %d" % (shift, rotate))

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
            self._log("[!] Empty command. Should not happen.")
            return
        if name not in handlers:
            self._loop.create_task(self._wait_for_robot_cmd())
            return

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
        self._log("Exit command received")
        msg = self.converter.create_status_message(top_row=config.EMPTY_LINE, game_is_over=True)
        self._send_to_agent(msg)
        self._future_stop.set_result(True)

        return False

    def _handle_cmd_version(self, params):
        """Handle Version command. Send to netris same command to start game."""
        self._log("Version command received")
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
        row = self.model.update_row(params)
        if row:
            msg = self.converter.create_status_message(top_row=row, game_is_over=False)
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
            self._log("[!] BrokenPipeError. Probably command was not sent.")

    def _send_to_agent(self, msg):
        """
        Send status to agent.
        """
        self._transport.write(msg.encode())

    def _log(self, *args, **kwargs):
        """
        Print log to other terminal or file.
        """
        if self._log_file:
            print(*args, **kwargs, file=self._log_file)
