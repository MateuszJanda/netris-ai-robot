#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import os
import sys
import copy


BOARD_WIDTH = 10
BORAD_HEIGHT = 20


# Piece index and it representation. Counterclockwise rotation.
PIECE = {
    0  : [
        [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    ],
    2  : [
        [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]
    ],
    3  : [
        [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    ],
    7  : [
        [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]],
    ],
    11 : [
        [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    ],
    15 : [
        [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    ],
    17 : [
        [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    ]
}


class Action:
    def __init__(self):
        self.piece = 0
        self.shift = 0
        self.rotate = 0
        self.points = 0
        self.dump = ""

        self._raw_board = ""
        self.board = ""

    @property
    def raw_board(self):
        return self._raw_board

    @raw_board.setter
    def raw_board(self, value):
        self._raw_board = value
        self.board = [[int(piece) for piece in "{:016b}".format(line)[:BOARD_WIDTH]] for line in self._raw_board]

    def print_stats(self):
        """Print action statistics."""
        print("Shape:", self.piece)
        print("Shift:", self.shift)
        print("Rotation:", self.rotate)

        for line in self.piece_as_matrix():
            print("".join(["1" if piece else "0" for piece in line]))

    def piece_as_matrix(self):
        """Get piece as matrix in right position on board."""
        matrix = []
        for line in PIECE[self.piece][self.rotate]:
            if self.shift == 0:
                matrix.append(line)
            elif self.shift < 0:
                shift = abs(self.shift)
                matrix.append(line[shift:] + [0 for _ in range(shift)])
            else:
                shift = self.shift
                matrix.append([0 for _ in range(shift)] + line[:-shift])

        return matrix

    def print_board(self, fill=True):
        """
        Print board for given action. When fill=True empty spaces are filled
        by zeros.
        """
        for line in self.raw_board:
            line = "{:016b}".format(line)[:BOARD_WIDTH]
            if not fill:
                line = line.replace("0", " ")
            print(line)


class ActionView:
    def __init__(self, action, next_action):
        self.current_board = [[int(piece) for piece in "{:016b}".format(line)[:BOARD_WIDTH]] for line in action.raw_board]
        self.next_board = [[int(piece) for piece in "{:016b}".format(line)[:BOARD_WIDTH]] for line in next_action.raw_board]
        self.action = action

    def recreate(self):
        """Check if board can be reconstructed properly by current action."""
        shape = self.action.piece_as_matrix()

        board = self._merge_shape_and_board(shape)
        board, points = self._reduce_board(board)

        return points == self.action.points and board == self.next_board

    def _merge_shape_and_board(self, shape):
        """Move and place piece in previous board."""
        BLOCK = 1
        board = copy.deepcopy(self.current_board)

        # Move piece
        for y in range(BORAD_HEIGHT):
            # If collision then revoke actual board
            for row, line in enumerate(shape):
                for col, block in enumerate(line):
                    if self.current_board[y+row][col] and block:
                        return board

            # Fill boad with piece blocks
            board = copy.deepcopy(self.current_board)
            for row, line in enumerate(shape):
                for col, block in enumerate(line):
                    if block:
                        board[y+row][col] = BLOCK

            # If next action is out of border, then break
            if (y+1) + len(shape) > BORAD_HEIGHT:
                break

        return board

    def _reduce_board(self, board):
        """Reduce full lines and count points."""
        FULL_LINE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        EMPTY_LINE = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        # Check for full lines
        cleared_board = []
        points = 0
        for line in board:
            if line == FULL_LINE:
                points += 1
            else:
                cleared_board.append(line)

        board = copy.deepcopy(cleared_board)

        # Fill missing lines in board
        if len(board) != BORAD_HEIGHT:
            missing = BORAD_HEIGHT - len(cleared_board)
            for _ in range(missing):
                board = EMPTY_LINE + board

        return board, points

    def current_max(self):
        """Get max block height on current baord."""
        return self._max(self.current_board)

    def next_max(self):
        """Get max block height on next baord (after action/move)."""
        return self._max(self.next_board)

    def _max(self, board):
        """Get max block height on baord."""
        max_height = self.col_height(0, board)
        for col in range(1, BOARD_WIDTH):
            max_height = max(max_height, self.col_height(col, board))

        return max_height

    def current_min(self):
        """Get min block height on next baord (after action/move)."""
        return self._min(self.next_board)

    def next_min(self):
        """Get min block height on next baord (after action/move)."""
        return self._min(self.next_board)

    def _min(self, board):
        """Get min block height on baord."""
        min_height = self.col_height(0, board)
        for col in range(1, BOARD_WIDTH):
            min_height = min(min_height, self.col_height(col, board))

        return min_height

    def points(self):
        return self.action.points

    def col_height(self, col, board):
        """Return height of given column."""
        for row in range(BORAD_HEIGHT):
            if board[row][col]:
                return BORAD_HEIGHT - row

        return 0


class Game:
    def __init__(self, file_name):
        self.game = []

        with open(file_name, "r") as f:
            self.game = self._read(f)


    def _read(self, trace):
        """Reading trace data with squeezed shift and rotation."""
        BYTES_PER_LINE = 4
        game = []
        action = None

        for line in trace:
            packet = line.split()

            if packet[0] == "[>]":
                if packet[1] == "NP_newPiece":
                    if action:
                        game.append(action)
                    action = Action()
                    action.piece = int(packet[2].split("=")[1])
                elif packet[1] == "NP_left":
                    action.shift -= 1
                elif packet[1] == "NP_right":
                    action.shift += 1
                elif packet[1] == "NP_rotate":
                    action.rotate += 1
                    action.rotate %= len(PIECE[action.piece])
            elif packet[0] == "[<]" and packet[1] == "NP_points":
                action.points = int(packet[2].split("=")[1])
            elif packet[0] == "[<]" and packet[1] == "NP_boardDump":
                action.dump = packet[3].split("=")[1]
                lines = [action.dump[i:i+BYTES_PER_LINE] for i in range(0, len(action.dump), BYTES_PER_LINE)]
                action.raw_board = list(reversed([int(line, 16) for line in lines]))

        return game

    def print_stats(self):
        """Print game statistics."""
        print("Points:", sum([a.points for a in self.game]))
        print("Actions:", len(self.game))
        print("Overall pieces:", set([a.piece for a in self.game]))


    def recreate(self):
        """Return percentage of actions recondtructed in game."""
        correct = 0
        for idx in range(len(self.game) - 1):
            a = ActionView(self.game[idx], self.game[idx+1])
            if a.recreate():
                print(a.next_max(), a.next_min())
                self.game[idx+1].print_board()
                correct += 1

        return correct / (len(self.game)-1)



def save_new_trace(game, file_name):
    """Save squeezed game to new trace file."""
    with open(file_name, "w") as f:
        for a in game:
            f.write("{piece} {shift} {rotate} {points} {raw_board}\n" \
                .format(piece=a.piece, shift=a.shift, rotate=a.rotate,
                    points=a.points, raw_board=str(a.raw_board)))

