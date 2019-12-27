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


class SingleMove:
    def __init__(self):
        self.shape = 0
        self.shift = 0
        self.rotate = 0
        self.points = 0
        self.board = ""
        self.raw_board = ""


class GameAnalyzer:
    # Shape number and this representation. Counterclockwise rotation.
    SHAPES = {
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


    def __init__(self, game):
        self.game = game

    def print_game_stats(self):
        """Print game statistics."""
        print("[+] Game stats:")
        print("Points:", sum([m.points for m in self.game]))
        print("game:", len(self.game))
        print("Overall blocks:", set([m.shape for m in self.game]))


    def print_move_stats(self, move):
        """Print move statistics."""
        print("[+] Move stats:")
        print("Shape", move.shape)
        print("Shift", move.shift)
        print("Rotation", move.rotate)

        for row in shape_as_matrix(move):
            print("".join(["1" if block else "0" for block in row]))


    def print_board(self, board, fill=True):
        """
        Print board for given move. When fill=True empty spaces are filled by zeros.
        """
        print("[+] Board dump:")
        for line in board:
            line = "{:016b}".format(line)[:10]
            if not fill:
                line = line.replace("0", " ")
            print(line)


    def validate(self):
        """Validate moves in game."""
        correct = 0
        for idx in range(len(self.game) - 1):
            if self._validate_move(self.game[idx], self.game[idx+1]):
                correct += 1

        return correct / (len(self.game)-1)


    def _validate_move(self, prev_move, current_move):
        """Check board and points after move."""
        prev_board = [[int(block) for block in "{:016b}".format(line)[:10]] for line in prev_move.board]
        current_board = [[int(block) for block in "{:016b}".format(line)[:10]] for line in current_move.board]
        shape = self._shape_as_matrix(current_move)

        board = self._move_blocks(prev_board, shape)
        board, points = self._reduce_board(board)

        # print("Prev board:")
        # for line in prev_board:
        #     print("".join(str(block) for block in line))
        # print("Board:")
        # for line in board:
        #     print("".join(str(block) for block in line))
        # print("Current board:")
        # for line in current_board:
        #     print("".join(str(block) for block in line))
        # print("Points", points)
        # print("Match", board == current_board)

        return points == current_move.points and board == current_board


    def _shape_as_matrix(self, move):
        """Get shape as matrix that fit in board."""
        ratation = move.rotate % len(GameAnalyzer.SHAPES[move.shape])
        shape = []
        for line in GameAnalyzer.SHAPES[move.shape][ratation]:
            if move.shift == 0:
                shape.append(line)
            elif move.shift < 0:
                shift = abs(move.shift)
                shape.append(line[shift:] + [0 for _ in range(shift)])
            else:
                shift = move.shift
                shape.append([0 for _ in range(shift)] + line[:-shift])

        return shape


    def _move_blocks(self, prev_board, shape):
        """Move and fit shape in previous board."""
        board = copy.deepcopy(prev_board)

        # Move block
        for y in range(BORAD_HEIGHT):
            # If collision then revoke actual board
            for row, line in enumerate(shape):
                for col, block in enumerate(line):
                    if prev_board[y+row][col] == 1 and block == 1:
                        return board

            # Fill boad with shape
            board = copy.deepcopy(prev_board)
            for row, line in enumerate(shape):
                for col, block in enumerate(line):
                    if block == 1:
                        board[y+row][col] = 1

            # If next move is out of border, then break
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

        # Fill missing lines in
        if len(board) != BORAD_HEIGHT:
            missing = BORAD_HEIGHT - len(cleared_board)
            for _ in range(missing):
                board = EMPTY_LINE + board

        return board, points


def main():
    # file_name = sys.argv[1]
    # file_name = "20190529201253.trace"

    # List all files with .trace extension
    file_names = []
    for r, _, f in os.walk('.'):
        for file_name in f:
            if file_name.endswith('.trace'):
                file_names.append(os.path.join(r, file_name))

    # Read game
    for file_name in file_names:
        with open(file_name, "r") as f:
            game = read_games(f)

            a = GameAnalyzer(game)
            print("%s: validation %.2f%%" % (file_name, a.validate() * 100))

            new_file = file_name.split(".")[0] + ".ctrace"
            # save_new_trace(game, new_file)


def read_games(trace_file):
    """Read trace data."""
    game = []
    m = None
    for line in trace_file:
        packet = line.split()

        if packet[0] == "[>]":
            if packet[1] == "NP_newPiece":
                if m:
                    game.append(m)
                m = SingleMove()
                m.shape = int(packet[2].split("=")[1])
            elif packet[1] == "NP_left":
                m.shift -= 1
            elif packet[1] == "NP_right":
                m.shift += 1
            elif packet[1] == "NP_rotate":
                m.rotate += 1
        elif packet[0] == "[<]" and packet[1] == "NP_points":
            m.points = int(packet[2].split("=")[1])
        elif packet[0] == "[<]" and packet[1] == "NP_boardDump":
            if len(game) > 0:
                board = packet[3].split("=")[1]
                lines = [board[i:i+4] for i in range(0, len(board), 4)]
                game[-1].board = list(reversed([int(line, 16) for line in lines]))
                game[-1].raw_board = board

    return game


def save_new_trace(game, file_name):
    """Save squeezed game to new trace file."""
    with open(file_name, "w") as f:
        for m in game:
            f.write("{shape} {shift} {rotate} {points} {raw_board}\n" \
                .format(shape=m.shape, shift=m.shift, rotate=m.rotate,
                    points=m.points, raw_board=str(m.raw_board)))


if __name__ == "__main__":
    main()
