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


class Action:
    def __init__(self):
        self.piece = 0
        self.shift = 0
        self.rotate = 0
        self.points = 0
        self.board = ""
        self.raw_board = ""


class Game:
    # Shape number and it representation. Counterclockwise rotation.
    PIECES = {
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
        print("Points:", sum([a.points for a in self.game]))
        print("game:", len(self.game))
        print("Overall pieces:", set([a.piece for a in self.game]))


    def print_action_stats(self, action):
        """Print action statistics."""
        print("[+] Move stats:")
        print("Shape", action.piece)
        print("Shift", action.shift)
        print("Rotation", action.rotate)

        for row in piece_as_matrix(action):
            print("".join(["1" if piece else "0" for piece in row]))


    def print_board(self, board, fill=True):
        """
        Print board for given action. When fill=True empty spaces are filled
        by zeros.
        """
        print("[+] Board dump:")
        for line in board:
            line = "{:016b}".format(line)[:10]
            if not fill:
                line = line.replace("0", " ")
            print(line)


    def reconstruct(self):
        """Reconstruct actions in game."""
        correct = 0
        for idx in range(len(self.game) - 1):
            if self._reconstruct_action(self.game[idx], self.game[idx+1]):
                correct += 1

        return correct / (len(self.game)-1)


    def _reconstruct_action(self, prev_action, current_action):
        """Check board and points after action."""
        prev_board = [[int(piece) for piece in "{:016b}".format(line)[:10]] for line in prev_action.board]
        current_board = [[int(piece) for piece in "{:016b}".format(line)[:10]] for line in current_action.board]
        piece = self._piece_as_matrix(current_action)

        board = self._action_pieces(prev_board, piece)
        board, points = self._reduce_board(board)

        # print("Prev board:")
        # for line in prev_board:
        #     print("".join(str(piece) for piece in line))
        # print("Board:")
        # for line in board:
        #     print("".join(str(piece) for piece in line))
        # print("Current board:")
        # for line in current_board:
        #     print("".join(str(piece) for piece in line))
        # print("Points", points)
        # print("Match", board == current_board)

        return points == current_action.points and board == current_board


    def _piece_as_matrix(self, action):
        """Get piece as matrix that fit in board."""
        piece = []
        for line in Game.PIECES[action.piece][action.rotate]:
            if action.shift == 0:
                piece.append(line)
            elif action.shift < 0:
                shift = abs(action.shift)
                piece.append(line[shift:] + [0 for _ in range(shift)])
            else:
                shift = action.shift
                piece.append([0 for _ in range(shift)] + line[:-shift])

        return piece


    def _action_pieces(self, prev_board, piece):
        """Move and fit piece in previous board."""
        board = copy.deepcopy(prev_board)

        # Move piece
        for y in range(BORAD_HEIGHT):
            # If collision then revoke actual board
            for row, line in enumerate(piece):
                for col, block in enumerate(line):
                    if prev_board[y+row][col] == 1 and block == 1:
                        return board

            # Fill boad with piece
            board = copy.deepcopy(prev_board)
            for row, line in enumerate(piece):
                for col, block in enumerate(line):
                    if block == 1:
                        board[y+row][col] = 1

            # If next action is out of border, then break
            if (y+1) + len(piece) > BORAD_HEIGHT:
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

            a = Game(game)
            print("%s: reconstruction %.2f%%" % (file_name, a.reconstruct() * 100))

            new_file = file_name.split(".")[0] + ".ctrace"
            # save_new_trace(game, new_file)


def read_games(trace_file):
    """Reading trace data with squeezed shift."""
    game = []
    action = None

    for line in trace_file:
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
                action.rotate %= len(Game.PIECES[action.piece])
        elif packet[0] == "[<]" and packet[1] == "NP_points":
            action.points = int(packet[2].split("=")[1])
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
        for a in game:
            f.write("{piece} {shift} {rotate} {points} {raw_board}\n" \
                .format(piece=a.piece, shift=a.shift, rotate=a.rotate,
                    points=a.points, raw_board=str(a.raw_board)))


if __name__ == "__main__":
    main()
