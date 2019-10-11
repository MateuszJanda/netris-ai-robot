#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import sys
import copy


BOARD_WIDTH = 10
BORAD_HEIGHT = 20


class MoveData:
    def __init__(self):
        self.shape = 0
        self.shift = 0
        self.rotate = 0
        self.points = 0
        self.board = ""
        self.raw_board = ""


def main():
    # file_name = sys.argv[1]
    file_name = "20190529201253.trace"

    print("Trace file:", file_name)

    # Squeez moves
    with open(file_name, "r") as f:
        moves = convert_trace(f)

    # print_board(moves[0].board)
    # print_move_stat(moves[0])
    # print_game_stats(moves)
    # check_move(moves[0], moves[1])

    for idx in range(len(moves) - 1):
        r = check_move(moves[idx], moves[idx+1])
        print(r)

    new_file = file_name.split(".")[0] + ".ctrace"
    # save_new_trace(moves, new_file)


def convert_trace(trace_file):
    """Squeez moves."""
    moves = []
    m = None
    for line in trace_file:
        packet = line.split()

        if packet[0] == "[>]":
            if packet[1] == "NP_newPiece":
                if m:
                    moves.append(m)
                m = MoveData()
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
            if len(moves) > 0:
                board = packet[3].split("=")[1]
                lines = [board[i:i+4] for i in range(0, len(board), 4)]
                moves[-1].board = list(reversed([int(line, 16) for line in lines]))
                moves[-1].raw_board = board

    return moves


def print_board(board, fill=True):
    """Print board for given move. when fill=True empty spaces are filled
    by zeros."""
    print("[+] Board dump:")
    for line in board:
        line = "{:016b}".format(line)[:10]
        if not fill:
            line = line.replace("0", " ")
        print(line)


def print_move_stat(move):
    print("[+] Move stats:")
    print("Rotation", move.rotate)
    print("Shape", move.shape)

    for row in shape_as_matrix(shift):
        print("".join(["1" if block else "0" for block in row]))


def shape_as_matrix(move):
    # Rotation counterclockwise
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

            [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
        ],
        7  : [
            [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],

            [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],

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

    ratation = move.rotate % len(SHAPES[move.shape])
    shape = []
    for line in SHAPES[move.shape][ratation]:
        if move.shift < 0:
            shift = abs(move.shift)
            shape.append(line[shift:] + [0 for _ in range(shift)])
        else:
            shift = move.shift
            shape.append([0 for _ in range(shift)] + line[:-shift])

    return shape


def check_move(prev_move, current_move):
    prev_board = [[int(block) for block in "{:016b}".format(line)[:10]] for line in prev_move.board]
    current_board = [[int(block) for block in "{:016b}".format(line)[:10]] for line in current_move.board]
    shape = shape_as_matrix(current_move)

    # Move block
    board = move_blocks(prev_board, shape)
    board, points = reduce_board(board)

    # print("Board:")
    # for line in board:
    #     print("".join(str(block) for block in line))
    # print("Current board:")
    # for line in current_board:
    #     print("".join(str(block) for block in line))
    # print("Points", points)
    # print("Match", board == current_board)

    return points == current_move.points and board == current_board


def move_blocks(prev_board, shape):
    # Move block
    for y in range(BORAD_HEIGHT):
        # If collision then revoke actual board
        for row, line in enumerate(shape):
            for col, block in enumerate(line):
                if prev_board[y+row][col] == 1 and block == 1:
                    break

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

def reduce_board(board):
    # Check for full lines
    cleared_board = []
    points = 0
    for line in board:
        if line == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]:
            points += 1
        else:
            cleared_board.append(line)

    board = copy.deepcopy(cleared_board)

    # Fill missing lines in
    if len(board) != BORAD_HEIGHT:
        missing = BORAD_HEIGHT - len(cleared_board)
        for _ in range(missing):
            board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + board

    return board, points

def print_game_stats(moves):
    """Print trace stats."""
    print("[+] Game stats:")
    print("Points:", sum([m.points for m in moves]))
    print("Moves:", len(moves))
    print("Overall blocks:", set([m.shape for m in moves]))


def save_new_trace(moves, file_name):
    """Save squeezed moves to new trace file."""
    with open(file_name, "w") as f:
        for m in moves:
            f.write("{shape} {shift} {rotate} {points} {raw_board}\n" \
                .format(shape=m.shape, shift=m.shift, rotate=m.rotate,
                    points=m.points, raw_board=str(m.raw_board)))


if __name__ == "__main__":
    main()
