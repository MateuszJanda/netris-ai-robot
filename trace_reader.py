#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import sys


class MoveData:
    def __init__(self):
        self.shape = 0
        self.move = 0
        self.rotate = 0
        self.points = 0
        self.board = ''
        self.raw_board = ''


def main():
    file_name = sys.argv[1]

    print('Trace file:', file_name)

    # Squeez moves
    with open(file_name, 'r') as f:
        moves = convert_trace(f)

    print_board(moves[0])
    print_move_stat(moves[0])
    print_game_stats(moves)

    new_file = file_name.split('.')[0] + '.ctrace'
    save_new_trace(moves, new_file)


def convert_trace(trace_file):
    """Squeez moves."""
    moves = []
    m = None
    for line in trace_file:
        packet = line.split()

        if packet[0] == '[>]':
            if packet[1] == 'NP_newPiece':
                if m:
                    moves.append(m)
                m = MoveData()
                m.shape = int(packet[2].split('=')[1])
            elif packet[1] == 'NP_left':
                m.move -= 1
            elif packet[1] == 'NP_right':
                m.move += 1
            elif packet[1] == 'NP_rotate':
                m.rotate += 1
        elif packet[0] == '[<]' and packet[1] == 'NP_points':
            m.points = int(packet[2].split('=')[1])
        elif packet[0] == '[<]' and packet[1] == 'NP_boardDump':
            if len(moves) > 0:
                board = packet[3].split('=')[1]
                lines = [board[i:i+4] for i in range(0, len(board), 4)]
                moves[-1].board = reversed([int(line, 16) for line in lines])
                moves[-1].raw_board = board

    return moves


def print_board(move, fill=False):
    """Print board for given move. when fill=True empty spaces are filled
    by zeros."""
    for line in move.board:
        line = '{:010b}'.format(line)
        if fill:
            line = line.replace('0', ' ')
        print(line)


def print_move_stat(move):
    print("Shape", move.shape)


def shape_as_matrix(shape):
    if shape == 0:
        return [ [0, 1, 1],
                 [1, 1, 0] ]


def print_game_stats(moves):
    """Print trace stats."""
    print('Points:', sum([m.points for m in moves]))
    print('Moves:', len(moves))
    print('Blocks:', set([m.shape for m in moves]))


def save_new_trace(moves, file_name):
    """Save squeezed moves to new trace file."""
    with open(file_name, 'w') as f:
        for m in moves:
            f.write('{shape} {move} {rotate} {points} {raw_board}\n' \
                .format(shape=m.shape, move=m.move, rotate=m.rotate,
                    points=m.points, raw_board=str(m.raw_board)))


if __name__ == '__main__':
    main()
