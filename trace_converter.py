#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""


class MoveData:
    def __init__(self):
        self.shape = 0
        self.move = 0
        self.rotate = 0
        self.points = 0
        self.board = ''


def main():
    with open('20190529201253.trace', 'r') as f:
        moves = convert_trace(f)

    draw_board(moves[0])
    stats(moves)


def convert_trace(trace_file):
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

    return moves


def draw_board(move):
    for line in move.board:
        print('{:010b}'.format(line))


def stats(moves):
    print('Points:', sum([m.points for m in moves]))
    print('Moves:', len(moves))
    print('Blocks:', set([m.shape for m in moves]))



if __name__ == '__main__':
    main()
