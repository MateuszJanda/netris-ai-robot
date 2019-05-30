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


def main():
    moves = []
    m = None

    with open('20190529201253.trace', 'r') as f:
        for line in f:
            packet = line.split():
            if packet[0] != '[>]':
                continue

            if packet[1] == 'NP_newPiece'
                if m:
                    moves.append(m)
                m = MoveData()
                m.shape = int(packet[2].split('=')[1])

            if packet[1] == 'NP_left':
                m.move -= 1
            elif packet[1] == 'NP_right':
                m.move += 1
            elif packet[1] == 'NP_rotate':
                m.rotate += 1


if __name__ == '__main__':
    main()
