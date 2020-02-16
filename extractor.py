#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import os
import pickle
import trace_parser as t


def main():
    only_wins()


def only_wins():
    """Extract data only from action where points have been scored."""
    total = 0
    data = []

    for action in t.Reader("20190529201253.trace"):
        total += 1

        if action.points():
            piece = action.piece()
            shift = action.shift()
            rotate = action.rotate()
            board = action.board_as_list()

            entry = [piece] + [shift] + [rotate] + board
            data.append(entry)

        # Print progress
        if total % 50 == 0:
            print('Progress:', total)

    pickle.dump(data, open('only_wins.pickle', 'wb'))
    print('Checked %d/%d' % (len(data), total))


if __name__ == '__main__':
    main()
