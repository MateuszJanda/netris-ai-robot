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

    for action in t.Reader("data/"):
        total += 1

        if action.points():
            piece = action.normalized_piece()
            shift = action.normalized_shift()
            rotate = action.normalized_rotate()
            board = action.normalized_board()

            data_in = piece + board
            data_out = shift + rotate

            data.append((data_in, data_out))

        # Print progress
        if total % 100 == 0:
            print('Progress: %d, extracted: %d' % (total, len(data)))

    pickle.dump(data, open('only_wins.pickle', 'wb'))
    print('Total: %d, extracted: %d' % (total, len(data)))


if __name__ == '__main__':
    main()
