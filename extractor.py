#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import os
import pickle
import trace_parser as t


BOARD_SIZE = 20*10
PIECE_SIZE = 1
SHIFT_SIZE = 10
ROTATE_SIZE = 4


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
            board = action.normalized_board()
            shift = action.normalized_shift()
            rotate = action.normalized_rotate()

            data_input = piece + board
            data_output = shift + rotate

            assert(len(data_input) == PIECE_SIZE + BOARD_SIZE)
            assert(len(data_output) == SHIFT_SIZE + ROTATE_SIZE)

            data.append((data_input, data_output))

        # Print progress
        if total % 100 == 0:
            print("Progress: %d, extracted: %d" % (total, len(data)))

    print("Total: %d, extracted: %d" % (total, len(data)))
    print("Single input size: %d", len(data[0][0]))
    print("Single output size: %d", len(data[0][1]))

    pickle.dump(data, open("only_wins.pickle", "wb"))


if __name__ == "__main__":
    main()
