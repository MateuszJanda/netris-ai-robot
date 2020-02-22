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
SHIFT_SIZE = 1
ROTATE_SIZE = 1


def main():
    only_wins()


def only_wins():
    """Extract data only from action where points have been scored."""
    total = 0
    data = []

    for action in t.Reader("data/"):
        total += 1

        if action.points():
            piece = action.piece()
            board = action.flat_board()
            shift = action.shift()
            rotate = action.rotate()

            data_input = piece + board
            data_output = shift + rotate

            assert(len(data_input) == PIECE_SIZE + BOARD_SIZE)
            assert(len(data_output) == SHIFT_SIZE + ROTATE_SIZE)

            data.append((data_input, data_output))

        # Print progress
        if total % 100 == 0:
            print("Extracted: %d" % (len(data)))

    print("Total: %d, extracted: %d" % (total, len(data)))
    print("Single input size: %d" % len(data[0][0]))
    print("Single output size: %d" % len(data[0][1]))

    pickle.dump(data, open("only_wins.pickle", "wb"))


if __name__ == "__main__":
    main()
