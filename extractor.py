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


def main():
    only_wins = lambda action: action.points()
    no_gaps = lambda action: action.points() or (action.gaps() == 0 and not action.cliff(3))

    # extract_data(only_wins, "only_wins.pickle")
    extract_data(no_gaps, "no_gaps.pickle")


def extract_data(action_filter, file_name):
    """Extract data only from action where filter allow to."""
    total = 0
    data = []

    for action in t.Reader("data/"):
        total += 1

        if not action_filter(action):
            continue

        piece = action.piece()
        board = action.flat_board()
        shift = action.shift()
        rotate = action.rotate()

        data_input = piece + board

        assert(len(data_input) == PIECE_SIZE + BOARD_SIZE)

        data.append((data_input, shift, rotate))

        # Print progress
        if total % 100 == 0:
            print("Extracted: %d" % (len(data)))

    print("Total: %d, extracted: %d (%0.2f%%)" % (total, len(data), len(data)/total))
    print("Single input size: %d" % len(data[0][0]))

    pickle.dump(data, open(file_name, "wb"))


if __name__ == "__main__":
    main()
