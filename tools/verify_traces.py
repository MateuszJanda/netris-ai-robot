#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import os
import trace_parser as t


def main():
    """
    Verify how many actions can be reproduced. for some actions there is not
    enough data to reproduce board state like exact time when piece should be
    shifted.
    """
    total_actions = 0

    for file_name in trace_files(path='data/'):
        game = t.Game(file_name)
        total_actions += len(game.tours) - 1
        print("%s: reconstruction %.2f%%, actions %d" % (file_name, game.recreate() * 100, len(game.tours) - 1))

    print("Total actions %d" % total_actions)


def trace_files(path):
    """List all files with .trace extension."""
    file_names = []
    for r, _, f in os.walk(path):
        for file_name in f:
            if file_name.endswith('.trace'):
                file_names.append(os.path.join(r, file_name))

    return sorted(file_names)


if __name__ == '__main__':
    main()
