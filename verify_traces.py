#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import os
import trace as t


def main():
    for file_name in trace_files(path='.'):
        game = t.Game(file_name)
        print("%s: reconstruction %.2f%%" % (file_name, game.reconstruct() * 100))


def trace_files(path):
    """List all files with .trace extension."""
    file_names = []
    for r, _, f in os.walk(path):
        for file_name in f:
            if file_name.endswith('.trace'):
                file_names.append(os.path.join(r, file_name))

    return file_names


if __name__ == '__main__':
    main()
