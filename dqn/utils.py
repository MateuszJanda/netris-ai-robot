#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""


def print_board(board, height, width):
    """Print board state. For debug only."""
    log("Board")
    board = board.reshape(height, width)
    for line in board:
        blocks = "".join(["1" if b else " " for b in line])
        log("|" + blocks + "|")
