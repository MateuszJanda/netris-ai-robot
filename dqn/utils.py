#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""


def board_look(board, height, width):
    """Print board state. For debug only."""
    look = "Board"
    board = board.reshape(height, width)
    for line in board:
        blocks_line = "".join(["[]" if b else "  " for b in line])
        look += "|" + blocks_line + "|\n"

    return look
