#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

from robot import config


def board_look(board, height, width):
    """Print board state. For debug only."""
    look = ""
    board = board.reshape(height, width)
    for line in board:
        blocks_line = "".join(["[]" if b else "  " for b in line])
        look += "|" + blocks_line + "|\n"

    return look


def columns_height(board):
    """
    Calculate height for each column.
    """
    columns_height = [0 for _ in range(config.BOARD_WIDTH)]
    for x in range(config.BOARD_WIDTH):
        for y, block in enumerate(board[:, x]):
            if block:
                columns_height[x] = config.BOARD_HEIGHT - y
                break

    return columns_height


def aggregate_height(board):
    """
    Calculate aggregate height of all columns.
    """
    return sum(columns_height(board))


def holes(board):
    """
    Calculate all holes in board.
    """
    columns_roof = [False for _ in range(config.BOARD_WIDTH)]
    holes_counter = 0

    for row in board:
        for x, block in enumerate(row):
            if block:
                columns_roof[x] = True
            elif columns_roof[x] and block == 0:
                holes_counter += 1

    return holes_counter


def bumpiness(board):
    """
    Calculate bumpiness - sum of the absolute differences in height between
    adjacent columns.

    """
    col_height = columns_height(board)
    counter = 0
    for x in range(len(col_height)):
        if x + 1 < config.BOARD_WIDTH:
            counter += abs(col_height[x] - col_height[x + 1])

    return counter
