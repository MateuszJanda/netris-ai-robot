#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import config


def board_look(board, height, width):
    """Print board state. For debug only."""
    look = ""
    board = board.reshape(height, width)
    for line in board:
        blocks_line = "".join(["[]" if b else "  " for b in line])
        look += "|" + blocks_line + "|\n"

    return look


def agregate_height(board):
    columns_height = [0 for _ in range(config.BOARD_WIDTH)]
    for x in range(config.BOARD_WIDTH):
        for y, block in enumerate(range(board[:, x])):
            if block:
                columns_height[x] = config.BOARD_HEIGHT - y + 1

    return sum(columns_height)


def holes(borad):
    columns_roof = [False for _ in range(config.BOARD_WIDTH)]
    holes_counter = 0

    for row in board:
        for x, block in enumerate(row):
            if block:
                columns_roof[x] = True
            elif columns_roof[x] and block == 0:
                holes_counter += 1

    return holes_counter


def  bumpiness(board):
    columns_height = [0 for _ in range(config.BOARD_WIDTH)]
    for x in range(config.BOARD_WIDTH):
        for y, block in enumerate(range(board[:, x])):
            if block:
                columns_height[x] = config.BOARD_HEIGHT - y + 1

    counter = 0
    for x in range(len(columns_height)):
        if x + 1 < config.BOARD_WIDTH:
            counter += abs(columns_height[x] - columns_height[x + 1])

    return counter
