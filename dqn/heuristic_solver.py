#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

class HeuristicSolver:

    PIECE = {
        1: [[1, 1, 1],
            [0, 1, 0]],
        2: [[1, 1, 1, 1]],
        3: [[1, 1],
            [1, 1]],
        4: [[1, 1, 1],
            [1, 0, 0]],
        5: [[1, 1, 1],
            [0, 0, 1]],
        6: [[0, 1, 1],
            [1, 1, 0]],
        7: [[1, 1, 0],
            [0, 1, 1]],
    }

    def action(self, new_piece, board):
        pass
