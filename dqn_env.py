#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

BOARD_WIDTH = 10
ROTATIONS = 4
ACTION_SPACE_SIZE = ROTATIONS * BOARD_WIDTH

SHFIT_OFFSET = 5


class Env:
    def reset(self):
        pass

    def step(self, action):
        if action >= ACTION_SPACE_SIZE:
            raise Exception("Action not in action space")

        shift = action % BOARD_WIDTH - SHFIT_OFFSET
        rotate = action // BOARD_WIDTH

        return None

def main():
    pass

if __name__ == '__main__':
    main()
