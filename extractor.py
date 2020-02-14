#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import os
import trace_parser as t

def main():
    for action in t.Reader("20190529201253.trace"):
        print(action.current_max())


if __name__ == '__main__':
    main()
