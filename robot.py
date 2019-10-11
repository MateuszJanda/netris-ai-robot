#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""


def main():
    with open('robot.log', 'w') as f:
        print('Version 1')
        while True:
            cmd = input()
            f.write('[>]' + cmd + '\n')




if __name__ == '__main__':
    main()
