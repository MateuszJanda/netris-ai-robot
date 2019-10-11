#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""
import time
import datetime


def main():
    ts = time.time()
    log_name = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S.bot')

    with open(log_name, 'w') as f:
        print('Version 1')
        while True:
            cmd = input()
            f.write('[>] ' + cmd + '\n')
            f.flush()

            if cmd.startswith('NewPiece'):
                print('Message Lorem ipsum\n')

            if cmd.startswith('Exit'):
                f.write('[!] Game end\n')
                f.flush()
                break

    return 0


if __name__ == '__main__':
    main()
