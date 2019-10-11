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
    log_name = datetime.datetime.fromtimestamp(ts).strftime('robot_%Y%m%d%H%M%S.txt')

    piece_count = None
    with open(log_name, 'w') as f:
        print('Version 1')
        while True:
            cmd = input()
            f.write('[>] ' + cmd + '\n')
            f.flush()

            if cmd.startswith('NewPiece'):
                piece_count = cmd.split(' ')[1]
                print('Message Lorem ipsum')
            elif cmd.startswith('TimeStamp') and piece_count:
                f.write('[<] Right ' + piece_count + '\n')
                f.flush()

                print('Right ' + piece_count)
                print('Right ' + piece_count)
                print('Right ' + piece_count)
                print('Right ' + piece_count)
                print('Right ' + piece_count)
                # print('Message right')
                piece_count = None
            elif cmd.startswith('Exit'):
                f.write('[!] Game end\n')
                f.flush()
                break

    return 0


if __name__ == '__main__':
    main()
