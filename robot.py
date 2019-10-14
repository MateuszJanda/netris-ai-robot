#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""
import time
import datetime


BOARD_WIDTH = 10
BORAD_HEIGHT = 20


def main():
    ts = time.time()
    log_name = datetime.datetime.fromtimestamp(ts).strftime('robot_%Y%m%d%H%M%S.txt')

    piece_id = None
    with open(log_name, 'w') as f:
        print('Version 1')
        while True:
            cmd = input()
            log(f, '[>] ' + cmd)

            if cmd.startswith('NewPiece'):
                piece_id = cmd.split(' ')[1]
                print('Message Lorem ipsum')

            elif cmd.startswith('BoardSize'):
                params = cmd.split(' ')
                height = int(params[1])
                width = int(params[2])

                if width != BOARD_WIDTH and height != BORAD_HEIGHT:
                    log(f, '[!] Validation board size fail')
                    break

            elif cmd.startswith('TimeStamp') and piece_id:
                log(f, '[<] Right ' + piece_id)

                print('Right ' + piece_id)
                print('Right ' + piece_id)
                print('Right ' + piece_id)
                print('Right ' + piece_id)
                print('Right ' + piece_id)
                piece_id = None

            elif cmd.startswith('Exit'):
                log(f, '[!] Game end')
                break

    return 0


def log(file, msg):
    file.write(msg + '\n')
    file.flush()


if __name__ == '__main__':
    main()
