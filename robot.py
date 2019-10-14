#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""
import traceback
import time
import datetime


BOARD_WIDTH = 10
BORAD_HEIGHT = 20
SCR_ID = 0

EMPTY_LINE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
EMPTY_BLOCK = 0
FULL_BLOCK = 1


def main():
    ts = time.time()
    # log_name = datetime.datetime.fromtimestamp(ts).strftime('robot_%Y%m%d%H%M%S.txt')
    log_name = '/dev/pts/3'

    board = [EMPTY_LINE for _ in range(BORAD_HEIGHT)]
    piece_id = None
    ver = False

    try:
        with open(log_name, 'w') as f:
            log(f, '[<] Version 1')
            print('Version 1')

            while True:
                cmd = input()
                log(f, '[>] ' + cmd)

                if cmd.startswith('NewPiece'):
                    piece_id = cmd.split(' ')[1]
                    log(f, '[<] Message Lorem ipsum')
                    print('Message Lorem ipsum')

                elif cmd.startswith('BoardSize'):
                    scr_id, height, width = [int(p) for p in cmd.split(' ')[1:]]

                    if width != BOARD_WIDTH and height != BORAD_HEIGHT:
                        log(f, '[<] Exit')
                        log(f, '[!] Validation board size fail %d %d %d %d' % (width, BOARD_WIDTH, height, BORAD_HEIGHT))
                        print('Exit')
                        break

                elif cmd.startswith('RowUpdate'):
                    params = [int(p) for p in cmd.split(' ')[1:]]

                    if params[0] != SCR_ID:
                        continue

                    y = params[1]
                    for x, val in enumerate(params[2:]):
                        board[BORAD_HEIGHT - 1 - y][x] = FULL_BLOCK if val != 0 else EMPTY_BLOCK

                elif cmd.startswith('TimeStamp') and piece_id:
                    log(f, '[<] Right ' + piece_id)
                    print('Right ' + piece_id)

                    log(f, '[<] Right ' + piece_id)
                    print('Right ' + piece_id)

                    log(f, '[<] Right ' + piece_id)
                    print('Right ' + piece_id)
                    piece_id = None

                elif cmd.startswith('Exit'):
                    break
    except:
        with open(log_name, 'w') as f:
            traceback.print_exc(file=f)

    return 0


def log(file, msg):
    file.write(msg + '\n')
    file.flush()


if __name__ == '__main__':
    main()
