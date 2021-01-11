#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import os
import sys
import signal
import time
import asyncio
import datetime
import argparse
from proxy_robot import ProxyRobot
import config


LOG_FILE = None


def main():
    # Should prevent BrokenPipeError
    signal.getsignal(signal.SIGPIPE)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    args = parse_args()
    setup_logging(args)
    log("Start robot, PID: %d. Connection to agent at %s:%d" % (os.getpid(), config.HOST, args.port))

    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    # Start monitoring the fd file descriptor for read availability and invoke
    # callback with the specified arguments once fd is available for reading
    loop.add_reader(sys.stdin, got_robot_cmd, queue)

    future_stop = loop.create_future()

    coroutine = loop.create_connection(lambda: ProxyRobot(loop, future_stop, queue, LOG_FILE), config.HOST, args.port)
    loop.run_until_complete(coroutine)

    try:
        loop.run_until_complete(future_stop)
    except KeyboardInterrupt:
        pass

    cancel_all_task()

    log("Stop robot, PID:", os.getpid())
    if LOG_FILE:
        LOG_FILE.close()


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Netris proxy robot for reinforcement learning (DQN)\n'
                'Mateusz Janda (c) <mateusz janda at gmail com>\n'
                'netris-ai-robot project github.com/MateuszJanda/netris-ai-robot\n'
                '\n'
                'Robot will try to connect with DQN agent at ' + config.HOST + ':' + str(config.PORT) + '\n',
        usage='Please try to use -h, --help for more informations',
        epilog=' \n',
        formatter_class=CustomFormatter)

    parser.add_argument('-p', '--port', required=False, action='store', default=config.PORT, dest='port',
                        help='Connect to DQN server port')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-l', '--log-to-file', required=False, action='store_true', dest='log_file',
                       help='Log to file - robot_%%Y%%m%%d%%H%%M%%S.txt')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-t', '--log-in-terminal', required=False, metavar='<pts>', dest='log_terminal',
                       help='Log in terminal - e.g. /dev/pts/1')

    args = parser.parse_args()
    if args.log_file:
        ts = time.time()
        args.log_name = datetime.datetime.fromtimestamp(ts).strftime("robot_%Y%m%d%H%M%S.txt")
    elif args.log_terminal:
        args.log_name = args.log_terminal
    else:
        args.log_name = None

    args.port = int(args.port)

    return args


def setup_logging(args):
    """Setup logging."""
    global LOG_FILE

    if args.log_name:
        LOG_FILE = open(args.log_name, "w")
        sys.stderr = LOG_FILE


def got_robot_cmd(queue):
    """Setup task waiting for Netris/Robot commands."""
    loop = asyncio.get_event_loop()
    loop.create_task(queue.put(sys.stdin.readline()))


def cancel_all_task(result=None):
    log("Cancel all tasks")
    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks(loop):
        task.cancel()


def log(*args, **kwargs):
    """
    Print log to other terminal or file.
    """
    if LOG_FILE:
        print(*args, **kwargs, file=LOG_FILE)


if __name__ == "__main__":
    main()
