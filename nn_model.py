#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import pickle
import numpy as np
import tensorflow as tf


def main():
    x_train, y_train = load_data()


def load_data():
    with open("only_wins.pickle", "rb") as f:
        data = pickle.load(f)

    data_input = []
    data_output = []
    for i, o in data:
        data_input.append(np.array(i))
        data_output.append(np.array(o))

    return np.array(data_input), np.array(data_output)



if __name__ == '__main__':
    main()
