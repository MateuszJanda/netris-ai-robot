#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import os
import sys
import copy
from typing import List, Optional, Tuple, TextIO


BOARD_WIDTH = 10
BORAD_HEIGHT = 20
BLOCK = 1
SHFIT_OFFSET = 5


# Piece index and it representation. Counterclockwise rotation.
PIECE = {
    0  : [
        [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    ],
    2  : [
        [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]
    ],
    3  : [
        [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    ],
    7  : [
        [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]],
    ],
    11 : [
        [[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    ],
    15 : [
        [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    ],
    17 : [
        [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    ]
}


Board = List[List[int]]
RawBoard = List[int]


class Tour:
    def __init__(self) -> None:
        self.piece = 0
        self.shift = 0
        self.rotate = 0
        self.points = 0
        self.dump: str = ""

        self._raw_board: RawBoard = []
        self.board: Board = []

    @property
    def raw_board(self) -> RawBoard:
        """Read raw board."""
        return self._raw_board

    @raw_board.setter
    def raw_board(self, value: RawBoard):
        """Set raw_board and create board array."""
        self._raw_board = value
        self.board = [[int(piece) for piece in "{:016b}".format(line)[:BOARD_WIDTH]] for line in self._raw_board]

    def max(self) -> int:
        """Get max block height on board."""
        max_height = column_height(0, self.board)
        for col in range(1, BOARD_WIDTH):
            max_height = max(max_height, column_height(col, self.board))

        return max_height

    def min(self) -> int:
        """Get min block height on board."""
        min_height = column_height(0, self.board)
        for col in range(1, BOARD_WIDTH):
            min_height = min(min_height, column_height(col, self.board))

        return min_height

    def piece_as_matrix(self) -> Board:
        """Get piece as matrix in right position on board before drop."""
        matrix = []
        for line in PIECE[self.piece][self.rotate]:
            if self.shift == 0:
                matrix.append(line)
            elif self.shift < 0:
                shift = abs(self.shift)
                matrix.append(line[shift:] + [0 for _ in range(shift)])
            else:
                shift = self.shift
                matrix.append([0 for _ in range(shift)] + line[:-shift])

        return matrix

    def piece_columns(self) -> Tuple[int, int]:
        """Return column range [start, end) where piece will be dropped."""
        matrix = self.piece_as_matrix()
        left = BOARD_WIDTH
        right = 0

        for line in matrix:
            l = next(idx for idx, val in enumerate(line) if val != 0)
            left = min(l, left)

            r = next(idx for idx, val in enumerate(reversed(line)) if val != 0)
            right = max(len(line) - r, right)

        return left, right

    def print_stats(self) -> None:
        """Print tour statistics."""
        print("Shape:", self.piece)
        print("Shift:", self.shift)
        print("Rotation:", self.rotate)

        for line in self.piece_as_matrix():
            print("".join(["1" if piece else "0" for piece in line]))

    def print_board(self, fill: bool=True) -> None:
        """
        Print board for given tour. When fill=True empty spaces are filled
        by zeros.
        """
        for num in self._raw_board:
            line = "{:016b}".format(num)[:BOARD_WIDTH]
            if not fill:
                line = line.replace("0", " ")
            print(line)


class Action:
    def __init__(self, tour: Tour, next_tour: Tour) -> None:
        self.tour = tour
        self.next_tour = next_tour

    def gaps(self) -> int:
        """
        Count all gaps (blocks that can't be reached in next tour) created
        by piece, under horizontal projection of piece.
        """
        piece = self.tour.piece_as_matrix()
        board = self._merge_piece_with_board(piece)

        counter = 0
        for col in range(BOARD_WIDTH):
            row1 = BORAD_HEIGHT - column_height(col, board)
            row2 = BORAD_HEIGHT - column_height(col, self.tour.board)

            for row in range(row1 + 1, row2):
                if board[row][col] != BLOCK:
                    counter += 1

        return counter

    def cliff(self, height: int) -> int:
        """Check if piece create cliff of given height or higher."""
        piece = self.tour.piece_as_matrix()
        board = self._merge_piece_with_board(piece)
        start, end = self.tour.piece_columns()

        left = self._cliff_height(start - 1, start, board)
        right = self._cliff_height(end, end - 1, board)

        return left >= height or right >= height

    def _cliff_height(self, cliff_col: int, piece_col: int, board: Board) -> int:
        """Calculate cliff height for given column."""
        if cliff_col < 0 or cliff_col >= BOARD_WIDTH:
            return 0

        h1 = column_height(cliff_col, board)
        h2 = column_height(piece_col, board)

        if h2 > h1:
            return h2 - h1

        return 0

    def valid(self) -> bool:
        """Check if board can be reconstructed properly by current tour."""
        piece = self.tour.piece_as_matrix()

        board = self._merge_piece_with_board(piece)
        board, points = self._remove_full_lines(board)

        return points == self.tour.points and board == self.next_tour.board

    def _merge_piece_with_board(self, piece: Board) -> Board:
        """Move and place piece in current tour board."""
        board = copy.deepcopy(self.tour.board)

        # Move piece
        for y in range(BORAD_HEIGHT):
            # If collision then revoke actual board
            for row, line in enumerate(piece):
                for col, block in enumerate(line):
                    if self.tour.board[y+row][col] and block:
                        return board

            # Fill board with piece blocks
            board = copy.deepcopy(self.tour.board)
            for row, line in enumerate(piece):
                for col, block in enumerate(line):
                    if block:
                        board[y+row][col] = BLOCK

            # If next move is out of border then break
            if (y+1) + len(piece) > BORAD_HEIGHT:
                break

        return board

    def _remove_full_lines(self, board: Board) -> Tuple[Board, int]:
        """Remove full lines and count points."""
        FULL_LINE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        EMPTY_LINE = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        # Check for full lines and count point
        cleared_board = []
        points = 0
        for line in board:
            if line == FULL_LINE:
                points += 1
            else:
                cleared_board.append(line)

        board = copy.deepcopy(cleared_board)

        # Replace missing by empty lines in board
        if len(board) != BORAD_HEIGHT:
            missing = BORAD_HEIGHT - len(cleared_board)
            for _ in range(missing):
                board = EMPTY_LINE + board

        return board, points

    def current_max(self) -> int:
        """Get max block height on current board."""
        return self.tour.max()

    def next_max(self) -> int:
        """Get max block height on next board (after tour/move)."""
        return self.next_tour.max()

    def current_min(self) -> int:
        """Get min block height on current board."""
        return self.tour.min()

    def next_min(self) -> int:
        """Get min block height on next board (after tour/move)."""
        return self.next_tour.min()

    def points(self) -> int:
        """Get points (erased full lines)."""
        return self.tour.points

    def piece(self) -> List[int]:
        """Return normalized piece id."""
        ids = {
            0: 0,
            2: 1,
            3: 2,
            7: 3,
            11: 4,
            15: 5,
            17: 6
        }
        return [ids[self.tour.piece]]

    def shift(self) -> int:
        """Return piece shift as list."""
        return self.tour.shift + SHFIT_OFFSET

    def rotate(self) -> int:
        """Return piece rotation as list."""
        return self.tour.rotate

    def flat_board(self) -> List[int]:
        """Return board as flat list."""
        result = []
        for line in self.tour.board:
            result.extend(line)

        return result


class Game:
    def __init__(self, file_name: str) -> None:
        """Read game from single trace file."""
        self.tours: List[Tour] = []

        with open(file_name, "r") as f:
            self.tours = self._read(f)

    def __iter__(self):
        """Return iterator."""
        self._idx = 0
        return self

    def __next__(self) -> Action:
        """Return valid Action."""
        while self._idx < len(self.tours) - 1:
            action = Action(self.tours[self._idx], self.tours[self._idx+1])
            self._idx += 1
            if action.valid():
                return action

        raise StopIteration

    def _read(self, trace: TextIO) -> List[Tour]:
        """Reading trace data with squeezed shift and rotation."""
        BYTES_PER_LINE = 4
        game: List[Tour] = []
        tour: Optional[Tour] = None

        for line in trace:
            packet = line.split()

            if packet[0] == "[>]":
                if packet[1] == "NP_newPiece":
                    if tour:
                        game.append(tour)
                    tour = Tour()
                    tour.piece = int(packet[2].split("=")[1])
                elif packet[1] == "NP_left":
                    tour.shift -= 1
                elif packet[1] == "NP_right":
                    tour.shift += 1
                elif packet[1] == "NP_rotate":
                    tour.rotate += 1
                    tour.rotate %= len(PIECE[tour.piece])
            elif packet[0] == "[<]" and packet[1] == "NP_points":
                tour.points = int(packet[2].split("=")[1])
            elif packet[0] == "[<]" and packet[1] == "NP_boardDump":
                tour.dump = packet[3].split("=")[1]
                lines = [tour.dump[i:i+BYTES_PER_LINE] for i in range(0, len(tour.dump), BYTES_PER_LINE)]
                tour.raw_board = list(reversed([int(line, 16) for line in lines]))

        return game

    def print_stats(self) -> None:
        """Print game statistics."""
        print("Points:", sum([a.points for a in self.tours]))
        print("Tours:", len(self.tours))
        print("Overall pieces:", set([a.piece for a in self.tours]))


    def recreate(self) -> float:
        """Return percentage of tours reconstructed in game."""
        correct = 0
        for idx in range(len(self.tours) - 1):
            action = Action(self.tours[idx], self.tours[idx+1])
            if action.valid():
                correct += 1

        return correct / (len(self.tours)-1)


def column_height(col: int, board: Board) -> int:
    """Return height of given column in board."""
    for row in range(BORAD_HEIGHT):
        if board[row][col]:
            return BORAD_HEIGHT - row

    return 0


class Reader:
    def __init__(self, path: str) -> None:
        """Read single trace of all trace in path."""
        if os.path.isfile(path):
            self._file_names = [path]
        else:
            self._file_names = self._all_files_in_path(path)

    def __iter__(self):
        """Get first game from list and return iterator."""
        self._idx = 0
        self._game = iter(Game(self._file_names[self._idx]))
        return self

    def __next__(self) -> Action:
        """Return Action from all games."""
        try:
            return next(self._game)
        except StopIteration as e:
            if self._idx + 1 < len(self._file_names):
                self._idx += 1
                self._game = iter(Game(self._file_names[self._idx]))
                return next(self._game)

        raise StopIteration

    def _all_files_in_path(self, dir_path: str) -> List[str]:
        """List all files with .trace extension."""
        _file_names = []
        for r, _, f in os.walk(dir_path):
            for file_name in f:
                if file_name.endswith(".trace"):
                    _file_names.append(os.path.join(r, file_name))

        return sorted(_file_names)
