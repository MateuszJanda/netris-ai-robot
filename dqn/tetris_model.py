#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import numpy as np
import config


class TetrisModel:
    COLOR_TO_PIECE = {
        -1: 4,      # Piece Id: 11
        -2: 0,      # Piece Id: 0
        -3: 1,      # Piece Id: 2
        -4: 2,      # Piece Id: 3
        -5: 3,      # Piece Id: 7
        -6: 5,      # Piece Id: 15
        -7: 6,      # Piece Id: 17
    }

    PIECE_TO_PIECE_ID = {
        4: 11,
        0: 0,
        1: 2,
        2: 3,
        3: 7,
        5: 15,
        6: 17,
    }

    PIECE_ID_TO_NAME = {
        11: "white pyramid",
         0: "blue log",
         2: "violet square",
         3: "azure L",
         7: "yellow mirror L",
        15: "green S",
        17: "red Z",
    }

    def __init__(self, log_file):
        self._log_file = log_file

        self._board = np.zeros(shape=(config.BOARD_HEIGHT, config.BOARD_WIDTH), dtype=int)
        self.sequence_num = None
        self._fresh_piece = False
        self._round = 0

        self._board_gaps_count = 0
        self._board_max_height = 0

        self._lines_cleared = 0

    def update_lines_cleared(self, params):
        """Handle Ext:LinesCleared - available only in netris-env."""
        scr, lines_cleared = [int(p) for p in params]

        # Skip data if they don't belong to robot
        if scr != config.SCREEN_ID:
            return

        self._lines_cleared = lines_cleared

    def update_new_piece(self, params):
        """
        Handle NewPiece from netris. Unfortunately game provide only
        sequence number not real piece id.
        """
        self.sequence_num = params[0]
        self._fresh_piece = True
        self._round += 1

    def update_row(self, params):
        """
        Handle RowUpdate command from netris. Update board.
        """
        scr, y, *row = [int(p) for p in params]

        # Skip data if they don't belong to robot
        if scr != config.SCREEN_ID:
            return ""

        # Netris inform about switch from "piece block" to "fixed block" starting
        # from second RowUpdate command after NewPiece command. This is to late,
        # for prediction, so better is assume that first line is always empty.
        if y != config.TOP_LINE:
            for x, val in enumerate(row):
                self._board[config.BOARD_HEIGHT - 1 - y][x] = config.FULL_BLOCK if val else config.EMPTY_BLOCK

        # Create board status if this is new piece
        msg = ""
        if self._fresh_piece and y == config.TOP_LINE:
            msg = self.create_status_message(top_row=row, game_is_over=False)
            self._fresh_piece = False

        return msg

    def create_status_message(self, top_row, game_is_over):
        """Create status message for agent."""
        new_board = self._board_with_piece_bits(top_row)
        flat_board = "".join([("%0.2f " % val) for val in new_board])
        reward = self._reward(game_is_over)
        game_is_over = int(game_is_over)

        self.log("reward:", reward)

        # Format status message
        return str(game_is_over) + " " + str(reward) + " " + flat_board + "\n"

    def _board_with_piece_bits(self, top_row):
        """Create flat board with four blocks representing piece."""
        piece_bits = self._piece_as_bits(top_row)

        # Combine board with piece bits
        new_board = np.copy(self._board).astype('float')
        for x in range(config.BOARD_WIDTH):
            if 3 < x < 8:
                new_board[0][x] = piece_bits[x - 4]

        return new_board.flatten()

    def _reward(self, game_is_over):
        """
        Calculate final reward. In competitive game this values, where extra
        lines are added by enemy, will be invalid.

        Return value in range [0, 1]
        """

        # Punish for not filling valleys
        if self._lines_cleared == 0:
            top, bottom = self._calc_valley()
            penalty_valley = (top - bottom) / (config.BOARD_HEIGHT - 1)
            if penalty_valley < 0 or penalty_valley > 1:
                raise Exception("Penalty out of range: %f, (top, bottom) = (%d, %d)"
                    % (penalty_valley, top, bottom))
        # When cleared lines reveal new bottom then skip
        else:
            penalty_valley = 0

        # Punish for creating gaps
        gaps_count = self._calc_gaps()
        penalty_gaps = min(max(0, gaps_count - self._board_gaps_count), config.MAX_ALLOWED_GAPS) / config.MAX_ALLOWED_GAPS
        if penalty_gaps < 0 or penalty_gaps > 1:
            raise Exception("Penalty out of range: %f, gaps = %d, board_gaps = %d"
                % (penalty_gaps, gaps_count, self._board_gaps_count))
        self._board_gaps_count = gaps_count

        # # Punish for increasing the height
        # max_height = self._calc_height()
        # penalty_height = max(0, max_height - self._board_max_height) / MAX_PIECE_HEIGHT
        # if penalty_height < 0 or penalty_height > 1:
        #     raise Exception("Penalty out of range: %f, max_height = %d, board_height = %d"
        #         % (penalty_height, max_height, self._board_max_height))
        # self._board_max_height = max_height

        # Punish for ending the game
        if game_is_over:
            score = 0
        # Reward for adding piece
        elif self._lines_cleared == 0:
            score = 0.08
        # Reward for clearing lines
        else:
            score = (self._lines_cleared + (self._lines_cleared - 1)) / config.REWAD_THRESHOLDS

        # Reset counter
        self._lines_cleared = 0

        # Normalize reward with punishment
        reward = score * (0.5 * (1 - penalty_valley) + 0.5 * (1 - penalty_gaps))

        if reward < 0 or reward > 1:
            raise Exception("Reward out of range: %f" % (reward))

        return reward

    def _calc_gaps(self):
        """Count all gaps (blocks that can't be reached in next tour)."""
        gaps_count = 0
        for x in range(config.BOARD_WIDTH):
            is_roof = False
            for y in range(config.BOARD_HEIGHT):
                if self._board[y][x] == config.FULL_BLOCK and not is_roof:
                    is_roof = True
                elif self._board[y][x] == config.EMPTY_BLOCK and is_roof:
                    gaps_count += 1

        return gaps_count

    def _calc_valley(self):
        """Get max and min height."""
        max_height = 0
        min_height = config.BOARD_HEIGHT - 1
        for x in range(config.BOARD_WIDTH):
            for y in range(config.BOARD_HEIGHT):
                if self._board[y][x] == config.FULL_BLOCK:
                    max_height = max(max_height, config.BOARD_HEIGHT - y)
                    min_height = min(min_height, config.BOARD_HEIGHT - y)
                    break
                elif y == config.BOARD_HEIGHT - 1 and self._board[y][x] == config.EMPTY_BLOCK:
                    max_height = max(max_height, 0)
                    min_height = min(min_height, 0)
                    break

        return max_height, min_height

    def _calc_height(self):
        """Calculate max height."""
        max_height = 0
        for x in range(config.BOARD_WIDTH):
            for y in range(config.BOARD_HEIGHT):
                if self._board[y][x] == config.FULL_BLOCK:
                    max_height = max(max_height, config.BOARD_HEIGHT - y)
                    break

        return max_height

    def _piece_as_bits(self, top_row):
        """Extract new piece and create his four bits representation."""
        for color_type in top_row:
            # Block of moving piece have negative values
            if color_type < 0:
                self.log("Extracted piece:", self._piece_name_by_color(color_type))
                piece = self.COLOR_TO_PIECE[color_type]
                # Piece +1 because 0 is reserved for "empty block"
                return [float(ch) for ch in list('{0:04b}'.format(piece + 1))]

        return [0.0, 0.0, 0.0, 0.0]

    def _normalized_board(self, top_row):
        """Create flat board with normalized values."""
        norm_piece = self._normalized_piece(top_row)

        # Combine board with normalized piece
        norm_board = np.copy(self._board).astype('float')
        for x, color_type in enumerate(top_row):
            if color_type < 0:
                norm_board[0][x] = norm_piece

        return norm_board.flatten()

    def _normalized_piece(self, top_row):
        """Extract new piece (order number) from row."""
        for color_type in top_row:
            # Block of moving piece have negative values
            if color_type < 0:
                self.log("Extracted piece:", self._piece_name_by_color(color_type))
                piece = self.COLOR_TO_PIECE[color_type]
                # All pieces with full block
                all_pieces = len(self.PIECE_TO_PIECE_ID) + 1
                # Piece +1 because 0 is reserved for "empty block"
                return (piece + 1) / all_pieces

        return None

    def _piece_name_by_color(self, color_type):
        """Convert color id to piece name."""
        piece = self.COLOR_TO_PIECE[color_type]
        piece_id = self.PIECE_TO_PIECE_ID[piece]
        return self.PIECE_ID_TO_NAME[piece_id]

    def log(self, *args, **kwargs):
        """Print log to other terminal or file."""
        if self._log_file:
            print(*args, **kwargs, file=self._log_file)
