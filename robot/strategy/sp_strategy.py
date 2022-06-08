#!/usr/bin/env python3

# Author: Mateusz Janda <mateusz janda at gmail com>
# Site: github.com/MateuszJanda/netris-ai-robot
# Ad maiorem Dei gloriam


"""
Stevens and Pradhan - Deep Reinforcement Learning
"""

import numpy as np
from robot.models.netris_solver import NetrisSolver
from robot.models import board_helpers
from robot import config
from robot import utils


EPSILON_DELTA = 9 * 10**-7
EPSILON_RAND = 0.05
MIN_EPSILON = 0.1
UPDATE_MODEL_AT_STEP = 1000


class SPStrategy:
    """SP strategy."""

    def __init__(self, epsilon_delta=EPSILON_DELTA, epsilon_rand=EPSILON_RAND, min_epsilon=MIN_EPSILON):
        self._epsilon_delta = epsilon_delta
        self._epsilon_rand = epsilon_rand
        self._min_epsilon = min_epsilon

    def play(self, total_steps, epsilon, env, agent, enable_learning):
        """
        Play one game. Stevens and Pradhan scoring (based on mistakes).
        """
        episode_reward = 0
        episode_lines = 0
        solver = NetrisSolver()

        # Reset environment and get initial state
        _, _, current_piece, raw_current_state, current_state = env.reset()

        # Reset flag and start iterating until episode ends
        last_round = False
        last_score = 0

        while not last_round:
            # Explore other actions with probability epsilon
            r = np.random.random()
            if enable_learning and r < epsilon:
                if r < self._epsilon_rand:
                    action = np.random.randint(0, config.ACTION_SPACE_SIZE)
                else:
                    action = solver.action(current_piece, raw_current_state)
            else:
                q_values = agent.q_values_for_state(current_state)
                # Choose best action
                action = np.argmax(q_values)

                if np.isnan(q_values[action]) or np.isinf(q_values[action]):
                    print("Error: Q value = ", q_values[action])
                    exit()

            last_round, lines, next_piece, raw_next_state, next_state = env.step(action)
            episode_lines += lines

            current_score = self._adjust_score(raw_next_state, lines)
            reward = current_score - last_score
            last_score = current_score

            # Transform new continuous state to new discrete state and count reward
            episode_reward += reward

            # Every step update replay memory and train NN model
            if enable_learning:
                transition = utils.Transition(current_state, action, reward, next_state, last_round)
                agent.update_replay_memory(transition)
                agent.train(last_round)

                epsilon = self._adjust_epsilon(epsilon)

            current_piece = next_piece
            current_state = next_state
            raw_current_state = raw_next_state

            # Update Q' model (this prevent instability when training)
            if enable_learning and total_steps % UPDATE_MODEL_AT_STEP == 0:
                agent.update_caching_model()

            total_steps += 1

        return total_steps, episode_reward, episode_lines, epsilon

    def _adjust_score(self, board, lines):
        """
        Adjust reward - intermediate scoring function.
        """
        return -0.51 * board_helpers.aggregate_height(board) + 0.76 * lines \
            - 0.36 * board_helpers.holes(board) - 0.18 * board_helpers.bumpiness(board)

    def _adjust_epsilon(self, epsilon):
        """
        Decay epsilon.
        """
        if epsilon > self._min_epsilon:
            epsilon -= self._epsilon_delta

        return epsilon
