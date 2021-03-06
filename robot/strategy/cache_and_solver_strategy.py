#!/usr/bin/env python3

# Author: Mateusz Janda <mateusz janda at gmail com>
# Site: github.com/MateuszJanda/netris-ai-robot
# Ad maiorem Dei gloriam


"""
Reinforcement learning - Deep Q-Network/Learning. Simple strategy (reward based
on cleared lines) with solver, cached model and episilon calculated at the end
of episode.
"""

import numpy as np
from robot.models.netris_solver import NetrisSolver
from robot import config
from robot import utils


# Exploration settings - try/explore random action with probability epsilon
EPSILON_DECAY = 0.99995     # Decay epsilon. Smarter NN is, then less random action should be taken
MIN_EPSILON = 0.02          # Epsilon shouldn't less than this. We always want to check something new
RAND_TRESHOLD = 0.005
UPDATE_MODEL_AT_STEP = 100


class CacheAndSolverStrategy:
    """
    Strategy (reward based on cleared lines) with solver, cached model
    and episilon calculated at the end of episode.
    """

    def __init__(self, epsilon_decay=EPSILON_DECAY, min_epsilon=MIN_EPSILON,
                 random_treshold=RAND_TRESHOLD):
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon
        self._random_treshold = random_treshold

    def play(self, episode, total_steps, epsilon, env, agent, enable_learning):
        """
        Play one game. Scoring: lines with solver support.
        """
        episode_reward = 0
        episode_lines = 0
        solver = NetrisSolver()

        # Reset environment and get initial state
        _, _, current_piece, raw_current_state, current_state = env.reset()

        # Reset flag and start iterating until episode ends
        last_round = False

        while not last_round and episode_lines < config.MAX_LINES_IN_EPISODE:
            # Explore other actions with probability epsilon
            r = np.random.random()
            if enable_learning and r < epsilon:
                if r < self._random_treshold:
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

            # Transform new continuous state to new discrete state and count reward
            reward = self._adjust_reward(lines)
            episode_reward += reward

            # Every step update replay memory and train NN model
            if enable_learning:
                transition = utils.Transition(current_state, action, reward, next_state, last_round)
                agent.update_replay_memory(transition)
                agent.train()

            # Update Q' model (this prevent instability when training)
            # Caching and training model can't differ when snapshot is saved
            if enable_learning and (total_steps % UPDATE_MODEL_AT_STEP == 0 or \
              (episode > 0 and episode % config.SNAPSHOT_MODULO == 0)):
                agent.update_caching_model()

            current_piece = next_piece
            current_state = next_state
            raw_current_state = raw_next_state

            total_steps += 1

        if enable_learning:
            epsilon = self._adjust_epsilon(epsilon)

        return total_steps, episode_reward, episode_lines, epsilon

    def _adjust_reward(self, lines):
        """
        Adjust reward - lines_cleared**2
        """
        return lines**2

    def _adjust_epsilon(self, epsilon):
        """
        Decay epsilon.
        """
        if epsilon > self._min_epsilon:
            epsilon = epsilon * self._epsilon_decay

        return epsilon
