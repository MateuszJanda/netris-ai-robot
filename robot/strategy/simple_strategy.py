#!/usr/bin/env python3

# Author: Mateusz Janda <mateusz janda at gmail com>
# Site: github.com/MateuszJanda/netris-ai-robot
# Ad maiorem Dei gloriam


"""
Reinforcement learning - Deep Q-Network/Learning.
"""

import numpy as np
from robot import config
from robot import utils


# Exploration settings - try/explore random action with probability epsilon
EPSILON_DECAY = 0.99995     # Decay epsilon. Smarter NN is, then less random action should be taken
MIN_EPSILON = 0.02          # Epsilon shouldn't less than this. We always want to check something new


class SimpleStrategy:
    """Simple strategy."""

    def __init__(self, epsilon_decay=EPSILON_DECAY, min_epsilon=MIN_EPSILON):
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon

    def play(self, _episode, total_steps, epsilon, env, agent, enable_learning):
        """
        Play one game. Scoring based on removed lines.
        """
        episode_reward = 0
        episode_lines = 0

        # Reset environment and get initial state
        _, _, _, _, current_state = env.reset()

        # Reset flag and start iterating until episode ends
        last_round = False

        while not last_round:
            # Explore other actions with probability epsilon
            if enable_learning and np.random.random() <= epsilon:
                action = np.random.randint(0, config.ACTION_SPACE_SIZE)
            else:
                q_values = agent.q_values_for_state(current_state)
                # Choose best action
                action = np.argmax(q_values)

            last_round, lines, _, _, next_state = env.step(action)
            episode_lines += lines

            # Transform new continuous state to new discrete state and count reward
            reward = self._adjust_reward(lines)
            episode_reward += reward

            # Every step update replay memory and train NN model
            if enable_learning:
                transition = utils.Transition(current_state, action, reward, next_state, last_round)
                agent.update_replay_memory(transition)
                agent.train()

                epsilon = self._adjust_epsilon(epsilon)

            total_steps += 1
            current_state = next_state

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
