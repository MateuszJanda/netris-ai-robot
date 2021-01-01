#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda/netris-ai-robot
Ad maiorem Dei gloriam
"""

import numpy as np
import random
from collections import deque
import config


class Agent:
    """DQN agent."""

    def __init__(self, model, episode=None):
        # Build NN model
        self._model = model

        # An array with last REPLAY_MEMORY_SIZE steps for training
        self.replay_memory = deque(maxlen=config.REPLAY_MEMORY_SIZE)

    def update_replay_memory(self, transition):
        """Adds transition (step's data) to a replay memory."""
        self.replay_memory.append(transition)

    def q_values_for_state(self, state):
        """
        Query NN model for Q values for current observation (state).
        """
        return self._model.predict(batch_size=1, state=state)[0]

    def train(self, last_round):
        """Trains NN model every step during episode."""

        # Start training only if certain number of samples is already saved in
        # replay memory
        if len(self.replay_memory) < config.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a mini-batch of random samples from replay memory
        minibatch = random.sample(self.replay_memory, config.MINIBATCH_SIZE)
        current_q_values, future_q_values = self.q_values_for_historic(minibatch)

        input_states = []     # Input (X)
        target_q_values = []  # Output (y)

        for index, transition in enumerate(minibatch):
            # If last round assign reward to Q
            if transition.last_round:
                new_q = transition.reward
            # Otherwise set new Q from future states (Bellman equation)
            else:
                max_future_q = np.max(future_q_values[index])
                new_q = transition.reward + config.DISCOUNT * max_future_q

            # Update Q value for given action, and append to training output (y) data
            current_qs = current_q_values[index]
            current_qs[transition.action] = new_q
            target_q_values.append(current_qs)

            # Append to training input (X) data
            input_states.append(transition.current_state)

        # Fit with new Q values
        self._model.fit(x=np.array(input_states), y=np.array(target_q_values),
            batch_size=config.MINIBATCH_SIZE, verbose=0, shuffle=False)

    def q_values_for_historic(self, minibatch):
        """
        Take historic current and next states (from minibach) and query NN model
        for Q values.
        """
        current_states = np.array([transition.current_state for transition in minibatch])
        current_q_values = self._model.predict(config.MINIBATCH_SIZE, current_states)

        next_states = np.array([transition.next_state for transition in minibatch])
        future_q_values = self._model.predict(config.MINIBATCH_SIZE, next_states)

        return current_q_values, future_q_values

    def get_tf_model(self):
        """Getter to tensorflow model."""
        return self._model.get_tf_model()

    def reshape_input(self, state):
        """Reshape input state if needed later by model."""
        return self._model.reshape_input(state)
