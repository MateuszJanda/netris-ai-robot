#!/usr/bin/env python3

# Author: Mateusz Janda <mateusz janda at gmail com>
# Site: github.com/MateuszJanda/netris-ai-robot
# Ad maiorem Dei gloriam

import random
from collections import deque
import numpy as np
from robot import config


class CachingAgent:
    """
    DQN agent with two models, one for training and second for prediction
    (caching - copied from training model every few episodes).
    """

    def __init__(self, training_model, caching_model):
        # Build NN model
        self._training_model = training_model
        self._caching_model = caching_model

        # An array with last REPLAY_MEMORY_SIZE steps for training
        self.replay_memory = deque(maxlen=config.REPLAY_MEMORY_SIZE)

    def load_replay_memory(self, replay_memory):
        """Load reply memory from snapshot."""
        self.replay_memory = replay_memory

    def update_replay_memory(self, transition):
        """Adds transition (step's data) to a replay memory."""
        self.replay_memory.append(transition)

    def q_values_for_state(self, state):
        """
        Query NN model for Q values for current observation (state).
        """
        return self._training_model.predict(batch_size=1, state=state)[0]

    def train(self):
        """Trains NN model every step during episode."""

        # Start training only if certain number of samples is already saved in
        # replay memory
        if len(self.replay_memory) < config.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a mini-batch of random samples from replay memory
        minibatch = random.sample(self.replay_memory, config.MINIBATCH_SIZE)
        current_q_values, future_q_values = self._q_values_for_historic(minibatch)

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
        self._training_model.fit(x=np.array(input_states), y=np.array(target_q_values),
            batch_size=config.MINIBATCH_SIZE, verbose=0, shuffle=False)

    def _q_values_for_historic(self, minibatch):
        """
        Take historic current and next states (from minibach) and query NN model
        for Q values.
        """
        current_states = np.array([transition.current_state for transition in minibatch])
        current_q_values = self._training_model.predict(config.MINIBATCH_SIZE, current_states)

        next_states = np.array([transition.next_state for transition in minibatch])
        future_q_values = self._caching_model.predict(config.MINIBATCH_SIZE, next_states)

        return current_q_values, future_q_values

    def update_caching_model(self):
        """Copy training model into caching model."""
        self._caching_model.get_tf_model().set_weights(self._training_model.get_tf_model().get_weights())

    def get_tf_model(self):
        """Getter to tensorflow model."""
        return self._training_model.get_tf_model()
