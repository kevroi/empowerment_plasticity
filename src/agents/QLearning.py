# agents/q_learning_agent.py
import numpy as np
import random
from collections import defaultdict

class QLearning:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def _argmax(self, values):
        # argmax with random tie-breaking
        max_value = np.max(values)
        candidates = [i for i, v in enumerate(values) if v == max_value]
        return random.choice(candidates)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return self._argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done, truncated):
        target = reward

        # don't bootstrap if done or truncated
        should_bootstrap = not (done or truncated)
        if should_bootstrap:
            max_next_q = np.max(self.q_table[next_state])
            target += self.gamma * max_next_q

        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])