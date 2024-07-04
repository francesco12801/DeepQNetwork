
import numpy as np

class TabularQAgent:
    def __init__(self, gamma, epsilon, lr, n_actions, state_bins, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.state_bins = state_bins
        self.state_space_size = np.prod([len(b) + 1 for b in state_bins])
        self.Q = np.zeros((self.state_space_size, n_actions))

    def discretize_state(self, state):
        state_discrete = sum([np.digitize(state[i], self.state_bins[i]) * (len(self.state_bins[i]) ** i) for i in range(len(state))])
        return state_discrete

    def choose_action(self, state):
        state_discrete = self.discretize_state(state)
        if np.random.random() > self.epsilon:
            action = np.argmax(self.Q[state_discrete, :])
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self, state, action, reward, state_):
        state_discrete = self.discretize_state(state)
        state_discrete_ = self.discretize_state(state_)
        max_future_q = np.max(self.Q[state_discrete_, :])
        current_q = self.Q[state_discrete, action]
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.Q[state_discrete, action] = new_q

        # Decay epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

