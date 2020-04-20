import numpy as np
import scipy.signal

class Rollout:
    def __init__(self, context_dim, state_dim, ep_len, dc_interv, gamma=0.97, lam=1):

        self.gamma, self.lam = gamma, lam  # for GAE computation

        self.position, self.dc_episodes = 0, 0
        self.dc_interv = dc_interv

        self.state_context = np.zeros((ep_len, state_dim + context_dim))
        self.action = np.zeros((ep_len, 1))
        self.reward = np.zeros(ep_len)
        self.value = np.zeros(ep_len)
        self.q = np.zeros(ep_len)

        self.reset_dc_buff()

    def reset_dc_buff(self):
        self.dc_episodes = 0
        self.dc_state = []
        self.dc_one_hot_act = []
        self.dc_context = np.zeros(self.dc_interv)

    def push(self, state, state_context, action, reward, value):
        self.dc_state.append(state)
        self.dc_one_hot_act.append(self.make_one_hot_act(action))

        self.state_context[self.position] = state_context
        self.action[self.position] = action
        self.reward[self.position] = reward
        self.value[self.position] = value
        self.position += 1

    def postprocessing_ep(self, log_p, context):
        rewards = np.append(self.reward, np.array([0]))
        values = np.append(self.value, np.array([0]))
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.rewards = self.discount_cumsum(rewards, self.gamma)[:-1]
        self.advantage = self.discount_cumsum(deltas, self.gamma * self.lam)
        self.q = log_p

        self.dc_context[self.dc_episodes] = context
        self.dc_episodes += 1

    def get_buffer(self):
        self.position = 0
        return [self.state_context, self.action, self.normalize(self.advantage.copy()),
                self.q, self.normalize(self.reward.copy())]

    def get_dc_buff(self):
        return self.dc_state, self.dc_one_hot_act, self.dc_context

    @staticmethod
    def make_one_hot_act(action):
        action_one_not = np.zeros(3)
        action_one_not[action] = 1
        return action_one_not

    @staticmethod
    def discount_cumsum(x, coef):
        return scipy.signal.lfilter([1], [1, float(-coef)], x[::-1], axis=0)[::-1]

    @staticmethod
    def normalize(array):
        return (array - array.mean()) / (array.std() + 1e-5)