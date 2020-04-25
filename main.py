import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from env import CroudsorsingEnv
from buffer import Rollout
from config import Parametrs
from models import Actor, Critic, Discriminator


class VALOR:
    def __init__(self):

        self.args = Parametrs()

        self.actor = Actor(state_dim=self.args.state_dim, context_dim=self.args.num_context)
        self.critic = Critic(state_dim=self.args.state_dim, context_dim=self.args.num_context)
        self.discriminator = Discriminator(state_dim=self.args.state_dim, input_dim=self.args.state_dim + self.args.action_dim,
                                           context_dim=self.args.num_context, hidden_dim=self.args.dc_hidden_dim,
                                           num_layers=self.args.dc_num_layers)

        self.buffer = Rollout(ep_len=self.args.ep_len, eps_for_update=self.args.eps_for_update, eps_for_dc=self.args.eps_for_dc,
                              gamma=self.args.gamma, lam=self.args.lam)

        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.lr_policy)
        self.value_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.lr_value)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.args.lr_descr)

        self.context = Categorical(logits=torch.Tensor(np.ones(self.args.num_context)))

    def act(self, state, context_one_hot):
        with torch.no_grad():
            logits, x_concat = self.actor(state, context_one_hot=context_one_hot)
        policy = Categorical(logits=logits)
        action = policy.sample()
        return action.numpy(), x_concat.squeeze().numpy()

    def eval_actions(self):
        logits, _ = self.actor(self.states, context_one_hot=self.contexts_one_not)

        policy = Categorical(logits=logits)
        log_prob = policy.log_prob(self.actions).unsqueeze(1)
        return log_prob

    def update_policy(self):

        self.states, self.contexts_one_not, self.actions, self.advantages, self.logs_context, self.rewards = self.buffer.get_buffer()
        self.buffer.reset_buff()

        self.__optimize_policy()
        self.__optimize_value()

    def __optimize_policy(self):
        self.actor.train()
        log_prob = self.eval_actions()
        entropy = (-log_prob).mean()

        L_policy = -(log_prob * (self.args.adv_coef * self.advantages + self.logs_context)).mean() - self.args.entropy_coef * entropy

        self.policy_optimizer.zero_grad()
        L_policy.backward()
        self.policy_optimizer.step()
        print("-" * 150)
        print(f"L_policy : {L_policy}")

        self.actor.eval()

    def __optimize_value(self):
        self.critic.train()
        for _ in range(self.args.iters_value_train):
            value = self.critic(self.states, context_one_hot=self.contexts_one_not)
            L_value = F.mse_loss(value.squeeze(), self.rewards)

            self.value_optimizer.zero_grad()
            L_value.backward()
            self.value_optimizer.step()
        self.critic.eval()

    def update_discriminator(self):
        self.discriminator.train()
        states, actions_one_hot, contexts = self.buffer.get_dc_buff()
        self.buffer.reset_dc_buff()

        for _ in range(self.args.iters_discriminator_train):
            _, logs_context = self.discriminator(states, actions_one_hot, contexts)
            L_discriminator = - logs_context.mean()

            self.discriminator_optimizer.zero_grad()
            L_discriminator.backward()
            self.discriminator_optimizer.step()

        print("-" * 150)
        print(f"L_discriminator : {L_discriminator:.3f}")
        self.discriminator.eval()

    def train(self):
        self.critic.eval()
        episodes = 0
        rewards = []
        skill_rewards = []

        for episode in range(self.args.max_episodes):
            actions = []
            state = env.reset()
            reward, total_reward, done = 0, 0, False
            context = self.context.sample()

            context_one_hot = F.one_hot(context, self.args.num_context).squeeze().float().reshape(1, -1)
            while not done:
                action, state_context = self.act(state, context_one_hot=context_one_hot)
                value = self.critic(state, context_one_hot=context_one_hot)
                next_state, reward, done, _ = env.step(action.item())
                self.buffer.push(state, context_one_hot,  action, reward, value.item())
                actions.append(action[0])
                total_reward += reward
                state = next_state

            dc_state, dc_action, context = self.buffer.get_dc_buff(context=[float(context)])
            label, log_context = self.discriminator(state=dc_state, action=dc_action, context=context)

            self.buffer.postprocessing(log_context.detach().numpy(), context)
            skill_rewards.append(log_context.item())
            rewards.append(total_reward)
            print("-" * 150)
            print(f"Ep. {episodes + 1} ||| context: {context.item():.0f} ||| label: {label.item()}"
                  f" ||| skill_r: {log_context:.3f} ||| mean_s_r: {np.mean(skill_rewards):.3f} "
                  f"reward: {total_reward} ||| mean {np.mean(rewards):.3f} ||| max {np.max(rewards)} ||| {actions}")

            episodes += 1
            if (episodes) % self.args.eps_for_update == 0:
                self.update_policy()
            if (episodes) % self.args.eps_for_dc == 0:
                self.update_discriminator()



if __name__ == '__main__':
    env = CroudsorsingEnv()
    agent = VALOR()
    agent.train()