import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from .rl_agent import RLAgent
import copy

# Tuned to Cartpole environment, as in:
# [Explainable Reinforcement Learning Through a Causal Lens]
# https://arxiv.org/abs/1905.10958


class PolicyGradient(RLAgent):
    name = "pg"

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
            super().__init__()

            self.fc_1 = nn.Linear(input_dim, hidden_dim)
            self.fc_2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.fc_1(x)
            x = self.dropout(x)
            x = F.relu(x)
            x = self.fc_2(x)
            return x

        def init_weights(self, m):
            if isinstance(m, nn.Linear):
                weights = torch.nn.init.xavier_normal_(m.weight)
                bias = m.bias.data.fill_(0)

                return weights, bias


    def __init__(self, environment):
        self.env = environment.env
        self.test_env = copy.deepcopy(self.env)
        self.action_space = environment.action_space
        self.state_space = environment.state_space
        self.policy = self.MLP(
            self.env.observation_space.shape[0],
            128,
            self.env.action_space.n)
        self.policy.apply(self.policy.init_weights)
        self.lr = 0.01
        self.optimiser = optim.Adam(self.policy.parameters(), self.lr)

    def calculate_returns(self, rewards, discount_factor, normalise=True):
        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + R * discount_factor
            returns.insert(0, R)

        returns = torch.tensor(returns)

        if normalise:
            returns = (returns - returns.mean()) / returns.std()

        return returns

    def update_policy(self, returns, log_prob_actions):
        returns = returns.detach()
        loss = - (returns * log_prob_actions).sum()
        self.optimiser.zero_grad()

        loss.backward()
        self.optimiser.step()

        return loss.item()

    def evaluate(self):
        self.policy.eval()

        done = False
        episode_reward = 0

        state, _ = self.test_env.reset()

        while not done:
            with torch.no_grad():
                action_pred = self.policy(torch.DoubleTensor(state))
                action_prob = F.softmax(action_pred, dim=-1)

            action = torch.argmax(action_prob, dim=-1)
            state, reward, done, _, _ = self.test_env.step(action.item())
            episode_reward += reward

        return episode_reward

    def train(
            self,
            episodes=2000,
            gamma=0.95,
            lr=0.01,
            reward_threshold=495,
            timestep=10):
        action_influence_data_set = []
        causal_discovery_data_set = []
        train_rewards = []
        test_rewards = []

        print('Performing Policy Gradient..')

        for episode in range(episodes):
            steps = 0
            episode_reward = 0
            log_prob_actions = []
            states = []
            actions = []
            rewards = []
            done = False

            state, _ = self.env.reset()
            prev_reward = 0

            while not done:
                steps += 1

                action_prediction = self.policy(torch.DoubleTensor(state))
                action_probability = F.softmax(action_prediction, dim=-1)
                dist = distributions.Categorical(action_probability)

                chosen_action = dist.sample()
                log_prob_action = dist.log_prob(chosen_action)
                log_prob_actions.append(log_prob_action)

                next_state, reward, done, _, _ = self.env.step(
                    chosen_action.item())
                episode_reward += reward

                rewards.append(reward)

                datapoint = np.concatenate(
                    (state, np.array(chosen_action), np.array(prev_reward), next_state), axis=None)
                causal_discovery_data_set.append(datapoint)
                action_influence_data_set.append(
                    (state, chosen_action, reward, next_state))

                state = next_state
                prev_reward = reward

            log_prob_actions = torch.stack(log_prob_actions)

            returns = self.calculate_returns(rewards, gamma)
            loss = self.update_policy(returns, log_prob_actions)

            test_reward = self.evaluate()

            train_rewards.append(episode_reward)
            test_rewards.append(test_reward)

            mean_train_rewards = np.mean(train_rewards[-5:])
            mean_test_rewards = np.mean(test_rewards[-5:])

            if episode % timestep == 0:
                print(
                    f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')

            if mean_test_rewards >= reward_threshold:
                print(f'Reached reward threshold in {episode} episodes')
                break

        print('Finished Policy Gradient...')

        return np.array(action_influence_data_set), np.array(causal_discovery_data_set)

    def generate_test_data(
            self,
            num_datapoints=1000,
            episodes=2000,
            gamma=0.99,
            lr=0.01,
            reward_threshold=800,
            timestep=10):
        causal_discovery_data_set = []
        train_rewards = []
        test_rewards = []
        episode = 0

        while len(causal_discovery_data_set) < num_datapoints:
            steps = 0
            episode_reward = 0
            log_prob_actions = []
            rewards = []
            done = False

            state, _ = self.env.reset()
            prev_reward = 0

            while not done and len(causal_discovery_data_set) < num_datapoints:
                steps += 1
                
                action_prediction = self.policy(torch.DoubleTensor(state))
                action_probability = F.softmax(action_prediction, dim=-1)
                dist = distributions.Categorical(action_probability)

                chosen_action = dist.sample()
                log_prob_action = dist.log_prob(chosen_action)
                log_prob_actions.append(log_prob_action)

                next_state, reward, done, _, _ = self.env.step(
                    chosen_action.item())
                episode_reward += reward

                rewards.append(reward)

                datapoint = np.concatenate(
                    (state, np.array(chosen_action), np.array(prev_reward), next_state), axis=None)
                causal_discovery_data_set.append(datapoint)

                state = next_state
                prev_reward = reward

            test_reward = self.evaluate()

            train_rewards.append(episode_reward)
            test_rewards.append(test_reward)

            mean_train_rewards = np.mean(train_rewards[-5:])
            mean_test_rewards = np.mean(test_rewards[-5:])

            if episode % timestep == 0:
                print(
                    f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')

            episode += 1
            print(f'Num datapoints collected so far: {len(causal_discovery_data_set)}')

        print('Finished Policy Gradient...')

        return np.array(causal_discovery_data_set)
    

    def get_q_func(self):
        # TODO: can we use the action probabilities as an approximation?
        pass
    

    def get_optimal_action(self, state):
        action_prediction = self.policy(torch.DoubleTensor(state))
        action_probability = F.softmax(action_prediction, dim=-1)
        dist = distributions.Categorical(action_probability)

        return dist.sample()
