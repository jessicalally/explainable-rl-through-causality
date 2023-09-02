from collections import deque, namedtuple
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .rl_agent import RLAgent

# Tuned to Lunar Lander environment, as in:
# [Explainable Reinforcement Learning Through a Causal Lens]
# https://arxiv.org/abs/1905.10958

class DDQN(RLAgent):

    class ReplayBuffer(object):
        def __init__(self, max_size):
            self.max_buffer_size = max_size
            self.buffer_size = 0
            self.index = 0
            self.buffer = deque(maxlen=self.max_buffer_size)

        def save(self, state, action, reward, next_state, done):
            self.buffer.append([state, action, reward, next_state, done])

            if self.buffer_size < self.max_buffer_size:
                self.buffer_size += 1

        def sample_from_buffer(self, batch_size):
            transitions = np.array(random.sample(self.buffer, k=batch_size))

            states = torch.from_numpy(np.vstack(transitions[:, 0])).float()
            actions = torch.from_numpy(np.vstack(transitions[:, 1])).long()
            rewards = torch.from_numpy(np.vstack(transitions[:, 2])).float()
            next_states = torch.from_numpy(np.vstack(transitions[:, 3])).float()
            dones = torch.from_numpy(np.vstack(transitions[:, 4]).astype(np.uint8)).float()

            return states, actions, rewards, next_states, dones

    class Network(nn.Module):
        def __init__(self, state_space, action_space):
            super().__init__()
            self.input_layer = nn.Linear(state_space, 128)
            self.hidden_layer = nn.Linear(128, 128)
            self.output_layer = nn.Linear(128, action_space)

        def forward(self, state):
            x = self.input_layer(state)
            x = F.relu(x)
            x = self.hidden_layer(x)
            x = F.relu(x)
            return self.output_layer(x)

    def __init__(
            self,
            environment,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.996,
            batch_size=128,
            lr=0.001,
            reward_threshold=100):
        self.name = "ddqn"

        # Environment
        self.environment = environment
        self.env = environment.env
        self.test_env = copy.deepcopy(self.env)
        self.action_space = environment.action_space
        self.state_space = environment.state_space
        self.reward_threshold = reward_threshold

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.lr = lr
        self.min_epsilon = 0.01
        self.buffer_size = 100000
        self.update_network_frequency = 4
        self.replay_buffer = self.ReplayBuffer(self.buffer_size)
        self.update_target_network_frequency = 100
        self.network = self.Network(self.state_space, self.action_space)
        self.target_network = self.Network(self.state_space, self.action_space)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)


    def _choose_action_epsilon_greedy(self, state):
        state = torch.DoubleTensor(state).unsqueeze(0)

        self.network.eval()
        q_values = self.network(state)
        self.network.train()

        if np.random.random() > self.epsilon:
            return np.argmax(q_values.cpu().data.numpy())
        else:
            return random.randint(0, self.action_space - 1)

    def _choose_action_deterministic(self, state):
        state = torch.DoubleTensor(state).unsqueeze(0)

        self.network.eval()
        action_values = self.network(state)

        return np.argmax(action_values.cpu().data.numpy())
    
    def _get_updated_q_values(self, next_states, rewards, dones):
        q_values_next_states = self.target_network(next_states.to(
            torch.float64)).detach().max(1)[0].unsqueeze(1)
        updated_q_values = rewards + self.gamma * q_values_next_states * (1 - dones)

        return updated_q_values
    
    def _update_optimizer(self, q, updated_q_values):
        loss = F.mse_loss(q, updated_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _update_target_network(self):
        if self.replay_buffer.buffer_size % self.update_target_network_frequency == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def _learn(self):
        if self.replay_buffer.buffer_size < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_from_buffer(
            self.batch_size)

        q = self.network(states.to(torch.float64)).gather(1, actions)
        updated_q_values = self._get_updated_q_values(next_states, rewards, dones)
        
        self._update_optimizer(q, updated_q_values)
        self._update_target_network()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def train(self, episodes=4000):
        print('Performing DDQN algorithm...')
        reward_test_data = []
        transition_test_data = []

        scores = []
        eps_history = []

        for e in range(episodes):
            terminated = False
            truncated = False
            score = 0
            state, _ = self.env.reset()
            steps = 0

            while not (terminated or truncated):
                action = self._choose_action_epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                
                self.replay_buffer.save(
                    state, action, reward, next_state, terminated or truncated)

                state = next_state

                if steps > 0 and steps % self.update_network_frequency == 0:
                    self._learn()

                if terminated:
                    reward_test_data.append(
                        np.concatenate((next_state, np.array(0)), axis=None)
                    )
                else:
                    reward_test_data.append(np.concatenate(
                        (next_state, np.array(reward)), axis=None))

                transition_test_data.append(
                    np.concatenate(
                        (state, np.array(action), next_state),
                        axis=None))

                steps += 1
                score += reward

            eps_history.append(self.epsilon)
            scores.append(score)

            avg_score = np.mean(scores[max(0, e - 100):(e + 1)])
            print("episode: {}/{}, score: {}, avg score: {}".format(e,
                  episodes, score, avg_score))

            if avg_score > self.reward_threshold:
                break

        print('Finished DDQN Algorithm...')

        return np.array(transition_test_data), np.array(reward_test_data)

    def generate_test_data_for_causal_discovery(
            self, num_datapoints, use_sum_rewards=False):
        transition_test_data = []
        reward_test_data = []

        print('Generating test data for DDQN algorithm...')

        while len(transition_test_data) < num_datapoints:
            terminated = False
            truncated = False
            state, _ = self.env.reset()

            while not (terminated or truncated):
                action = self._choose_action_epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)

                if use_sum_rewards and terminated:
                    reward_test_data.append(
                        np.concatenate((next_state, np.array(0)), axis=None)
                    )
                else:
                    reward_test_data.append(np.concatenate(
                        (next_state, np.array(reward)), axis=None))

                transition_test_data.append(
                    np.concatenate(
                        (state, np.array(action), next_state),
                        axis=None))

                state = next_state

            print(
                "num datapoints collected so far: {}".format(
                    len(transition_test_data)))

        print('Finished generating test data for DDQN Algorithm...')

        return np.array(transition_test_data), np.array(reward_test_data)
    
    def generate_test_data_for_scm_training(
            self, num_datapoints, use_sum_rewards=False):
        transition_test_data = []
        reward_test_data = []

        print('Generating test data for DDQN algorithm...')

        while len(transition_test_data) < num_datapoints:
            terminated = False
            truncated = False
            state, _ = self.env.reset()

            while not (terminated or truncated):
                action = random.randint(0, self.action_space - 1)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)

                if use_sum_rewards and terminated:
                    reward_test_data.append(
                        np.concatenate((next_state, np.array(0)), axis=None)
                    )
                else:
                    reward_test_data.append(np.concatenate(
                        (next_state, np.array(reward)), axis=None))

                transition_test_data.append(
                    np.concatenate(
                        (state, np.array(action), next_state),
                        axis=None))

                state = next_state

            print(
                "num datapoints collected so far: {}".format(
                    len(transition_test_data)))

        print('Finished generating test data for DDQN Algorithm...')

        return np.array(transition_test_data), np.array(reward_test_data)
    
    def generate_random_test_data(
            self, num_datapoints, use_sum_rewards=False):
        transition_test_data = []
        reward_test_data = []

        print('Generating test data for DDQN algorithm...')

        while len(transition_test_data) < num_datapoints:
            terminated = False
            truncated = False
            state, _ = self.env.reset()

            while not (terminated or truncated):
                action = self._choose_action_epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)

                if use_sum_rewards and terminated:
                    reward_test_data.append(
                        np.concatenate((next_state, np.array(0)), axis=None)
                    )
                else:
                    reward_test_data.append(np.concatenate(
                        (next_state, np.array(reward)), axis=None))

                transition_test_data.append(
                    np.concatenate(
                        (state, np.array(action), next_state),
                        axis=None))

                state = self.env.observation_space.sample()

            print(
                "num datapoints collected so far: {}".format(
                    len(transition_test_data)))

        print('Finished generating test data for DDQN Algorithm...')

        return np.array(transition_test_data), np.array(reward_test_data)

    # Methods needed for estimating feature importance
    def get_q_values(self, state):
        q_func = self.network.eval()

        state_tensor = torch.DoubleTensor(state).unsqueeze(0)
        q_values = q_func(state_tensor).cpu().data.numpy()[0]

        return q_values

    def get_optimal_action(self, state):
        return self._choose_action_deterministic(state)
