from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# Hyper-parameters
Transition = namedtuple(
    'Transition', [
        'state', 'action', 'reward', 'next_state'])


class Net(nn.Module):
    def __init__(self, state_space, action_space):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_space, 100)
        self.fc2 = nn.Linear(100, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = self.fc2(x)
        return action_prob


class DQN():

    def __init__(self, environment):
        super(DQN, self).__init__()
        self.name = "dqn"
        self.env = environment.env
        self.state_space = environment.state_space
        self.action_space = environment.action_space
        self.target_net, self.act_net = Net(
            self.state_space, self.action_space), Net(
            self.state_space, self.action_space)
        self.loss_func = nn.MSELoss()

        # Hyperparameters
        self.capacity = 8000
        self.memory = [None] * self.capacity
        self.learning_rate = 1e-3
        self.memory_count = 0
        self.batch_size = 256
        self.gamma = 0.995
        self.update_count = 0
        self.optimizer = optim.Adam(
            self.act_net.parameters(),
            self.learning_rate)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
        value = self.act_net(state)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) >= 0.9:  # epslion greedy
            action = np.random.choice(range(self.action_space), 1).item()
        return action
    
    def select_action_deterministic(self, state):
        state = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
        value = self.act_net(state)
        _, index = torch.max(value, 1)
        action = index.item()

        return action

    def store_transition(self, transition):
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1
        return self.memory_count >= self.capacity

    def update(self):
        if self.memory_count >= self.capacity:
            state = torch.tensor([t.state for t in self.memory]).double()
            action = torch.LongTensor(
                [t.action for t in self.memory]).view(-1, 1).long()
            reward = torch.tensor([t.reward for t in self.memory]).double()
            next_state = torch.tensor(
                [t.next_state for t in self.memory]).double()

            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            with torch.no_grad():
                target_v = reward + self.gamma * \
                    self.target_net(next_state).max(1)[0]

            for index in BatchSampler(SubsetRandomSampler(
                    range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
                v = (self.act_net(state).gather(1, action))[index]
                loss = self.loss_func(target_v[index].unsqueeze(
                    1), (self.act_net(state).gather(1, action))[index])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.update_count += 1
                if self.update_count % 100 == 0:
                    self.target_net.load_state_dict(self.act_net.state_dict())
        else:
            print("Memory Buff is too less")

    def train(self, num_episodes=100000):
        causal_discovery_dataset = []

        scores = []
        for i_ep in range(num_episodes):
            episode_score = 0
            state = self.env.reset()

            for t in range(10000):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                causal_discovery_dataset.append(np.concatenate(
                    (state, np.array(action), next_state), axis=None))
                episode_score += reward
                transition = Transition(state, action, reward, next_state)
                self.store_transition(transition)
                state = next_state
                if done or t >= 9999:
                    print(f"episode score {episode_score}")
                    scores.append(episode_score)
                    self.update()
                    if i_ep % 10 == 0:
                        print("episodes {}, step is {} ".format(i_ep, t))
                    break

            avg_score = np.mean(scores[-10:])
            print("episode: {}/{}, score: {}".format(i_ep, num_episodes, avg_score))

            if avg_score > -180:
                break

        print("finished")

        return causal_discovery_dataset
    
    def generate_test_data_for_causal_discovery(self, num_datapoints):
        test_data = []

        print('Generating test data for DQN algorithm...')

        while len(test_data) < num_datapoints:
            state = self.env.reset()
            prev_reward = 0

            for _ in range(10000):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                test_data.append(
                        np.concatenate(
                            (state,
                            np.array(action),
                                np.array(prev_reward),
                                next_state),
                            axis=None))
                
                prev_reward = reward

                if done:
                    print("Number of datapoints collected so far: {}".format(len(test_data)))
                    break


        print('Finished generating test data for DDQN Algorithm...')

        return np.array(test_data)
    

    def generate_test_data_for_scm(self, num_datapoints):
        test_data = []

        print('Generating test data for DQN algorithm...')

        while len(test_data) < num_datapoints:
            state = self.env.reset()

            for _ in range(10000):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                test_data.append(
                    np.concatenate(
                        (state,
                         np.array(action),
                            next_state,
                            np.array(reward)),
                        axis=None))

                if done:
                    print("Number of datapoints collected so far: {}".format(len(test_data)))
                    break

        print('Finished generating test data for DQN Algorithm...')

        return np.array(test_data)
