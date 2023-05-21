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

    class MemoryBuffer(object):
        def __init__(self, max_size):
            self.memory_size = max_size
            self.trans_counter = 0  # num of transitions in the memory
            # this count is required to delay learning
            # until the buffer is sensibly full
            self.index = 0         # current pointer in the buffer
            self.buffer = deque(maxlen=self.memory_size)
            self.transition = namedtuple(
                "Transition",
                field_names=[
                    "state",
                    "action",
                    "reward",
                    "new_state",
                    "terminal"])

        def save(self, state, action, reward, new_state, terminal):
            t = self.transition(state, action, reward, new_state, terminal)
            self.buffer.append(t)
            self.trans_counter = (self.trans_counter + 1) % self.memory_size

        def random_sample(self, batch_size):
            # should begin sampling only when sufficiently full
            assert len(self.buffer) >= batch_size
            # number of transitions to sample
            transitions = random.sample(self.buffer, k=batch_size)
            states = torch.from_numpy(np.vstack(
                [e.state for e in transitions if e is not None])).float()
            actions = torch.from_numpy(np.vstack(
                [e.action for e in transitions if e is not None])).long()
            rewards = torch.from_numpy(np.vstack(
                [e.reward for e in transitions if e is not None])).float()
            new_states = torch.from_numpy(np.vstack(
                [e.new_state for e in transitions if e is not None])).float()
            terminals = torch.from_numpy(np.vstack(
                [e.terminal for e in transitions if e is not None]).astype(np.uint8)).float()

            return states, actions, rewards, new_states, terminals

    class QNN(nn.Module):
        def __init__(self, state_space, action_space):
            super().__init__()
            self.fc1 = nn.Linear(state_space, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, action_space)

        def forward(self, state):
            x = self.fc1(state)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            return self.fc3(x)

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
        self.replay_buffer = self.MemoryBuffer(self.buffer_size)
        self.replace_q_target = 100
        self.q_func = self.QNN(self.state_space, self.action_space)
        self.q_func_target = self.QNN(self.state_space, self.action_space)
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=self.lr)

    def _save(self, state, action, reward, new_state, done):
        # self.memory.trans_counter += 1
        self.replay_buffer.save(state, action, reward, new_state, done)

    def _choose_action(self, state):
        # state = state[np.newaxis, :]
        rand = np.random.random()
        state = torch.DoubleTensor(state).unsqueeze(0)
        self.q_func.eval()
        action_values = self.q_func(state)
        self.q_func.train()
        # print(state)
        if rand > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # exploring: return a random action
            return np.random.choice([i for i in range(self.action_space)])

    def _choose_action_deterministic(self, state):
        state = torch.DoubleTensor(state).unsqueeze(0)

        self.q_func.eval()
        action_values = self.q_func(state)

        return np.argmax(action_values.cpu().data.numpy())

    def _reduce_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > \
            self.min_epsilon else self.min_epsilon

    def _learn(self):
        if self.replay_buffer.trans_counter < self.batch_size:  # wait before you start learning
            return

        # 1. Choose a sample from past transitions:
        states, actions, rewards, new_states, terminals = self.replay_buffer.random_sample(
            self.batch_size)

        # 2. Update the target values
        q_next = self.q_func_target(new_states.to(
            torch.float64)).detach().max(1)[0].unsqueeze(1)
        q_updated = rewards + self.gamma * q_next * (1 - terminals)
        q = self.q_func(states.to(torch.float64)).gather(1, actions)

        # 3. Update the main NN
        loss = F.mse_loss(q, q_updated)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Update the target NN (every N-th step)
        if self.replay_buffer.trans_counter % self.replace_q_target == 0:  # wait before you start learning
            for target_param, local_param in zip(
                    self.q_func_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(local_param.data)

        # 5. Reduce the exploration rate
        self._reduce_epsilon()

    def _save_model(self, path):
        torch.save(self.q_func.state_dict(), path)
        torch.save(self.q_func.state_dict(), path + '.target')

    def _load_saved_model(self, path):
        self.q_func = self.QNN(8, 4, 42)
        self.q_func.load_state_dict(torch.load(path))
        self.q_func.eval()

        self.q_func_target = self.QNN(8, 4, 42)
        self.q_func_target.load_state_dict(torch.load(path + '.target'))
        self.q_func_target.eval()

    def train(self, episodes=2000):
        print('Performing DDQN algorithm...')
        reward_test_data = []
        transition_test_data = []

        LEARN_EVERY = 4
        scores = []
        eps_history = []

        for e in range(episodes):
            terminated = False
            truncated = False
            score = 0
            state, _ = self.env.reset()
            steps = 0

            while not (terminated or truncated):
                action = self._choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                self._save(state, action, reward, next_state, terminated)

                state = next_state

                if steps > 0 and steps % LEARN_EVERY == 0:
                    self._learn()

                if terminated:
                    reward = 0

                    reward_test_data.append(
                        np.concatenate((next_state, np.array(reward)), axis=None)
                    )
                else:
                    reward_test_data.append(
                        np.concatenate((next_state, np.array(reward)), axis=None)
                        )

                    transition_test_data.append(
                        np.concatenate(
                            (state, np.array(action), next_state),
                            axis=None))

                steps += 1
                score += reward

            eps_history.append(self.epsilon)
            scores.append(score)

            avg_score = np.mean(scores[max(0, e - 100):(e + 1)])
            print("episode: {}/{}, score: {}".format(e, episodes, avg_score))

            if avg_score > self.reward_threshold:
                break

        print('Finished DDQN Algorithm...')

        return np.array(transition_test_data), np.array(reward_test_data)


    def generate_test_data_for_causal_discovery(self, num_datapoints, use_sum_rewards=False):
        transition_test_data = []
        reward_test_data = []

        print('Generating test data for DDQN algorithm...')

        while len(transition_test_data) < num_datapoints:
            terminated = False
            truncated = False
            state, _ = self.env.reset()
            episode_rewards = 0

            while not (terminated or truncated):
                action = self._choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)

                # TODO: do we want this for just cartpole or for lunar lander as well???
                if use_sum_rewards:
                    if terminated:
                        reward = 0

                    reward_test_data.append(
                        np.concatenate((next_state, np.array(reward)), axis=None)
                    )
                else:
                    reward_test_data.append(
                        np.concatenate((next_state, np.array(reward)), axis=None)
                    )

                transition_test_data.append(
                    np.concatenate(
                        (state, np.array(action), next_state),
                        axis=None))
            

                state = next_state

            print("num datapoints collected so far: {}".format(len(transition_test_data)))

        print('Finished generating test data for DDQN Algorithm...')

        return np.array(transition_test_data), np.array(reward_test_data)

    # TODO: adjust rewards in same way as above
    def generate_test_data_for_scm(self, num_datapoints):
        transition_scm_test_data = []
        reward_scm_test_data = []

        print('Generating test data for DDQN algorithm...')

        while len(transition_scm_test_data) < num_datapoints:
            terminated = False
            truncated = False
            state, _ = self.env.reset()

            episode_rewards = 0

            while not (terminated or truncated):
                action = self._choose_action_deterministic(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)

                if terminated:
                    episode_rewards = 0
                else:
                    episode_rewards += reward

                transition_scm_test_data.append(
                    np.concatenate(
                        (state, np.array(action), next_state),
                        axis=None))
                
                reward_scm_test_data.append(
                    np.concatenate((next_state, np.array(episode_rewards)), axis=None)
                )

                state = next_state

            print("num datapoints collected so far: {}".format(len(transition_scm_test_data)))

        print('Finished generating test data for DDQN Algorithm...')

        return np.array(transition_scm_test_data), np.array(reward_scm_test_data)

    # Methods needed for estimating feature importance

    def get_q_values(self, state):
        q_func = self.q_func.eval()

        state_tensor = torch.DoubleTensor(state).unsqueeze(0)
        q_values = q_func(state_tensor).cpu().data.numpy()[0]

        return q_values

    def get_optimal_action(self, state):
        return self._choose_action_deterministic(state)
