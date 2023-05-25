from collections import defaultdict
import copy
import numpy as np
import random
from .rl_agent import RLAgent

# Tuned to Taxi environment, as in:
# [Explainable Reinforcement Learning Through a Causal Lens]
# https://arxiv.org/abs/1905.10958


class SARSA(RLAgent):
    def __init__(self, environment):
        self.name = "sarsa"

        # Environment
        self.env = environment.env
        self.test_env = copy.deepcopy(self.env)
        self.action_space = environment.action_space
        self.state_space = environment.state_space

        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.01
        self.alpha = 0.01
        self.gamma = 1

        self.Q = defaultdict(lambda: np.zeros(self.action_space))

    def _choose_action_from_probs(self, action_probs):
        if np.random.rand(1) > self.epsilon:
            return np.argmax(action_probs)

        return random.randrange(self.action_space)

    def _generate_epsilon_greedy_policy(self):
        def policy_fn(state):
            policy = np.ones(self.action_space) * \
                self.epsilon / self.action_space
            best_action = np.argmax(self.Q[state])
            policy[best_action] += 1.0 - self.epsilon

            return policy

        return policy_fn

    def _generate_deterministic_policy(self):
        def policy_fn(state):
            policy = np.zeros(self.action_space)
            best_action = np.argmax(self.Q[state])
            policy[best_action] = 1.0

            return policy

        return policy_fn

    def train(self, episodes=100000, reward_threshold=10):
        test_data = []
        reward_test_data = []

        print('Performing SARSA algorithm...')

        results = []
        policy = self._generate_epsilon_greedy_policy()
        max_steps = 50

        for e in range(episodes):
            state, _ = self.env.reset()
            action_probs = policy(state)
            action = self._choose_action_from_probs(action_probs)
            total_reward = 0

            for _ in range(max_steps):
                next_state, reward, done, _, _ = self.env.step(action)
                next_action_probs = policy(next_state)
                next_action = self._choose_action_from_probs(next_action_probs)

                # Taxi environment requires decoding the state into the separate
                # state variables
                taxi_row, taxi_col, pass_loc, dest_idx = self.env.decode(state)
                decoded_state = [taxi_row, taxi_col, pass_loc, dest_idx]

                taxi_row, taxi_col, pass_loc, dest_idx = self.env.decode(
                    next_state)
                decoded_next_state = [taxi_row, taxi_col, pass_loc, dest_idx]

                test_data.append(
                    np.concatenate(
                        (decoded_state,
                        np.array(action),
                        decoded_next_state),
                        axis=None))
                                
                reward_test_data.append(np.concatenate((decoded_next_state, np.array(reward)), axis=None))

                td_target = reward + self.gamma * \
                    self.Q[next_state][next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta

                total_reward += reward

                if done:
                    break

                action = next_action
                state = next_state

            print("episode: {}/{}, score: {}".format(e, episodes, total_reward))

            results.append(total_reward)

            if np.mean(results[-100:]) > reward_threshold:
                print('\n Task Completed! \n')
                break

            self.epsilon = max(
                self.epsilon *
                self.epsilon_decay,
                self.min_epsilon)

        print('Finished SARSA Algorithm...')

        return np.array(test_data), np.array(reward_test_data)

    # Generates datapoints from the trained RL agent
    def generate_test_data_for_causal_discovery(self, num_datapoints):
        test_data = []
        reward_discovery_test_data = []

        policy = self._generate_deterministic_policy()
        episode = 0

        print("Generating test data...")

        while len(test_data) < num_datapoints:
            state, _ = self.env.reset()
            action_probs = policy(state)
            action = self._choose_action_from_probs(action_probs)
            done = False
            total_reward = 0

            while not done and len(test_data) < num_datapoints:
                next_state, reward, done, _, _ = self.env.step(action)
                next_action_probs = policy(next_state)
                next_action = self._choose_action_from_probs(next_action_probs)

                # Taxi environment requires decoding the state into the separate
                # state variables
                taxi_row, taxi_col, pass_loc, dest_idx = self.env.decode(state)
                decoded_state = [taxi_row, taxi_col, pass_loc, dest_idx]

                taxi_row, taxi_col, pass_loc, dest_idx = self.env.decode(
                    next_state)
                decoded_next_state = [taxi_row, taxi_col, pass_loc, dest_idx]

                test_data.append(
                    np.concatenate(
                        (decoded_state,
                        np.array(action),
                        decoded_next_state),
                        axis=None))
                
                reward_discovery_test_data.append(np.concatenate((decoded_next_state, np.array(reward)), axis=None))

                total_reward += reward
                action = next_action
                state = next_state

            print("episode: {}, score: {}".format(episode, total_reward))
            print("num datapoints collected so far: {}".format(len(test_data)))
            episode += 1

        print("Finished generating test data...")

        return np.array(test_data), np.array(reward_discovery_test_data)
    

    # Generates datapoints from the trained RL agent
    def generate_test_data_for_scm(self, num_datapoints):
        test_data = []
        reward_scm_test_data = []
        
        policy = self._generate_deterministic_policy()
        episode = 0

        print("Generating test data...")

        while len(test_data) < num_datapoints:
            state, _ = self.env.reset()
            action_probs = policy(state)
            action = self._choose_action_from_probs(action_probs)
            done = False
            total_reward = 0

            while not done and len(test_data) < num_datapoints:
                next_state, reward, done, _, _ = self.env.step(action)
                next_action_probs = policy(next_state)
                next_action = self._choose_action_from_probs(next_action_probs)

                # Taxi environment requires decoding the state into the separate
                # state variables
                taxi_row, taxi_col, pass_loc, dest_idx = self.env.decode(state)
                decoded_state = (taxi_row, taxi_col, pass_loc, dest_idx)

                taxi_row, taxi_col, pass_loc, dest_idx = self.env.decode(
                    next_state)
                decoded_next_state = (taxi_row, taxi_col, pass_loc, dest_idx)

                test_data.append(
                    np.concatenate(
                        (decoded_state,
                        np.array(action),
                        decoded_next_state),
                        axis=None))
                
                reward_scm_test_data.append(np.concatenate((decoded_next_state, np.array(reward)), axis=None))

                total_reward += reward
                action = next_action
                state = next_state

            print("episode: {}, score: {}".format(episode, total_reward))
            print("num datapoints collected so far: {}".format(len(test_data)))
            episode += 1

        print("Finished generating test data...")

        return np.array(test_data)


    # Methods needed for estimating feature importance

    def get_q_values(self, state):
        return self.Q[state]

    def get_optimal_action(self, state):
        policy = self._generate_deterministic_policy()
        next_action_probs = policy(state)

        return self._choose_action_from_probs(next_action_probs)
