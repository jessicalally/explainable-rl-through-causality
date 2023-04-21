import numpy as np
from .rl_agent import RLAgent


class QLearning(RLAgent):
    def __init__(self, env, state_space, action_space, bin_size=30):
        self.env = env
        self.q_table = np.random.uniform(
            low=-1,
            high=1,
            size=([bin_size] * state_space + [action_space])
        )

        self.bin_size = bin_size

        # TODO: these bins are for cartpole so will have to change
        self.bins = [
            np.linspace(-4.8, 4.8, bin_size),
            np.linspace(-4, 4, bin_size),
            np.linspace(-0.418, 0.418, bin_size),
            np.linspace(-4, 4, bin_size)
        ]

    def Discrete(self, state, bins):
        index = []

        for i in range(len(state)):
            index.append(np.digitize(state[i], bins[i]) - 1)
        return tuple(index)

    def train(
            self,
            episodes=2000,
            gamma=0.95,
            lr=0.1,
            timestep=100,
            epsilon=0.2):
        data_set = []

        print('Performing Q-learning...')
        rewards = 0
        steps = 0

        for episode in range(episodes):
            steps += 1
            current_observation, _ = self.env.reset()
            current_state = self.Discrete(current_observation, self.bins)

            score = 0
            done = False

            while not done:
                # if episode % timestep == 0:
                #     env.render()

                if np.random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[current_state])

                next_observation, reward, done, _, _ = self.env.step(action)
                next_state = self.Discrete(next_observation, self.bins)
                score += 1

                if not done:
                    max_future_q = np.max(self.q_table[next_state])
                    current_q = self.q_table[current_state + (action,)]
                    new_q = (1 - lr) * current_q + lr * \
                        (reward + gamma * max_future_q)
                    self.q_table[current_state + (action,)] = new_q

                data_set.append(
                    (current_observation, action, reward, next_observation))
                current_state = next_state
                current_observation = next_observation

            # End of the loop update
            else:
                rewards += score
                if score > 300:
                    print('Solved')

            if episode % timestep == 0:
                print(reward / timestep)

        print('Finished Q-learning...')

        return self.q_table, data_set, self.bins

    # TODO: we want this test data to be more random

    def generate_test_data(self, env, q_table, episodes=10):
        test_data = []

        print('Generating Q-learning test data...')
        steps = 0

        for _ in range(episodes):
            steps += 1
            # env.reset() => initial observation
            current_observation, _ = self.env.reset()
            current_state = self.Discrete(current_observation, self.bins)

            done = False

            while not done:
                # Pick best action from trained q table
                action = np.argmax(self.q_table[current_state])

                next_observation, _, done, _, _ = self.env.step(action)
                next_state = self.Discrete(next_observation, self.bins)

                test_data.append(
                    (current_observation, action, next_observation))

                # Update state
                current_state = next_state
                current_observation = next_observation

        print('Finished generating Q-learning test data...')

        return test_data

    def generate_data_for_causal_discovery(self, env, q_table, episodes=1000):
        data = []

        print('Generating Q-learning data for causal discovery...')
        steps = 0

        for _ in range(episodes):
            steps += 1
            # env.reset() => initial observation
            current_observation, _ = self.env.reset()
            current_state = self.Discrete(current_observation, self.bins)

            done = False

            while not done:
                # Pick best action from trained q table
                action = np.argmax(self.q_table[current_state])

                next_observation, _, done, _, _ = self.env.step(action)
                next_state = self.Discrete(next_observation, self.bins)

                datapoint = np.concatenate(
                    (current_observation, np.array(action), next_observation), axis=None)
                data.append(datapoint)

                # Update state
                current_state = next_state
                current_observation = next_observation

        print('Finished generating Q-learning data for causal discovery...')

        return data
