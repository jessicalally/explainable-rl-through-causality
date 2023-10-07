from collections import deque
import copy
import numpy as np
import random
from .rl_agent import RLAgent

import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.layers import Dense, Input
from tensorflow.compat.v1.keras import Model
from tensorflow.compat.v1.keras.optimizers import Adam

# Tuned to Mountain Car environment, as in:
# [Explainable Reinforcement Learning Through a Causal Lens]
# https://arxiv.org/abs/1905.10958

class DQN(RLAgent):
    class Agent():
        def __init__(self, session, environment):
            K.set_session(session)
            self.action_space = environment.action_space
            self.state_space = environment.state_space
            self.model = self._init_model()

        def _init_model(self):
            input_layer = Input(shape=(self.state_space))
            layer1 = Dense(400, activation='relu')(input_layer)
            layer2 = Dense(300, activation='relu')(layer1)
            output_layer = Dense(self.action_space, activation='linear')(layer2)
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(loss='mse', optimizer=Adam(0.005))
            return model
        
        def get_q_values(self, state):
            return self.model.predict(np.expand_dims(state, axis=0))

    def __init__(
            self,
            environment,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.99,
            batch_size=32,
            reward_threshold=-110):
        self.name = "dqn"

        # Environment
        self.environment = environment
        self.env = environment.env
        self.test_env = copy.deepcopy(self.env)
        self.action_space = environment.action_space
        self.state_space = environment.state_space
        self.reward_threshold = reward_threshold

        # Agent
        self.session = tf.Session()
        self.agent = self.Agent(self.session, self.environment)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.min_epsilon = 0.001
        self.buffer_size = 100000
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.min_replay_buffer_size = 1000
        self.reward_threshold = reward_threshold

    def _choose_action_epsilon_greedy(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.agent.model.predict(np.expand_dims(state, axis=0))[0])
        else:
            return random.randint(0, self.action_space - 1)
        
    def _train_agent(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = []
        next_states = []

        for sample in batch:
            state, action, reward, next_state, done = sample
            states.append(state)
            next_states.append(next_state)
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        action_predictions = self.agent.model.predict(states)
        next_action_predictions = self.agent.model.predict(next_states)
        
        for index, sample in enumerate(batch):
            state, action, reward, next_state, done = sample

            if not done:
                action_predictions[index][action] = reward + self.gamma * np.amax(next_action_predictions[index])
            else:
                action_predictions[index][action] = reward

        self.agent.model.fit(states, action_predictions, verbose=0)

    def train(self, episodes=1000):
        print('Performing DQN algorithm...')
        reward_test_data = []
        transition_test_data = []

        scores = []
        eps_history = []
        max_score = -99999

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
                
                score += reward

                if (terminated or truncated) and e < 200:
                    # Increase reward on win to encourage convergence, as DQN
                    # is an unstable algorithm
                    reward = 250 + score
                else:
                    # Reward is proportional to the position of the car
                    reward = 5*abs(next_state[0] - state[0]) + 3*abs(state[1])

                self.replay_buffer.append((
                    state, action, reward, next_state, terminated or truncated))

                state = next_state

                reward_test_data.append(np.concatenate(
                    (next_state, np.array(reward)), axis=None))

                transition_test_data.append(
                    np.concatenate(
                        (state, np.array(action), next_state),
                        axis=None))

                if (len(self.replay_buffer) >= self.min_replay_buffer_size):
                    self._train_agent()
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            
            eps_history.append(self.epsilon)
            scores.append(score)

            if (score > max_score):
                self.agent.model.save_weights(str(score)+"_agent_.h5")

            max_score = max(max_score, score)

            avg_score = np.mean(scores[max(0, e - 10):(e + 1)])
            print("episode: {}/{}, score: {}, avg score: {}".format(e,
                  episodes, score, avg_score))

            if avg_score > self.reward_threshold:
                self.agent.model.save_weights("solved_agent.h5")
                break

        print('Finished DQN Algorithm...')

        return np.array(transition_test_data), np.array(reward_test_data)

    def generate_test_data_for_causal_discovery(
            self, num_datapoints, use_sum_rewards=False):
        
        transition_test_data = []
        reward_test_data = []

        print('Generating test data for DQN algorithm...')

        while len(transition_test_data) < num_datapoints:
            terminated = False
            truncated = False
            state, _ = self.env.reset()

            while not (terminated or truncated):
                action = self._choose_action_epsilon_greedy(state)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)

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

        print('Finished generating test data for DQN Algorithm...')

        return np.array(transition_test_data), np.array(reward_test_data)
    
    def generate_test_data_for_scm_training(
            self, num_datapoints, use_sum_rewards=False):
        transition_test_data = []
        reward_test_data = []

        print('Generating test data for DQN algorithm...')

        while len(transition_test_data) < num_datapoints:
            terminated = False
            truncated = False
            state, _ = self.env.reset()

            while not (terminated or truncated):
                action = random.randint(0, self.action_space - 1)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)

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
