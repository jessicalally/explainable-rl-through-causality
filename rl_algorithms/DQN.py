from tensorflow.compat.v1.keras.layers import Dense, Input
from tensorflow.compat.v1.keras import Model
from tensorflow.compat.v1.keras.optimizers import Adam
import numpy as np
import random

# This allows pretrained DQN models to be loaded.

class DQN:
    def __init__(self, environment, action_space, state_space):
        self.env = environment.env
        self.state_space = state_space
        self.action_space = action_space
        self.model = self.create_model()
        self.name = "dqn"

    def create_model(self):
        input_layer = Input(shape=(self.state_space))
        layer1 = Dense(400, activation='relu')(input_layer)
        layer2 = Dense(300, activation='relu')(layer1)
        output_layer = Dense(self.action_space, activation='linear')(layer2)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer=Adam(0.005))

        return model

    def get_q_values(self, state):
        return self.model.predict(np.expand_dims(state, axis=0))
    
    def _choose_action_deterministic(self, state):
        return np.argmax(self.model.predict(np.expand_dims(state, axis=0))[0])
    
    def generate_test_data_for_causal_discovery(
            self, num_datapoints, use_sum_rewards=False):
        
        transition_test_data = []
        reward_test_data = []
        num_collected = 0

        print('Generating test data for DQN algorithm...')

        while num_collected < num_datapoints:
            transition_test_data = []
            reward_test_data = []

            terminated = False
            truncated = False
            state, _ = self.env.reset()

            while not (terminated or truncated):
                action = self._choose_action_deterministic(state)
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
            
            with open('temp-transitions.txt','ba') as f:
                np.savetxt(f, transition_test_data)

            with open('temp-rewards.txt','ba') as f:
                np.savetxt(f, reward_test_data)
            
            num_collected += len(transition_test_data)

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

            with open('temp-transitions-random.txt','ba') as f:
                np.savetxt(f, transition_test_data)

            with open('temp-rewards-random.txt','ba') as f:
                np.savetxt(f, reward_test_data)
        print('Finished generating test data for DDQN Algorithm...')

        return np.array(transition_test_data), np.array(reward_test_data)
