from tensorflow.compat.v1.keras.layers import Dense, Input
from tensorflow.compat.v1.keras import Model
from tensorflow.compat.v1.keras.optimizers import Adam
import numpy as np

# This allows for compatibility with a Keras DQN implementation. We used the
# implementation here:
# [https://github.com/nitish-kalan/MountainCar-v0-Deep-Q-Learning-DQN-Keras]

class DQN:
    def __init__(self, action_space, state_space):
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
