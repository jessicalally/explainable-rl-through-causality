class RLAgent:
    def __init__(self, env):
        pass

    def train(self):
        """ Trains the RL agent and generates training data for learning the
        action influence model, and for causal discovery. """
        pass

    def generate_test_data(self, num_datapoints):
        """ Generates random data from the trained RL agent for evaluation. """
        pass

    def get_q_func(self):
        """ Returns the learnt Q (state-action value) function for the trained
            RL agent. """
        pass
