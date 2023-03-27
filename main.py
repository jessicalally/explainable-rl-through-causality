from action_influence_model import ActionInfluenceModel
from environments.cartpole import Cartpole
import evaluation
from rl_algorithms.q_learning import QLearning
import gym
import utils

# Choose which environment to use
environment = Cartpole()

# Choose which RL algorithm to use 
q_learning = QLearning(environment.env, environment.state_space, environment.action_space)
q_table, data_set, bins = q_learning.train()

# Initialise action influence model
action_influence_model = ActionInfluenceModel(
    environment.causal_graph,
    environment.action_matrix,
    data_set
)
action_influence_model.train()

# Evaluation
# test_data = q_learning.generate_test_data(environment.env, q_table)
# accuracy = evaluation.task_prediction(test_data, action_influence_model)
# print("Accuracy="+str(accuracy))
# fidelity = evaluation.evaluate_fidelity(test_data, action_influence_model)
# print("Fidelity="+str(fidelity))
# faithfulness = evaluation.evaluate_faithfulness(
#     test_data, 
#     action_influence_model
# )

# TODO: processing explanations

# Causal graph discovery
data = q_learning.generate_data_for_causal_discovery(environment.env, q_table)
utils.convert_data_to_csv(data, "X.csv")
