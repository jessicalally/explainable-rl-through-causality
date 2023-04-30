from causal_discovery.environment import *
import evaluation
import numpy as np
from rl_algorithms.SARSA import SARSA
from structural_causal_model import StructuralCausalModel

# Choose which environment to use
env = Taxi()

# Choose which RL algorithm to use
rl_agent = SARSA(env)
action_influence_dataset, causal_discovery_dataset = rl_agent.train()

# Initialise action influence model
# action_influence_model = ActionInfluenceModel(
#     environment.causal_graph,
#     environment.action_matrix,
#     data_set
# )
# action_influence_model.train()

num_datapoints = 100000

data = rl_agent.generate_test_data(num_datapoints)
rnd_indices = np.random.choice(len(data), 25000)
data = data[rnd_indices]

print(f'Data: {data.shape}')

scm = StructuralCausalModel(
    env,
    data
)

scm.train()

# Evaluation
test_data = rl_agent.generate_test_data(num_datapoints)

# accuracy = evaluation.task_prediction(data, scm)
# print("Accuracy="+str(accuracy))

mse, action_predictions = evaluation.evaluate_fidelity(scm, test_data)
print("MSE=" + str(mse))
print("Correct action predictions=" + str(action_predictions))

# faithfulness = evaluation.evaluate_faithfulness(
#     test_data,
#     action_influence_model
# )

# TODO: processing explanations

# Causal graph discovery
# data = q_learning.generate_data_for_causal_discovery(environment.env, q_table)
# utils.convert_data_to_csv(data, "X.csv")
