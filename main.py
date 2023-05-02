from causal_discovery.environment import *
from causal_discovery.method import *
import causal_discovery
import evaluation
import numpy as np
from rl_algorithms.SARSA import SARSA
from structural_causal_model import StructuralCausalModel

# Choose which environment to use
env = Taxi()

# Choose which RL algorithm to use
rl_agent = SARSA(env)
action_influence_dataset, causal_discovery_dataset = rl_agent.train()
print(causal_discovery_dataset.shape)
print(causal_discovery_dataset)

# Perform causal discovery to generate causal graph
method = VarLiNGAM()

causal_matrix_with_assumptions = method.generate_causal_matrix(
    causal_discovery_dataset,
    env,
    with_assumptions=True
)

learned_causal_graph = causal_graph = nx.from_numpy_matrix(
            causal_matrix_with_assumptions, create_using=nx.MultiDiGraph())

# num_datapoints = 1000

# test_data = rl_agent.generate_test_data(num_datapoints)
# rnd_indices = np.random.choice(len(test_data), 1000)
# test_data = test_data[rnd_indices]

# TODO: test_data should be replaced by the action influence training data or randomly generated states and actions from the env
# TODO: especially as it looks like it might be overfitting
scm = StructuralCausalModel(
    env,
    causal_discovery_dataset,
    learned_causal_graph
)

scm.train()

# Evaluation
num_datapoints = 1000

test_data = rl_agent.generate_test_data(num_datapoints)
rnd_indices = np.random.choice(len(test_data), 1000)
test_data = test_data[rnd_indices]

print(f'Data: {test_data.shape}')

# accuracy = evaluation.task_prediction(data, scm)
# print("Accuracy="+str(accuracy))

mse, action_predictions = evaluation.evaluate_fidelity(scm, test_data)
print(causal_matrix_with_assumptions)
print("MSE=" + str(mse))
print("Correct action predictions=" + str(action_predictions))

# faithfulness = evaluation.evaluate_faithfulness(
#     test_data,
#     action_influence_model
# )

# Processing explanations
# why_explanations, why_not_explanations = scm.process_explanations(test_data)
# print(f'why explanations: {why_explanations}')
# print(f'why not explanations: {why_not_explanations}')
