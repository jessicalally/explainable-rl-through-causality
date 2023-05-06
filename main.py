from causal_discovery.environment import *
from causal_discovery.method import *
import causal_discovery
import evaluation
import os
import numpy as np
import dill as pickle
from rl_algorithms.SARSA import SARSA
from rl_algorithms.policy_gradient import PolicyGradient
from structural_causal_model import StructuralCausalModel

# Choose which environment to use
env = Cartpole()

# Choose which RL algorithm to use
rl_agent = PolicyGradient(env)

# Train agent from scratch
# action_influence_dataset, causal_discovery_dataset = rl_agent.train()
# print(causal_discovery_dataset.shape)
# print(causal_discovery_dataset)

rl_agent_path = "trained_rl_agents/pg.pickle"
# os.makedirs(os.path.dirname(rl_agent_path), exist_ok=True)

dataset_path = "causal_discovery_dataset/cartpole_pg.pickle"
# os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

# with open(rl_agent_path, 'wb') as agent_file:
#     pickle.dump(rl_agent,agent_file)

# with open(dataset_path, 'wb') as dataset_file:
#     pickle.dump(causal_discovery_dataset, dataset_file)

# OR load from file
with open(rl_agent_path,'rb') as agent_file:
    rl_agent = pickle.load(agent_file)

with open(dataset_path, 'rb') as dataset_file:
    causal_discovery_dataset = pickle.load(dataset_file)

# Perform causal discovery to generate causal graph
method = VarLiNGAM()

causal_matrix_with_assumptions = method.generate_causal_matrix(
    causal_discovery_dataset,
    env,
    with_assumptions=True
)

learned_causal_graph = nx.from_numpy_matrix(
            causal_matrix_with_assumptions, create_using=nx.MultiDiGraph())

# Train structural causal model
scm = StructuralCausalModel(
    env,
    causal_discovery_dataset,
    learned_causal_graph
)

scm.train()

# Evaluation
num_datapoints = 10000

test_data = rl_agent.generate_test_data(num_datapoints)
rnd_indices = np.random.choice(len(test_data), 2500)
test_data = test_data[rnd_indices]

print(f'Data: {test_data.shape}')

# accuracy = evaluation.task_prediction(data, scm)
# print("Accuracy="+str(accuracy))

mse, action_predictions = evaluation.evaluate_fidelity(scm, test_data)
print(causal_matrix_with_assumptions)
print("MSE=" + str(mse))
print("Correct action predictions=" + str(action_predictions))

# Processing explanations
why_explanations, why_not_explanations = scm.process_explanations(test_data)
print(f'why explanations: {why_explanations}')
# print(f'why not explanations: {why_not_explanations}')
