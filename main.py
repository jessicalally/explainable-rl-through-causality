from castle.metrics import MetricsDAG
from causal_discovery.environment import *
from causal_discovery.method import *
from explanation_generation import ExplanationGenerator
import evaluation
import os
import numpy as np
import dill as pickle
from random import choice
from rl_algorithms.SARSA import SARSA
from rl_algorithms.policy_gradient import PolicyGradient
from rl_algorithms.DQN import DQN
from rl_algorithms.DDQN import DDQN
from structural_causal_model import StructuralCausalModel

# Choose which environment to use
env = Cartpole()
# env = LunarLander()

# Choose which RL algorithm to use
# rl_agent = PolicyGradient(env)
rl_agent = DDQN(
    env,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.99,
    batch_size=512,
    lr=0.01)
# rl_agent = DDQN(env)

# Train agent from scratch
action_influence_dataset, causal_discovery_dataset = rl_agent.train(
    reward_threshold=495)
# action_influence_dataset, causal_discovery_dataset = rl_agent.train()
print(len(causal_discovery_dataset))

rl_agent_path = f"trained_rl_agents/{env.name}_{rl_agent.name}.pickle"
os.makedirs(os.path.dirname(rl_agent_path), exist_ok=True)

dataset_path = f"causal_discovery_dataset_with_reward/{env.name}_{rl_agent.name}.pickle"
os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

# with open(rl_agent_path, 'wb') as agent_file:
#     pickle.dump(rl_agent, agent_file)

# with open(dataset_path, 'wb') as dataset_file:
#     pickle.dump(causal_discovery_dataset, dataset_file)

# # OR load from file
# with open(rl_agent_path,'rb') as agent_file:
#     rl_agent = pickle.load(agent_file)

# with open(dataset_path, 'rb') as dataset_file:
#     causal_discovery_dataset = pickle.load(dataset_file)

# Perform causal discovery to generate causal graph

num_datapoints = 10000

causal_discovery_dataset = rl_agent.generate_test_data_for_causal_discovery(
    num_datapoints)
# rnd_indices = np.random.choice(len(test_data), 2500)
# test_data = test_data[rnd_indices]

print(f'Data: {causal_discovery_dataset.shape}')

method = VarLiNGAM()

causal_matrix_with_assumptions = method.generate_causal_matrix(
    causal_discovery_dataset,
    env,
    with_assumptions=True)

print(f'causal matrix\n {causal_matrix_with_assumptions}')

learned_causal_graph = nx.from_numpy_matrix(
    causal_matrix_with_assumptions, create_using=nx.MultiDiGraph())

met = MetricsDAG(causal_matrix_with_assumptions, env.true_dag)


# Train structural causal model
num_datapoints = 10000
scm_dataset = rl_agent.generate_test_data_for_scm(num_datapoints)

scm = StructuralCausalModel(
    env,
    scm_dataset,
    learned_causal_graph
)

scm.train()

# Evaluation
num_datapoints = 10000

test_data = rl_agent.generate_test_data_for_scm(num_datapoints)
rnd_indices = np.random.choice(len(test_data), 2500)
test_data = test_data[rnd_indices]

print(f'Data: {test_data.shape}')

# accuracy = evaluation.task_prediction(data, scm)
# print("Accuracy="+str(accuracy))

mse, action_predictions = evaluation.evaluate_fidelity(scm, test_data)
print(causal_matrix_with_assumptions)
print(met.metrics)
print("MSE=" + str(mse))
print("Correct action predictions=" + str(action_predictions))

# Processing explanations
# explanation_generator = ExplanationGenerator(env, scm, rl_agent)
# for i in range(10):
#     example_state = test_data[i][:env.state_space]
#     example_action = test_data[i][env.state_space]
#     example_counter_action = choice([i for i in range(env.action_space) if i != example_action])
#     why_explanation = explanation_generator.generate_why_explanation(example_state, example_action)
#     print(f'Why {env.actions[example_action]}?\n {why_explanation}')

#     why_not_explanation = explanation_generator.generate_why_not_explanation(example_state, example_action, example_counter_action)
#     print(f'Why not {env.actions[example_counter_action]}?\n {why_not_explanation}')
