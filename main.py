import argparse
import itertools
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

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Explanations')
    parser.add_argument(
        '--env',
        type=str,
        default='cartpole',
        help='RL environment')
    
    parser.add_argument(
        '--rl',
        type=str,
        default='ddqn',
        help='RL algorithm')

    args = parser.parse_args()

    return args


def get_environment(args):
    if args.env == "cartpole":
        return Cartpole()
    elif args.env == "lunarlander":
        return LunarLander()
    elif args.env == "mountaincar":
        return MountainCar()
    elif args.env == "taxi":
        return Taxi()
    else:
        raise ValueError(f"{args.env} environment is not implemented")


# TODO: refactor this so that RL hyperparameters are passed in from args
def get_rl_algorithm(args, env):
    if args.rl == "pg":
        return PolicyGradient(env)
    elif args.rl == "dqn":
        return DQN(env)
    elif args.rl == "ddqn":
        if args.env == "cartpole":
            return DDQN(
                env,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=0.99,
                batch_size=512,
                lr=0.01,
                reward_threshold=495
            )
        return DDQN(env)
    elif args.rl == "sarsa":
        return SARSA(env)
    else:
        raise ValueError(f"{args.rl} algorithm is not implemented")
    

# Generate causal matrix
def causal_discovery(dataset, env, forbidden_edges, required_edges, true_dag, restructure=False):
    print(f'Data: {dataset.shape}')

    method = VarLiNGAM()

    causal_matrix_with_assumptions = method.generate_causal_matrix(
        dataset,
        env,
        forbidden_edges,
        required_edges,
        with_assumptions=True,
        threshold=0.3, restructure=restructure)
    
    print(causal_matrix_with_assumptions)

    learned_causal_graph = nx.from_numpy_matrix(
        causal_matrix_with_assumptions, create_using=nx.MultiDiGraph())

    met = MetricsDAG(causal_matrix_with_assumptions, true_dag)
    print(met.metrics)

    return learned_causal_graph, met.metrics, causal_matrix_with_assumptions


def main(args):
    env = get_environment(args)
    rl_agent = get_rl_algorithm(args, env)

    rl_agent_path = f"output/trained_rl_agents/{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(rl_agent_path), exist_ok=True)

    # Train agent from scratch or load
    # rl_agent.train()

    with open(rl_agent_path,'rb') as agent_file:
        rl_agent = pickle.load(agent_file)
   
    ## Generate datasets ##
    # num_datapoints = 500000
    # causal_discovery_dataset, reward_causal_discovery_dataset = rl_agent.generate_test_data_for_causal_discovery(num_datapoints)

    dataset_path = f"output/causal_discovery_dataset/test/causal{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    reward_dataset_path = f"output/causal_discovery_dataset/test/reward{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(reward_dataset_path), exist_ok=True)

    # with open(rl_agent_path, 'wb') as agent_file:
    #     pickle.dump(rl_agent, agent_file)

    # with open(dataset_path, 'wb') as dataset_file:
    #     pickle.dump(causal_discovery_dataset, dataset_file)

    # with open(reward_dataset_path, 'wb') as dataset_file:
    #     pickle.dump(reward_causal_discovery_dataset, dataset_file)

    with open(dataset_path, 'rb') as dataset_file:
        causal_discovery_dataset = pickle.load(dataset_file)

    with open(reward_dataset_path, 'rb') as dataset_file:
        reward_causal_discovery_dataset = pickle.load(dataset_file)


    # Or load given datasets

    # with open(dataset_path, 'rb') as dataset_file:
    #     causal_discovery_dataset = pickle.load(dataset_file)

    # causal_discovery_dataset = np.loadtxt("output/causal_discovery_dataset/starcraft_a2c.csv", delimiter=',', encoding="utf_8_sig")

    ## Learn causal graph ##
    learned_causal_graph, met, causal_matrix_with_assumptions = causal_discovery(causal_discovery_dataset[:,:-env.state_space], env, env.forbidden_edges, env.required_edges, env.true_dag, restructure=True)
    
    forbidden_edges = [(i, j) for i, j in itertools.product(range(env.state_space), range(env.state_space))]
    required_edges = []

    for i,j in itertools.product(range(env.state_space), range(env.state_space)):
        if causal_matrix_with_assumptions[i][j] == 1:
            forbidden_edges.remove((i, j))
            required_edges.append((i, j))


    for i in range(env.state_space):
        forbidden_edges.append((env.state_space, i))
    
    # learned_reward_causal_graph, met, _ = causal_discovery(reward_causal_discovery_dataset, env, forbidden_edges, required_edges, env.reward_true_dag)

    method = PC()

    print(f'forbidden {forbidden_edges}')
    print(f'required {required_edges}')
    
    causal_matrix_with_assumptions = method.generate_causal_matrix(
        reward_causal_discovery_dataset,
        env,
        forbidden_edges,
        required_edges,
        with_assumptions=True)
    
    print(causal_matrix_with_assumptions)

    reward_met = MetricsDAG(causal_matrix_with_assumptions, env.reward_true_dag)

    learned_reward_causal_graph = nx.from_numpy_matrix(
        causal_matrix_with_assumptions, create_using=nx.MultiDiGraph())


    # metrics_path = f"output/metrics/{env.name}_{rl_agent.name}.pickle"
    # os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # causal_graph_path = f"output/causal_graph/{env.name}_{rl_agent.name}.pickle"
    # os.makedirs(os.path.dirname(causal_graph_path), exist_ok=True)

    # print(met)

    # with open(metrics_path, 'wb') as metrics_file:
    #     pickle.dump(met, metrics_file)

    # with open(causal_graph_path, 'wb') as graph_file:
    #     pickle.dump(learned_causal_graph, graph_file)

    # with open(metrics_path,'rb') as metrics_file:
    #     met = pickle.load(metrics_file)

    # with open(causal_graph_path, 'rb') as graph_file:
    #     learned_causal_graph = pickle.load(graph_file)


    ## Train structural causal model ##

    scm_dataset_path = f"output/scm_dataset/test/causal{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(scm_dataset_path), exist_ok=True)

    reward_scm_dataset_path = f"output/scm_dataset/test/reward{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(reward_scm_dataset_path), exist_ok=True)

    num_datapoints = 500000
    # TODO: these two functions are the same now
    # scm_dataset = rl_agent.generate_test_data_for_scm(num_datapoints)
    scm_dataset, reward_scm_dataset = rl_agent.generate_test_data_for_causal_discovery(num_datapoints, use_sum_rewards=True)

    with open(scm_dataset_path, 'wb') as dataset_file:
        pickle.dump(scm_dataset, dataset_file)

    with open(reward_scm_dataset_path, 'wb') as dataset_file:
        pickle.dump(reward_scm_dataset, dataset_file)

    # with open(scm_dataset_path, 'rb') as dataset_file:
    #     scm_dataset = pickle.load(dataset_file)

    # with open(reward_scm_dataset_path, 'rb') as dataset_file:
    #     reward_scm_dataset = pickle.load(dataset_file)

    scm = StructuralCausalModel(
        env,
        scm_dataset,
        learned_causal_graph
    )

    scm.train()

    reward_scm = StructuralCausalModel(
        env,
        reward_scm_dataset,
        learned_reward_causal_graph,
        for_reward = True,
    )

    reward_scm.train()

    # scm_path = f"output/scm/learned_dag/{env.name}_{rl_agent.name}.pickle"
    # os.makedirs(os.path.dirname(scm_path), exist_ok=True)

    # # with open(scm_path, 'wb') as scm_file:
    # #     pickle.dump(scm, scm_file)

    # with open(scm_path, 'rb') as scm_file:
        # scm = pickle.load(scm_file)

    ## Evaluation ##

    num_datapoints = 10000

    test_data, reward_test_data = rl_agent.generate_test_data_for_causal_discovery(num_datapoints, use_sum_rewards=True)
    print(test_data.shape)
    rnd_indices = np.random.choice(len(test_data), 2500)
    test_data = test_data[rnd_indices]

    print(f'Data: {test_data.shape}')

    # accuracy = evaluation.task_prediction(data, scm)
    # print("Accuracy="+str(accuracy))

    mse, action_predictions = evaluation.evaluate_fidelity(scm, test_data)
    print(met)
    print("MSE=" + str(mse))
    print("Correct action predictions=" + str(action_predictions))

    mse, action_predictions = evaluation.evaluate_fidelity(reward_scm, reward_test_data, REWARD_DAG=True)
    print(reward_met.metrics)
    print("MSE=" + str(mse))
    print("Correct action predictions=" + str(action_predictions))

    # with open(f"{env.name}_{rl_agent.name}_metrics.txt", 'w') as f:
    #     f.write("MSE=" + str(mse))
    #     f.write("MSE ignoring reward=" + str(mse_ignoring_reward))
    #     f.write("Correct action predictions=" + str(action_predictions))

    # ## Processing explanations ##
    # why_explanations = set()
    # why_not_explanations = set()

    # explanation_generator = ExplanationGenerator(env, scm, rl_agent)
    # for i in range(10):
    #     if env.name == "taxi":
    #         pertubation = 1.0
    #     else:
    #         pertubation = 0.01

    #     example_state = test_data[i][:env.state_space]
    #     example_action = test_data[i][env.state_space]
    #     example_counter_action = choice([i for i in range(env.action_space) if i != example_action])
    #     why_explanation = explanation_generator.generate_why_explanation(example_state, example_action, pertubation)
    #     why_explanations.add(why_explanation)
    #     print(f'Why {env.actions[example_action]}?\n {why_explanation}')

        # why_not_explanation = explanation_generator.generate_why_not_explanation(example_state, example_action, example_counter_action, pertubation)
        # print(f'Why not {env.actions[example_counter_action]}?\n {why_not_explanation}')
        # why_not_explanations.add(why_not_explanation)
    
    # with open(f"{env.name}_{rl_agent.name}_explanations.txt", 'w') as f:
    #     f.write(str(why_explanations))
    #     f.write(str(why_not_explanations))

if __name__ == '__main__':
    args = parse_args()
    main(args)
