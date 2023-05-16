import argparse
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
def causal_discovery(dataset, env):
    print(f'Data: {dataset.shape}')

    method = VarLiNGAM()

    causal_matrix_with_assumptions = method.generate_causal_matrix(
        dataset,
        env,
        with_assumptions=True,
        threshold=0.3)

    learned_causal_graph = nx.from_numpy_matrix(
        causal_matrix_with_assumptions, create_using=nx.MultiDiGraph())

    met = MetricsDAG(causal_matrix_with_assumptions, env.true_dag)

    return learned_causal_graph, met


def main(args):
    env = get_environment(args)
    rl_agent = get_rl_algorithm(args, env)

    # Train agent from scratch
    # TODO: remove returned datasets since these are wrong anyway
    _, _ = rl_agent.train()

    rl_agent_path = f"output/trained_rl_agents/{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(rl_agent_path), exist_ok=True)

    dataset_path = f"output/causal_discovery_dataset/{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    ## Generate datasets ##

    num_datapoints = 500000
    causal_discovery_dataset = rl_agent.generate_test_data_for_causal_discovery(num_datapoints)

    with open(rl_agent_path, 'wb') as agent_file:
        pickle.dump(rl_agent, agent_file)

    with open(dataset_path, 'wb') as dataset_file:
        pickle.dump(causal_discovery_dataset, dataset_file)

    # Or load given datasets

    # with open(rl_agent_path,'rb') as agent_file:
    #     rl_agent = pickle.load(agent_file)

    # with open(dataset_path, 'rb') as dataset_file:
    #     causal_discovery_dataset = pickle.load(dataset_file)

    ## Learn causal graph ##

    learned_causal_graph, met = causal_discovery(causal_discovery_dataset, env)

    metrics_path = f"output/metrics/{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    causal_graph_path = f"output/causal_graph/{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(causal_graph_path), exist_ok=True)

    with open(metrics_path, 'wb') as metrics_file:
        pickle.dump(met, metrics_file)

    with open(causal_graph_path, 'wb') as graph_file:
        pickle.dump(learned_causal_graph, graph_file)

    ## Train structural causal model ##

    scm_dataset_path = f"scm_dataset/{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(scm_dataset_path), exist_ok=True)

    num_datapoints = 500000
    scm_dataset = rl_agent.generate_test_data_for_scm(num_datapoints)

    with open(scm_dataset_path, 'wb') as dataset_file:
        pickle.dump(scm_dataset, dataset_file)

    # with open(scm_dataset_path, 'rb') as dataset_file:
    #     scm_dataset = pickle.load(dataset_file)

    scm = StructuralCausalModel(
        env,
        scm_dataset,
        learned_causal_graph
    )

    scm.train()

    scm_path = f"output/scm/{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(scm_path), exist_ok=True)

    with open(scm_path, 'wb') as scm_file:
        pickle.dump(scm, scm_file)

    ## Evaluation ##

    # num_datapoints = 10000

    # test_data = rl_agent.generate_test_data_for_scm(num_datapoints)
    # print(test_data.shape)
    # rnd_indices = np.random.choice(len(test_data), 2500)
    # test_data = test_data[rnd_indices]

    # print(f'Data: {test_data.shape}')

    # accuracy = evaluation.task_prediction(data, scm)
    # print("Accuracy="+str(accuracy))

    # mse, action_predictions = evaluation.evaluate_fidelity(scm, test_data)
    # print(met.metrics)
    # print("MSE=" + str(mse))
    # print("Correct action predictions=" + str(action_predictions))

    ## Processing explanations ##
    # explanation_generator = ExplanationGenerator(env, scm, rl_agent)
    # for i in range(10):
    #     example_state = test_data[i][:env.state_space]
    #     example_action = test_data[i][env.state_space]
    #     example_counter_action = choice([i for i in range(env.action_space) if i != example_action])
    #     why_explanation = explanation_generator.generate_why_explanation(example_state, example_action)
    #     print(f'Why {env.actions[example_action]}?\n {why_explanation}')

    #     # why_not_explanation = explanation_generator.generate_why_not_explanation(example_state, example_action, example_counter_action)
    #     # print(f'Why not {env.actions[example_counter_action]}?\n {why_not_explanation}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
