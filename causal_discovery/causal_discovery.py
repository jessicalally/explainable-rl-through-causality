import argparse
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
import csv
from environment import Environment
from method import Method
import numpy as np
import os
import dill as pickle


def parse_args():
    parser = argparse.ArgumentParser(description='Causal Discovery')
    parser.add_argument(
        'path_to_dataset',
        type=str,
        help='Path to CSV file containing training data')
    parser.add_argument(
        '--env',
        type=str,
        default='cartpole',
        help='RL environment for performing causal discovery on')
    parser.add_argument('--method', type=str, default='varlingam',
                        help='Causal discovery method')
    parser.add_argument('--save_path', type=str, default='output/',
                        help='Path to save generated causal graphs')
    parser.add_argument(
        '--num_points',
        type=int,
        default=10000,
        help='Number of datapoints to use when inferring the causal structure'
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(f'{args.save_path}/graphs/')
        os.makedirs(f'{args.save_path}/causal_matrices/')
        os.makedirs(f'{args.save_path}/metrics/')

    return args


def get_data(path, num_datapoints=30000):
    if not os.path.exists(path):
        raise argparse.ArgumentError(f'file {args.dataset} does not exist')

    data = open(path)
    data = np.loadtxt(data, delimiter=",")

    num_datapoints = min(num_datapoints, data.shape[0])
    data = data[:num_datapoints, :]

    print(f'Data: {data.shape}')

    return data


def display_graphs(causal_matrix, true_dag, args, with_assumptions):
    save_path = f'{args.save_path}/graphs/{args.env}_{args.method}_assumptions_{with_assumptions}'
    GraphDAG(causal_matrix, true_dag, show=True, save_name=save_path)


def evaluate(causal_matrix, true_dag, args, with_assumptions):
    met = MetricsDAG(causal_matrix, true_dag)
    print(met.metrics)
    save_metrics(met.metrics, args, with_assumptions)


def save_metrics(metrics, args, with_assumptions):
    file = f'{args.save_path}/metrics/{args.env}_{args.method}_assumptions_{with_assumptions}.csv'

    with open(file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)


def main(args):
    env = Environment.get_env(args.env)
    method = Method.get_method(args.method)

    dataset_path = f"../output/causal_discovery_dataset/{env.name}_ddqn.pickle"
    reward_dataset_path = f"../output/reward_discovery_dataset/{env.name}_ddqn.pickle"

    with open(dataset_path, 'rb') as dataset_file:
        causal_discovery_dataset = pickle.load(dataset_file)

    with open(reward_dataset_path, 'rb') as dataset_file:
        reward_causal_discovery_dataset = pickle.load(dataset_file)

    causal_matrix_with_assumptions = method.generate_causal_matrix(
        causal_discovery_dataset,
        env,
        env.forbidden_edges,
        env.required_edges,
        with_assumptions=True
    )

    reward_causal_matrix_with_assumptions = method.generate_causal_matrix(
        reward_causal_discovery_dataset,
        env,
        env.forbidden_edges_reward,
        [],
        with_assumptions=True
    )

    display_graphs(
        causal_matrix_with_assumptions,
        env.true_dag,
        args,
        True
    )

    display_graphs(
        reward_causal_matrix_with_assumptions,
        env.reward_true_dag,
        args,
        True
    )

    evaluate(causal_matrix_with_assumptions, env.true_dag, args, True)
    evaluate(reward_causal_matrix_with_assumptions,
             env.reward_true_dag, args, True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
