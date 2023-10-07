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
from rl_algorithms.DQN2 import DQN
from rl_algorithms.DQN import DQN as DQNPretrained
from rl_algorithms.DDQN import DDQN
from rl_algorithms.A2C import A2C
from structural_causal_model import StructuralCausalModel
import time


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
    elif args.env == "starcraft":
        return Starcraft()
    else:
        raise ValueError(f"{args.env} environment is not implemented")


def get_rl_algorithm(args, env):
    if args.rl == "pg":
        return PolicyGradient(env)
    elif args.rl == "dqn":
        return DQN(env)
    elif args.rl == "ddqn":
        if args.env == "cartpole":
            # Since the default hyperparameters are tuned to Lunar Lander
            return DDQN(
                env,
                gamma=0.99,
                epsilon=1.0,
                epsilon_decay=0.99,
                batch_size=512,
                lr=0.01,
                reward_threshold=475
            )
        return DDQN(env, reward_threshold=200)
    elif args.rl == "sarsa":
        return SARSA(env)
    elif args.rl == "a2c":
        return A2C(env)
    else:
        raise ValueError(f"{args.rl} algorithm is not implemented")


# Generate causal matrix
def causal_discovery(
        dataset,
        method,
        env,
        forbidden_edges,
        required_edges,
        true_dag,
        threshold=0.3,
        restructure=False):
    print(f'Data: {dataset.shape}')

    causal_matrix_with_assumptions = method.generate_causal_matrix(
        dataset,
        env,
        forbidden_edges,
        required_edges,
        with_assumptions=True,
        threshold=threshold, restructure=restructure)

    print(causal_matrix_with_assumptions)
    if not restructure:
        for i in range(len(causal_matrix_with_assumptions)):
            for j in range(len(causal_matrix_with_assumptions[i])):
                if j != len(causal_matrix_with_assumptions[i]) - 1:
                    causal_matrix_with_assumptions[i][j] = 0

    learned_causal_graph = nx.from_numpy_matrix(
        causal_matrix_with_assumptions, create_using=nx.MultiDiGraph())

    met = MetricsDAG(causal_matrix_with_assumptions, true_dag)

    return learned_causal_graph, met.metrics, causal_matrix_with_assumptions


def get_known_edges(env, causal_matrix):
    forbidden_edges = [
        (i, j) for i, j in itertools.product(
            range(
                env.state_space), range(
                env.state_space))]
    required_edges = []

    for i, j in itertools.product(
        range(
            env.state_space), range(
            env.state_space)):
        if causal_matrix[i][j] == 1:
            forbidden_edges.remove((i, j))
            required_edges.append((i, j))

    for i in range(env.state_space):
        forbidden_edges.append((env.state_space, i))

    return forbidden_edges, required_edges


def run_causal_discovery_method(
        env,
        method,
        method_name,
        causal_discovery_dataset,
        reward_causal_discovery_dataset):
    # Since the VarLiNGAM algorithm is designed for handling time-series data
    # we don't need the explicit values for the variables at the next state in
    # each row
    if method_name == "VarLiNGAM":
        causal_discovery_dataset = causal_discovery_dataset[:,
                                                            :-env.state_space]

    print(method_name)
    print("feature causal discovery")
    st = time.process_time()
    learned_feature_causal_graph, met, causal_matrix_with_assumptions = causal_discovery(
        causal_discovery_dataset, method, env, env.forbidden_edges, env.required_edges, env.true_dag, restructure=True)
    et = time.process_time()
    print(causal_matrix_with_assumptions)
    print(met)
    print(f"elapsed time {et - st}")
    if method_name != "RL":
        print("reward causal discovery")
        forbidden_edges, required_edges = get_known_edges(
            env, causal_matrix_with_assumptions)
        st = time.process_time()
        learned_reward_causal_graph, met, causal_matrix_with_assumptions = causal_discovery(
            reward_causal_discovery_dataset, method, env, forbidden_edges, required_edges, env.reward_true_dag)
        et = time.process_time()
        print(causal_matrix_with_assumptions)
        print(met)
        print(f"elapsed time {et - st}")
    else:
        learned_reward_causal_graph = None
    return learned_feature_causal_graph, learned_reward_causal_graph


# Uses VarLiNGAM for features, and PC for rewards
def run_causal_discovery_optimal_methods(
        env,
        causal_discovery_dataset,
        reward_causal_discovery_dataset):
    causal_discovery_dataset = causal_discovery_dataset[:, :-env.state_space]
    print("Running VarLiNGAM for feature causal discovery...")
    method = VarLiNGAM()
    st = time.process_time()
    learned_feature_causal_graph, met, causal_matrix_with_assumptions = causal_discovery(
        causal_discovery_dataset, method, env, env.forbidden_edges, env.required_edges, env.true_dag, restructure=True)
    et = time.process_time()
    print(causal_matrix_with_assumptions)
    print(met)
    print(f"elapsed time {et - st}")

    print("Running PC for reward causal discovery...")
    method = PC()
    forbidden_edges, required_edges = get_known_edges(
        env, causal_matrix_with_assumptions)
    st = time.process_time()
    learned_reward_causal_graph, met, causal_matrix_with_assumptions = causal_discovery(
        reward_causal_discovery_dataset, method, env, forbidden_edges, required_edges, env.reward_true_dag)
    et = time.process_time()
    print(causal_matrix_with_assumptions)
    print(met)
    print(f"elapsed time {et - st}")

    if not np.any(causal_matrix_with_assumptions[:-1, -1]):
        print(f"old matrix {causal_matrix_with_assumptions}")
        # Reward causal discovery has picked up no reward features, so we make the assumption
        # that every feature influences the reward
        causal_matrix_with_assumptions[:-1, -1] = 1
        print(f"new matrix {causal_matrix_with_assumptions}")

        learned_reward_causal_graph = nx.from_numpy_matrix(
            causal_matrix_with_assumptions, create_using=nx.MultiDiGraph())

    return learned_feature_causal_graph, learned_reward_causal_graph


def run_all_causal_discovery_methods(
        env,
        causal_discovery_dataset,
        reward_causal_discovery_dataset):
    # PC
    run_causal_discovery_method(
        env,
        PC(),
        "PC",
        causal_discovery_dataset,
        reward_causal_discovery_dataset)

    # VarLiNGAM
    run_causal_discovery_method(
        env,
        VarLiNGAM(),
        "VarLiNGAM",
        causal_discovery_dataset,
        reward_causal_discovery_dataset)

    # Direct LiNGAM
    run_causal_discovery_method(
        env,
        DirectLiNGAM(),
        "DirectLiNGAM",
        causal_discovery_dataset,
        reward_causal_discovery_dataset)

    # NOTEARS
    run_causal_discovery_method(
        env,
        NOTEARS(),
        "NOTEARS",
        causal_discovery_dataset,
        reward_causal_discovery_dataset)

    # RL
    #run_causal_discovery_method(
    #    env,
    #    RL(),
    #    "RL",
    #    causal_discovery_dataset,
    #    reward_causal_discovery_dataset)

def run_causal_discovery(args, iter=0):
    print(f"running causal discovery iter {iter}");
    env = get_environment(args)

    if env.name == "mountaincar":
        # The DQN algorithm used for mountaincar can be quite unstable.
        # Because of this, it is sometimes easier to use pretrained models instead.
        # To train an agent, uncomment the if-branch of this if-statement.
        rl_agent = DQNPretrained(env, env.action_space, env.state_space)
        rl_agent.model.load_weights(f"output/trained_rl_agents/mountaincar_dqn_{iter}.h5")
        #causal_discovery_dataset = np.empty((0, 5))
        #reward_causal_discovery_dataset = np.empty((0, 3))
        with open('temp-transitions.txt','r') as f:
            causal_discovery_dataset = np.loadtxt(f)[iter*500000:(iter+1)*500000,:]
        with open('temp-rewards.txt','r') as f:
            reward_causal_discovery_dataset = np.loadtxt(f)[iter*500000:(iter+1)*500000,:]
        print(causal_discovery_dataset[-1, :])
        print(f"running causal discovery iter {iter}");
    else:
        rl_agent = get_rl_algorithm(args, env)

        # Train agent from scratch
        causal_discovery_dataset, reward_causal_discovery_dataset = rl_agent.train()
    # Generate datasets ##
    if len(causal_discovery_dataset) < 500000:
        num_datapoints = 500000 - len(causal_discovery_dataset)
        causal_discovery_dataset_extended, reward_causal_discovery_dataset_extended = rl_agent.generate_test_data_for_causal_discovery(num_datapoints, use_sum_rewards=True)
        print(causal_discovery_dataset.shape)
        print(reward_causal_discovery_dataset.shape)
        print(causal_discovery_dataset_extended.shape)
        print(reward_causal_discovery_dataset_extended.shape)
        causal_discovery_dataset = np.append(
            causal_discovery_dataset, causal_discovery_dataset_extended, axis=0)
        reward_causal_discovery_dataset = np.append(
            reward_causal_discovery_dataset,
            reward_causal_discovery_dataset_extended, 
            axis=0
        )

    causal_discovery_dataset = causal_discovery_dataset[:500000]
    reward_causal_discovery_dataset = reward_causal_discovery_dataset[:500000]

    print(causal_discovery_dataset.shape)
    print(reward_causal_discovery_dataset.shape)

    rl_agent_path = f"output/trained_rl_agents/{env.name}_{rl_agent.name}_{iter}.pickle"
    os.makedirs(os.path.dirname(rl_agent_path), exist_ok=True)

    dataset_path = f"output/causal_discovery_dataset/{env.name}_{rl_agent.name}_{iter}.pickle"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    reward_dataset_path = f"output/reward_discovery_dataset/{env.name}_{rl_agent.name}_{iter}.pickle"
    os.makedirs(os.path.dirname(reward_dataset_path), exist_ok=True)

    with open(rl_agent_path, 'wb') as agent_file:
        pickle.dump(rl_agent, agent_file)

    with open(dataset_path, 'wb') as dataset_file:
       pickle.dump(causal_discovery_dataset, dataset_file)

    with open(reward_dataset_path, 'wb') as dataset_file:
       pickle.dump(reward_causal_discovery_dataset, dataset_file)

    with open(rl_agent_path, 'rb') as rl_agent_file:
        rl_agent = pickle.load(rl_agent_file)

    if env.name == "starcraft":
        causal_discovery_dataset = np.genfromtxt(
            'starcraft_causal_discovery.csv', delimiter=',')
        causal_discovery_dataset = causal_discovery_dataset[:500000, :]
        reward_causal_discovery_dataset = np.genfromtxt(
            'starcraft_reward_causal_discovery.csv', delimiter=',')
        reward_causal_discovery_dataset = reward_causal_discovery_dataset[:500000, :]
    else:
        with open(dataset_path, 'rb') as dataset_file:
            causal_discovery_dataset = pickle.load(dataset_file)

        with open(reward_dataset_path, 'rb') as dataset_file:
            reward_causal_discovery_dataset = pickle.load(dataset_file)

    #run_all_causal_discovery_methods(
    #    env,
    #    causal_discovery_dataset,
    #    reward_causal_discovery_dataset)


def run_scm_training(args, iter=0):
    total_datapoints = 10000
    env = get_environment(args)
    rl_agent = get_rl_algorithm(args, env)

    if env.name == "starcraft":
        causal_discovery_dataset = np.genfromtxt(
            'starcraft_causal_discovery_adjusted.csv', delimiter=',')
        reward_causal_discovery_dataset = np.genfromtxt(
            'starcraft_reward_causal_discovery.csv', delimiter=',')
    elif env.name == "mountaincar":
        # The DQN algorithm used for mountaincar can be quite unstable.
        # Because of this, it is sometimes easier to use pretrained models instead.
        # To train an agent, uncomment the if-branch of this if-statement.
        rl_agent = DQNPretrained(env, env.action_space, env.state_space)
        rl_agent.model.load_weights(f"output/trained_rl_agents/mountaincar_dqn_{iter}.h5")
        
        with open('temp-transitions.txt','r') as f:
            causal_discovery_dataset = np.loadtxt(f)[iter*500000:(iter+1)*500000,:]
        with open('temp-rewards.txt','r') as f:
            reward_causal_discovery_dataset = np.loadtxt(f)[iter*500000:(iter+1)*500000,:]
    else:
        if env.name != "mountaincar":
            rl_agent_path = f"output/trained_rl_agents/{env.name}_{rl_agent.name}_{iter}.pickle"
            os.makedirs(os.path.dirname(rl_agent_path), exist_ok=True)

            with open(rl_agent_path, 'rb') as rl_agent_file:
                rl_agent = pickle.load(rl_agent_file)

        dataset_path = f"output/causal_discovery_dataset/{env.name}_{rl_agent.name}_{iter}.pickle"
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        reward_dataset_path = f"output/reward_discovery_dataset/{env.name}_{rl_agent.name}_{iter}.pickle"
        os.makedirs(os.path.dirname(reward_dataset_path), exist_ok=True)

        with open(dataset_path, 'rb') as dataset_file:
            causal_discovery_dataset = pickle.load(dataset_file)

        with open(reward_dataset_path, 'rb') as dataset_file:
            reward_causal_discovery_dataset = pickle.load(dataset_file)

    feature_causal_graph, reward_causal_graph = run_causal_discovery_optimal_methods(
        env, causal_discovery_dataset, reward_causal_discovery_dataset)
    feature_scm_test_data = None
    reward_scm_test_data = None

    scm_dataset = np.empty(causal_discovery_dataset.shape)
    reward_scm_dataset = np.empty(reward_causal_discovery_dataset.shape)
    scm_dataset, reward_scm_dataset = rl_agent.generate_test_data_for_scm_training(total_datapoints, use_sum_rewards=True)

    scm_dataset = scm_dataset[:total_datapoints]
    reward_scm_dataset = reward_scm_dataset[:total_datapoints]

    ## Train structural causal model ##
    # if env.name == "starcraft":
    #     rnd_indices = np.random.choice(len(causal_discovery_dataset), 20000)
    #     feature_scm_training_data = causal_discovery_dataset[rnd_indices[:10000]]
    #     feature_scm_test_data = causal_discovery_dataset[rnd_indices[10000:]]

    #     rnd_indices = np.random.choice(
    #         len(reward_causal_discovery_dataset), 20000)
    #     reward_scm_training_data = reward_causal_discovery_dataset[rnd_indices[:10000]]
    #     reward_scm_test_data = reward_causal_discovery_dataset[rnd_indices[10000:]]
    # else:
    #     # Reduce dataset and randomise to reduce overfitting
    #     rnd_indices = np.random.choice(len(causal_discovery_dataset), 10000)
    #     feature_scm_training_data = causal_discovery_dataset[rnd_indices]

    #     rnd_indices = np.random.choice(
    #         len(reward_causal_discovery_dataset), 10000)
    #     reward_scm_training_data = reward_causal_discovery_dataset[rnd_indices]

    # SCMs using true DAG
    scm = StructuralCausalModel(
        env,
        rl_agent,
        scm_dataset,
        nx.from_numpy_matrix(env.true_dag, create_using=nx.MultiDiGraph()),
        uses_true_dag=True,
        uses_nn_regressor=True
    )

    st = time.process_time()
    scm.train()
    et = time.process_time()
    print(f"feature scm elapsed time {et - st}")

    reward_scm = StructuralCausalModel(
        env,
        rl_agent,
        reward_scm_dataset,
        nx.from_numpy_matrix(
            env.reward_true_dag,
            create_using=nx.MultiDiGraph()),
        is_reward_scm=True,
        uses_true_dag=True,
        uses_nn_regressor=True)

    st = time.process_time()
    reward_scm.train()
    et = time.process_time()
    print(f"reward scm elapsed time {et - st}")
    
    scm_path = f"output/scm/true_dag/feature/{env.name}_{rl_agent.name}_{iter}.pickle"
    os.makedirs(os.path.dirname(scm_path), exist_ok=True)

    reward_scm_path = f"output/scm/true_dag/reward/{env.name}_{rl_agent.name}_{iter}.pickle"
    os.makedirs(os.path.dirname(reward_scm_path), exist_ok=True)
    
    with open(scm_path, 'wb') as f:
        pickle.dump(scm, f)

    with open(reward_scm_path, 'wb') as f:
        pickle.dump(reward_scm, f)
    
    with open(scm_path, 'rb') as f:
        scm = pickle.load(f)

    with open(reward_scm_path, 'rb') as f:
        reward_scm = pickle.load(f)
    feature_scm_test_data = None
    reward_scm_test_data = None
    scm_evaluation(
        env,
        rl_agent,
        scm,
        reward_scm,
        feature_scm_test_data,
        reward_scm_test_data)
    
    # SCMs using learned DAG
    scm = StructuralCausalModel(
        env,
        rl_agent,
        scm_dataset,
        feature_causal_graph,
        uses_true_dag=False,
        uses_nn_regressor=True,
    )

    st = time.process_time()
    scm.train()
    et = time.process_time()
    print(f"feature scm elapsed time {et - st}")

    reward_scm = StructuralCausalModel(
        env,
        rl_agent,
        reward_scm_dataset,
        reward_causal_graph,
        is_reward_scm=True,
        uses_true_dag=False,
        uses_nn_regressor=True
    )

    st = time.process_time()
    reward_scm.train()
    et = time.process_time()
    print(f"reward scm elapsed time {et - st}")
    
    scm_path = f"output/scm/learned_dag/feature/{env.name}_{rl_agent.name}_{iter}.pickle"
    os.makedirs(os.path.dirname(scm_path), exist_ok=True)

    reward_scm_path = f"output/scm/learned_dag/reward/{env.name}_{rl_agent.name}_{iter}.pickle"
    os.makedirs(os.path.dirname(reward_scm_path), exist_ok=True)
    
    with open(scm_path, 'wb') as f:
        pickle.dump(scm, f)

    with open(reward_scm_path, 'wb') as f:
        pickle.dump(reward_scm, f)
    
    with open(scm_path, 'rb') as f:
        scm = pickle.load(f)

    with open(reward_scm_path, 'rb') as f:
        reward_scm = pickle.load(f)

    scm_evaluation(
        env,
        rl_agent,
        scm,
        reward_scm,
        feature_scm_test_data,
        reward_scm_test_data)


def scm_evaluation(
        env,
        rl_agent,
        scm,
        reward_scm,
        test_data=None,
        reward_test_data=None):
    if test_data is None and reward_test_data is None:
        if env.name == "mountaincar":
            with open("transition_dqn_mountaincar_test_data.pickle", 'rb') as f:
                test_data = np.array(pickle.load(f))

            with open("adjusted_reward_dqn_mountaincar_test_data.pickle", 'rb') as f:
                reward_test_data = np.array(pickle.load(f))
        else:
            num_datapoints = 10000

            test_data, reward_test_data = rl_agent.generate_test_data_for_scm_training(
                num_datapoints, use_sum_rewards=True)
            print(test_data)
            print(test_data.shape)
            rnd_indices = np.random.choice(len(test_data), 2500)
            test_data = test_data[rnd_indices]

    print(f'Data: {test_data.shape}')

    avg_nrmse, action_accuracy = evaluation.evaluate_scm(scm, test_data)
    print(f"action prediction accuracy = {action_accuracy}")
    print(f"avg nrmse = {avg_nrmse}")

    avg_nrmse, action_accuracy = evaluation.evaluate_scm(
        reward_scm, reward_test_data)
    print(f"action prediction accuracy = {action_accuracy}")
    print(f"avg nrmse = {avg_nrmse}")


def run_explanation_generation(iter=0):
    env = get_environment(args)
    rl_agent = get_rl_algorithm(args, env)

    if env.name != "mountaincar" and env.name != "starcraft":
        rl_agent_path = f"output/trained_rl_agents/{env.name}_{rl_agent.name}_{iter}.pickle"
        os.makedirs(os.path.dirname(rl_agent_path), exist_ok=True)

        with open(rl_agent_path, 'rb') as rl_agent_file:
            rl_agent = pickle.load(rl_agent_file)

    scm_path = f"output/scm/learned_dag/feature/{env.name}_{rl_agent.name}_{iter}.pickle"
    os.makedirs(os.path.dirname(scm_path), exist_ok=True)

    reward_scm_path = f"output/scm/learned_dag/reward/{env.name}_{rl_agent.name}_{iter}.pickle"
    os.makedirs(os.path.dirname(reward_scm_path), exist_ok=True)

    with open(scm_path, 'rb') as f:
        scm = pickle.load(f)

    with open(reward_scm_path, 'rb') as f:
        reward_scm = pickle.load(f)

    # Generate test data
    if env.name == "mountaincar":
        with open("transition_dqn_mountaincar_test_data.pickle", 'rb') as f:
            test_data = np.array(pickle.load(f))
    elif env.name == "starcraft":
        test_data = np.genfromtxt(
            'starcraft_causal_discovery_adjusted.csv',
            delimiter=',')
    else:
        num_datapoints = 1000
        test_data, _ = rl_agent.generate_random_test_data(
            num_datapoints, use_sum_rewards=True)

    print(f'Data: {test_data.shape}')

    why_explanations = set()
    why_not_explanations = set()
    os.makedirs(os.path.dirname("output/explanations"), exist_ok=True)

    rnd_indices = np.random.choice(len(test_data), 10)
    test_data = test_data[rnd_indices]

    explanation_generator = ExplanationGenerator(
        env, scm, reward_scm, rl_agent)
    for i in range(10):
        print(f"iter {i}")
        if env.name == "taxi":
            pertubation = 1.0
        else:
            pertubation = 0.01

        example_state = test_data[i][:env.state_space]
        example_action = test_data[i][env.state_space]
        example_counter_action = choice(
            [i for i in range(env.action_space) if i != example_action])
        why_explanation = explanation_generator.generate_why_explanation(
            example_state, example_action, pertubation)
        # why_explanations.add(why_explanation)
        print(f'Why {env.actions[example_action]}?\n {why_explanation}')

        why_not_explanation = explanation_generator.generate_why_not_explanation(
            example_state, example_action, example_counter_action, pertubation)
        print(
            f'Why not {env.actions[example_counter_action]}?\n {why_not_explanation}')
        # why_not_explanations.add(why_not_explanation)


def main(args):
    # Evaluate causal discovery methods
    for i in range(0, 1):
        run_causal_discovery(args, i)

        # Run SCM training
        run_scm_training(args, i)

        # Generate explanations
        run_explanation_generation(i)


if __name__ == '__main__':
    args = parse_args()
    main(args)
