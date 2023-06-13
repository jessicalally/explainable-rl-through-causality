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


# TODO: refactor this so that RL hyperparameters are passed in from args
def get_rl_algorithm(args, env):
    if args.rl == "pg":
        return PolicyGradient(env)
    elif args.rl == "dqn":
        agent = DQN(env.action_space, env.state_space)
        agent.model.load_weights("output/trained_rl_agents/mountaincar_dqn_0.h5")
        return agent
    elif args.rl == "ddqn":
        if args.env == "cartpole":
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
def causal_discovery(dataset, method, env, forbidden_edges, required_edges, true_dag, threshold=0.3, restructure=False):
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
    forbidden_edges = [(i, j) for i, j in itertools.product(range(env.state_space), range(env.state_space))]
    required_edges = []

    for i,j in itertools.product(range(env.state_space), range(env.state_space)):
        if causal_matrix[i][j] == 1:
            forbidden_edges.remove((i, j))
            required_edges.append((i, j))

    for i in range(env.state_space):
        forbidden_edges.append((env.state_space, i))

    return forbidden_edges, required_edges

def run_causal_discovery_method(env, method, method_name, causal_discovery_dataset, reward_causal_discovery_dataset):
    if method_name == "VarLiNGAM":
        causal_discovery_dataset = causal_discovery_dataset[:, :-env.state_space]

    print(method_name)
    print("feature causal discovery")
    st = time.process_time()
    learned_feature_causal_graph, met, causal_matrix_with_assumptions = causal_discovery(causal_discovery_dataset, method, env, env.forbidden_edges, env.required_edges, env.true_dag, restructure=True)
    et = time.process_time()
    print(causal_matrix_with_assumptions)
    print(met)
    print(f"elapsed time {et - st}")

    print("reward causal discovery")
    forbidden_edges, required_edges = get_known_edges(env, causal_matrix_with_assumptions)
    st = time.process_time()
    learned_reward_causal_graph, met, causal_matrix_with_assumptions = causal_discovery(reward_causal_discovery_dataset, method, env, forbidden_edges, required_edges, env.reward_true_dag)
    et = time.process_time()
    print(causal_matrix_with_assumptions)
    print(met)
    print(f"elapsed time {et - st}")

    return learned_feature_causal_graph, learned_reward_causal_graph


def run_all_causal_discovery_methods(env, causal_discovery_dataset, reward_causal_discovery_dataset):
    # PC
    run_causal_discovery_method(env, PC(), "PC", causal_discovery_dataset, reward_causal_discovery_dataset)

    # VarLiNGAM
    run_causal_discovery_method(env, VarLiNGAM(), "VarLiNGAM", causal_discovery_dataset, reward_causal_discovery_dataset)

    # Direct LiNGAM
    run_causal_discovery_method(env, DirectLiNGAM(), "DirectLiNGAM", causal_discovery_dataset, reward_causal_discovery_dataset)

    # NOTEARS
    run_causal_discovery_method(env, NOTEARS(), "NOTEARS", causal_discovery_dataset, reward_causal_discovery_dataset)

    # RL
    run_causal_discovery_method(env, RL(), "RL", causal_discovery_dataset, reward_causal_discovery_dataset)


def run_iter(args, iter):
    env = get_environment(args)
    rl_agent = get_rl_algorithm(args, env)

    # Paths
    rl_agent_path = f"output/trained_rl_agents/{env.name}_{rl_agent.name}_{iter}.pickle"
    os.makedirs(os.path.dirname(rl_agent_path), exist_ok=True)

    dataset_path = f"output/causal_discovery_dataset/{env.name}_{rl_agent.name}_{iter}.pickle"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    reward_dataset_path = f"output/reward_discovery_dataset/{env.name}_{rl_agent.name}_{iter}.pickle"
    os.makedirs(os.path.dirname(reward_dataset_path), exist_ok=True)

    # Train agent from scratch or load
    # causal_discovery_dataset, reward_causal_discovery_dataset = rl_agent.train()

    # # Generate datasets ##
    # if len(causal_discovery_dataset) < 500000:
    #    num_datapoints = 500000 - len(causal_discovery_dataset)
    #    causal_discovery_dataset_extended, reward_causal_discovery_dataset_extended = rl_agent.generate_test_data_for_causal_discovery(num_datapoints, use_sum_rewards=True)
    #    print(causal_discovery_dataset.shape)
    #    print(reward_causal_discovery_dataset.shape)
    #    print(causal_discovery_dataset_extended.shape)
    #    print(reward_causal_discovery_dataset_extended.shape)
    #    causal_discovery_dataset = np.append(causal_discovery_dataset, causal_discovery_dataset_extended, axis=0)
    #    reward_causal_discovery_dataset = np.append(reward_causal_discovery_dataset, reward_causal_discovery_dataset_extended, axis=0)

    # causal_discovery_dataset = causal_discovery_dataset[:500000]
    # reward_causal_discovery_dataset = reward_causal_discovery_dataset[:500000]

    # print(causal_discovery_dataset.shape)
    # print(reward_causal_discovery_dataset.shape)

    # with open(rl_agent_path, 'wb') as agent_file:
    #     pickle.dump(rl_agent, agent_file)

    # with open(dataset_path, 'wb') as dataset_file:
    #    pickle.dump(causal_discovery_dataset, dataset_file)

    # with open(reward_dataset_path, 'wb') as dataset_file:
    #    pickle.dump(reward_causal_discovery_dataset, dataset_file)

    # with open(rl_agent_path, 'rb') as rl_agent_file:
    #     rl_agent = pickle.load(rl_agent_file)

    with open(dataset_path, 'rb') as dataset_file:
        causal_discovery_dataset = pickle.load(dataset_file)

    with open(reward_dataset_path, 'rb') as dataset_file:
        reward_causal_discovery_dataset = pickle.load(dataset_file)

    if env.name == "mountaincar":
        causal_discovery_dataset = np.array(causal_discovery_dataset)
        reward_causal_discovery_dataset = np.array(reward_causal_discovery_dataset)

    run_all_causal_discovery_methods(env, causal_discovery_dataset, reward_causal_discovery_dataset)


def run_scm_training(args):
    env = get_environment(args)
    rl_agent = get_rl_algorithm(args, env)

    rl_agent_path = f"output/trained_rl_agents/{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(rl_agent_path), exist_ok=True)

    dataset_path = f"output/causal_discovery_dataset/{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    reward_dataset_path = f"output/reward_discovery_dataset/{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(reward_dataset_path), exist_ok=True)

    with open(rl_agent_path, 'rb') as rl_agent_file:
        rl_agent = pickle.load(rl_agent_file)

    with open(dataset_path, 'rb') as dataset_file:
        causal_discovery_dataset = pickle.load(dataset_file)

    with open(reward_dataset_path, 'rb') as dataset_file:
        reward_causal_discovery_dataset = pickle.load(dataset_file)

    feature_causal_graph, reward_causal_graph = run_causal_discovery_method(env, VarLiNGAM(), "VarLiNGAM", causal_discovery_dataset, reward_causal_discovery_dataset)

    ## Train structural causal model ##

    # Reduce dataset and randomise to reduce overfitting
    rnd_indices = np.random.choice(len(causal_discovery_dataset), 10000)
    feature_scm_training_data = causal_discovery_dataset[rnd_indices]

    rnd_indices = np.random.choice(len(reward_causal_discovery_dataset), 10000)
    reward_scm_training_data = reward_causal_discovery_dataset[rnd_indices]

    scm = StructuralCausalModel(
        env,
        rl_agent,
        feature_scm_training_data,
        feature_causal_graph,
        uses_true_dag=True
    )

    st = time.process_time()
    scm.train()
    et = time.process_time()
    print(f"feature scm elapsed time {et - st}")

    reward_scm = StructuralCausalModel(
        env,
        rl_agent,
        reward_scm_training_data,
        reward_causal_graph,
        is_reward = True,
        uses_true_dag=True
    )

    st = time.process_time()
    reward_scm.train()
    et = time.process_time()
    print(f"reward scm elapsed time {et - st}")

    scm_path = f"output/scm/learned_dag/feature/{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(scm_path), exist_ok=True)

    reward_scm_path = f"output/scm/learned_dag/reward/{env.name}_{rl_agent.name}.pickle"
    os.makedirs(os.path.dirname(reward_scm_path), exist_ok=True)

    with open(scm_path, 'wb') as f:
        pickle.dump(scm, f)

    with open(reward_scm_path, 'wb') as f:
        pickle.dump(reward_scm, f)

    # with open(scm_path, 'rb') as f:
    #     scm = pickle.load(f)

    # with open(reward_scm_path, 'rb') as f:
    #     reward_scm = pickle.load(f)

    ## Evaluation ##

    if env.name == "mountaincar":
        with open("transition_dqn_mountaincar_test_data.pickle", 'rb') as f:
            test_data = pickle.load(f)

        with open("adjusted_reward_dqn_mountaincar_test_data.pickle", 'rb') as f:
            reward_test_data = pickle.load(f)
    else:
        num_datapoints = 10000

        test_data, reward_test_data = rl_agent.generate_test_data_for_causal_discovery(num_datapoints, use_sum_rewards=True)
        print(test_data.shape)
        rnd_indices = np.random.choice(len(test_data), 2500)
        test_data = test_data[rnd_indices]

    print(f'Data: {test_data.shape}')

    avg_nrmse, action_accuracy = evaluation.evaluate_scm(scm, test_data)
    print(f"action prediction accuracy = {action_accuracy}")
    print(f"avg nrmse = {avg_nrmse}")

    avg_nrmse, action_accuracy = evaluation.evaluate_scm(reward_scm, reward_test_data)
    print(f"action prediction accuracy = {action_accuracy}")
    print(f"avg nrmse = {avg_nrmse}")

    # with open(f"{env.name}_{rl_agent.name}_metrics.txt", 'w') as f:
    #     f.write("MSE=" + str(mse))
    #     f.write("MSE ignoring reward=" + str(mse_ignoring_reward))
    #     f.write("Correct action predictions=" + str(action_predictions))


def main(args):
    # for iter in range(0, 3):
    #     run_iter(args, iter)

    # run_scm_training(args)
    env = get_environment(args)
    rl_agent = get_rl_algorithm(args, env)

    ## Learn causal graph ##

#     forbidden_edges = [(i, j) for i, j in itertools.product(range(env.state_space), range(env.state_space))]
#     required_edges = []

#     for i,j in itertools.product(range(env.state_space), range(env.state_space)):
#         if causal_matrix_with_assumptions[i][j] == 1:
#             forbidden_edges.remove((i, j))
#             required_edges.append((i, j))

#     for i in range(env.state_space):
#         forbidden_edges.append((env.state_space, i))
    
#     print("VARLINGAM")
#     learned_reward_causal_graph, met, causal_matrix_with_assumptions = causal_discovery(reward_causal_discovery_dataset, env, forbidden_edges, required_edges, env.reward_true_dag)
#     print(causal_matrix_with_assumptions)

#     method = PC()

#     print(f'forbidden {forbidden_edges}')
#     print(f'required {required_edges}')
    
#     causal_matrix_with_assumptions = method.generate_causal_matrix(
#         reward_causal_discovery_dataset,
#         env,
#         forbidden_edges,
#         required_edges,
#         with_assumptions=True)
#     print("PC")
#     print(causal_matrix_with_assumptions)

#     reward_met = MetricsDAG(causal_matrix_with_assumptions, env.reward_true_dag)

#     learned_reward_causal_graph = nx.from_numpy_matrix(
#         causal_matrix_with_assumptions, create_using=nx.MultiDiGraph())


#     metrics_path = f"output/metrics/{env.name}_{rl_agent.name}.pickle"
#     os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

#     causal_graph_path = f"output/causal_graph/{env.name}_{rl_agent.name}.pickle"
#     os.makedirs(os.path.dirname(causal_graph_path), exist_ok=True)

#     print(met)

#     # with open(metrics_path, 'wb') as metrics_file:
#     #     pickle.dump(met, metrics_file)

#     # with open(causal_graph_path, 'wb') as graph_file:
#     #     pickle.dump(learned_causal_graph, graph_file)

#     # with open(metrics_path,'rb') as metrics_file:
#     #     met = pickle.load(metrics_file)

#     # with open(causal_graph_path, 'rb') as graph_file:
#     #     learned_causal_graph = pickle.load(graph_file)


#     ## Train structural causal model ##

#     # scm_dataset_path = f"output/scm_dataset/test/causal{env.name}_{rl_agent.name}.pickle"
#     # os.makedirs(os.path.dirname(scm_dataset_path), exist_ok=True)

#     # reward_scm_dataset_path = f"output/scm_dataset/test/reward{env.name}_{rl_agent.name}.pickle"
#     # os.makedirs(os.path.dirname(reward_scm_dataset_path), exist_ok=True)

# #     num_datapoints = 500000
# #     # TODO: these two functions are the same now
# #     # scm_dataset = rl_agent.generate_test_data_for_scm(num_datapoints)
# #     scm_dataset, reward_scm_dataset = rl_agent.generate_test_data_for_causal_discovery(num_datapoints, use_sum_rewards=True)

# #     with open(scm_dataset_path, 'wb') as dataset_file:
# #         pickle.dump(scm_dataset, dataset_file)

# #     with open(reward_scm_dataset_path, 'wb') as dataset_file:
# #         pickle.dump(reward_scm_dataset, dataset_file)

# #     # with open(scm_dataset_path, 'rb') as dataset_file:
# #     #     scm_dataset = pickle.load(dataset_file)

# #     # with open(reward_scm_dataset_path, 'rb') as dataset_file:
# #     #     reward_scm_dataset = pickle.load(dataset_file)

#     num_datapoints = 500000

#     scm_training_data, reward_test_data = rl_agent.generate_test_data_for_causal_discovery(num_datapoints, use_sum_rewards=True)

#     rnd_indices = np.random.choice(len(scm_training_data), 10000)
#     scm_training_data = scm_training_data[rnd_indices]

#     # TODO: trying to reduce overfitting of this model
#     scm = StructuralCausalModel(
#         env,
#         scm_training_data,
#         learned_causal_graph
#     )

#     scm.train()

#     # TODO: we need to train the reward SCM on the RL training data because otherwise the RL agent
#     # never properly terminates due to falling out the required range of cart position and pole angle
#     reward_scm = StructuralCausalModel(
#         env,
#         reward_causal_discovery_dataset,
#         learned_reward_causal_graph,
#         for_reward = True,
#     )

#     reward_scm.train()

#     scm_path = f"output/scm/learned_dag/causal{env.name}_{rl_agent.name}.pickle"
#     os.makedirs(os.path.dirname(scm_path), exist_ok=True)

#     reward_scm_path = f"output/scm/learned_dag/reward{env.name}_{rl_agent.name}.pickle"
#     os.makedirs(os.path.dirname(reward_scm_path), exist_ok=True)

#     # with open(scm_path, 'wb') as f:
#     #     pickle.dump(scm, f)

#     # with open(reward_scm_path, 'wb') as f:
#     #     pickle.dump(reward_scm, f)

#     with open(scm_path, 'rb') as f:
#         scm = pickle.load(f)

#     with open(reward_scm_path, 'rb') as f:
#         reward_scm = pickle.load(f)

#     ## Evaluation ##

#     num_datapoints = 10000

#     test_data, reward_test_data = rl_agent.generate_test_data_for_causal_discovery(num_datapoints, use_sum_rewards=True)
#     print(test_data.shape)
#     rnd_indices = np.random.choice(len(test_data), 2500)
#     test_data = test_data[rnd_indices]

#     print(f'Data: {test_data.shape}')

#     # accuracy = evaluation.task_prediction(data, scm)
#     # print("Accuracy="+str(accuracy))

# #     # with open(f"{env.name}_{rl_agent.name}_metrics.txt", 'w') as f:
# #     #     f.write("MSE=" + str(mse))
# #     #     f.write("MSE ignoring reward=" + str(mse_ignoring_reward))
# #     #     f.write("Correct action predictions=" + str(action_predictions))

#     ## Processing explanations ##
    # os.makedirs(os.path.dirname("output/explanations/"), exist_ok=True)

#     why_explanations = set()
#     why_not_explanations = set()

#     explanation_generator = ExplanationGenerator(env, scm, reward_scm, rl_agent)
#     for i in range(10):
#         if env.name == "taxi":
#             pertubation = 1.0
#         else:
#             pertubation = 0.01

#         example_state = test_data[i][:env.state_space]
#         example_action = test_data[i][env.state_space]
#         example_counter_action = choice([i for i in range(env.action_space) if i != example_action])
#         why_explanation = explanation_generator.generate_why_explanation(example_state, example_action, pertubation)
#         why_explanations.add(why_explanation)
#         print(f'Why {env.actions[example_action]}?\n {why_explanation}')

#         why_not_explanation = explanation_generator.generate_why_not_explanation(example_state, example_action, example_counter_action, pertubation)
#         print(f'Why not {env.actions[example_counter_action]}?\n {why_not_explanation}')
#         why_not_explanations.add(why_not_explanation)
    
#     with open(f"{env.name}_{rl_agent.name}_explanations.txt", 'w') as f:
#         f.write(str(why_explanations))
#         f.write(str(why_not_explanations))

#     # mse, action_predictions = evaluation.evaluate_fidelity(scm, test_data)
#     # reward_mse, _ = evaluation.evaluate_fidelity(reward_scm, reward_test_data, REWARD_DAG=True)
#     # print(met)
#     # print("MSE=" + str(mse))
#     # print("Correct action predictions=" + str(action_predictions))

#     # print(reward_met.metrics)
#     # print("MSE=" + str(reward_mse))

if __name__ == '__main__':
    args = parse_args()
    main(args)
