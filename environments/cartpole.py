import copy
import tensorflow.compat.v1 as tf
import networkx as nx
import numpy as np
import pandas as pd
import explanation_templates as explanations
import rl_algorithms.q_learning
import gym

def Discrete(state, bins):
    index = []

    for i in range(len(state)):
        index.append(np.digitize(state[i],bins[i]) - 1)
    return tuple(index)

causal_graph = np.array([
    [1, 0, 0, 0], # 0 = cart position
    [1, 1, 0, 0], # 1 = cart velocity
    [0, 0, 1, 0], # 2 = pole angle
    [0, 0, 1, 1], # 3 = pole angular velocity
])

causal_graph2 = np.array([
    [0, 0, 0, 0], # 0 = cart position
    [1, 0, 0, 0], # 1 = cart velocity
    [0, 0, 0, 0], # 2 = pole angle
    [0, 0, 1, 0], # 3 = pole angular velocity
])

action_set = (0, 1) # 0 = push cart to left, 1 = push cart to right

equation_predictions = {}

causal_graph = nx.from_numpy_matrix(causal_graph, create_using=nx.MultiDiGraph())
causal_graph2 = nx.from_numpy_matrix(causal_graph2, create_using=nx.MultiDiGraph())

for edge in causal_graph.edges():
    causal_graph.remove_edge(edge[0], edge[1])
    causal_graph.add_edge(edge[0], edge[1], action=0) # Push cart to left
    causal_graph.add_edge(edge[0], edge[1], action=1) # Push cart to right

for edge in causal_graph2.edges():
    causal_graph2.remove_edge(edge[0], edge[1])
    causal_graph2.add_edge(edge[0], edge[1], action=0) # Push cart to left
    causal_graph2.add_edge(edge[0], edge[1], action=1) # Push cart to right

num_episodes = 300
time_frame = 500 # Max number of steps per episodes

# def test_trained_agent(q_table, bins, structural_equations):
#     current_observation, _ = env.reset()
#     current_state = Discrete(current_observation, bins)
#     alternative_env = copy.deepcopy(env)

#     for _ in range(20):
#         action = np.argmax(q_table[current_state])
        
#         actual_next_observation, reward, done, terminated, info = env.step(action)
#         alternative_next_observation, _, _, _, _ = alternative_env.step(1 - action)
#         actual_next_state = Discrete(actual_next_observation, bins)

#         predicted_next_observation = predict_from_scm_given_action(structural_equations, current_observation, action)
#         predicted_next_observation_given_other_action = predict_from_scm_given_action(structural_equations, current_observation, 1 - action)

#         print("Current state: " + str(current_observation))
#         print("Chosen action: " + str(action))
#         print("Actual next state: " + str(actual_next_observation))
#         print("Predicted next state: " + str(predicted_next_observation))
#         print("Alternative next state: " + str(alternative_next_observation))
#         print("Predicted alternative state: " + str(predicted_next_observation_given_other_action))
#         print("\n")

#         alternative_env = copy.deepcopy(env)
#         current_state = actual_next_state
#         current_observation = actual_next_observation


def train_scm(data_set):
    state_set = []
    action_set = []
    next_state_set = []

    for (s, a, _, s_next) in data_set: 
        state_set.append(s)
        action_set.append(a)
        next_state_set.append(s_next)
    
    assert(len(state_set) == len(action_set))
    
    structural_equations = train_model(state_set, action_set, next_state_set, data_set)

    return structural_equations
    # process_explanations(structural_equations, state_set, action_set, next_state_set)

# def evaluate_accuracy(q_table, bins, structural_equations):
#     print("Starting SCM evaluation...")

#     num_correct = 0
#     total = 0

#     current_observation, _ = env.reset()
#     current_state = Discrete(current_observation, bins)

#     for idx in range(20):
#         action = np.argmax(q_table[current_state])

#         actual_next_observation, reward, done, terminated, info = env.step(action)
#         actual_next_state = Discrete(actual_next_observation, bins)

#         if idx % 1 == 0:
#             print(idx)

#             print("state: " + str(current_observation))
#             print("next state: " + str(actual_next_observation))
#             print("action: " + str(action))
#             print("\n")

#             predict_next_states = predict_from_scm(structural_equations, current_observation) # these don't seem to be correct
#             print("predicted next states: " + str(predict_next_states))

#             diff_with_actual_value = {}
#             total_diffs_per_action = {}

#             for key in structural_equations:
#                 predicted_value = predict_next_states[key]
#                 actual_value = actual_next_observation[key[0]]
#                 diff_with_actual_value[key] = abs(predicted_value - actual_value)

#                 if key[1] in total_diffs_per_action:
#                     total_diffs_per_action[key[1]] += diff_with_actual_value[key]
#                 else:
#                     total_diffs_per_action[key[1]] = diff_with_actual_value[key]
                
#             print("diffs: " + str(total_diffs_per_action))
#             predicted_action = min(total_diffs_per_action, key=total_diffs_per_action.get)

#             print("predicted action: " + str(predicted_action))
#             print("\n")

#             # print(diff_with_actual_value)
#             # predicted_action = max(diff_with_actual_value, key=diff_with_actual_value.get)[1]
#             # we could look at the total diffs for each action
#             # then the predicted action is the one with the least difference
#             total += 1

#             if action == predicted_action:
#                 num_correct += 1

#         current_observation = actual_next_observation
#         current_state = actual_next_state

#     accuracy = (num_correct / total) * 100
#     print("accuracy is " + str(accuracy) + "%")

def predict_from_scm_given_action(structural_equations, s, a):
    predicted_state = [0, 0, 0, 0]
    nodes = [0, 1, 2, 3]

    for node in nodes:
        key = (node, a)
        x_data = []

        if key[0] == 0:
            x_data = s[[0, 1]]
        elif key[0] == 1:
            x_data = [s[1]]
        elif key[0] == 2:
            x_data = s[[2, 3]]
        elif key[0] == 3:
            x_data = [s[3]]

        pred = structural_equations[key]['function'].predict(input_fn=get_predict_fn(x_data,                          
                    num_epochs=1,                          
                    n_batch = 128,                          
                    shuffle=False))

        for item in pred:
            predicted_state[node] = item['predictions'][0]

    return predicted_state

def predict_from_scm(structural_equations, s):
    predict_y = {}

    for key in structural_equations:
        x_data = []
        if key[0] == 0:
            x_data = s[[0, 1]]
        elif key[0] == 1:
            x_data = [s[1]]
        elif key[0] == 2:
            x_data = s[[2, 3]]
        elif key[0] == 3:
            x_data = [s[3]]

        pred = structural_equations[key]['function'].predict(input_fn=get_predict_fn(x_data,                          
                num_epochs=1,                          
                n_batch = 128,                          
                shuffle=False))

        predict_y[key] = np.array([item['predictions'][0] for item in pred])

    return predict_y

def train_structural_equations(structural_equations):
    for key in structural_equations:
        structural_equations[key]['function'].train(input_fn=get_input_fn(structural_equations[key],                                       
                                                                            num_epochs=None,                                      
                                                                            n_batch = 128,                                      
                                                                            shuffle=False),                                      
                                                                            steps=1000)
    return structural_equations

def train_model(state_set, action_set, next_state_set, data_set):
    print("Starting SCM training...")
    action_influence_dataset = {}

    for idx, action in enumerate(action_set):
        if action in action_influence_dataset:
            action_influence_dataset[action]['state'].append(state_set[idx])
            action_influence_dataset[action]['next_state'].append(next_state_set[idx])
        else:
            action_influence_dataset[action] = {'state' : [], 'next_state': []}
            action_influence_dataset[action]['state'].append(state_set[idx])
            action_influence_dataset[action]['next_state'].append(next_state_set[idx])

    structural_equations = initialize_structural_equations(causal_graph, action_influence_dataset)
    structural_equations = train_structural_equations(structural_equations)
    print('Ending SCM training...')

    return structural_equations

def process_explanations(structural_equations, state_set, action_set, next_state_set):
    print('Processing explanations...')
    

    # for agent_step in range(1, len(state_set) + 1, 1000):
    # print(str(agent_step) + "/" + str(len(state_set)))
    agent_step = 1000
    action = action_set[agent_step]
    why_explanations = {}
    why_not_explanations = {}

    why_explanations[(agent_step, action)] = {'state': state_set[agent_step], 'why_exps': generate_why_explanations(state_set[agent_step], action, agent_step, causal_graph2, structural_equations)}

    # poss_counter_actions = set(action_set).difference({action})
    # for counter_action in poss_counter_actions:
        # why_not_explanations[(agent_step, action, counter_action)] = {'state': state_set[agent_step], 
        #                                         'why_not_exps': generate_counterfactual_explanations(state_set[agent_step], action, counter_action, agent_step, causal_graph, structural_equations)}

    pd.DataFrame.from_dict(data=why_explanations, orient='index').to_csv('why_explanations_cartpole.csv', mode='a', header=False)
    # pd.DataFrame.from_dict(data=why_not_explanations, orient='index').to_csv('why_not_explanations_cartpole.csv', mode='a', header=False)


def initialize_structural_equations(causal_graph, action_influence_dataset):
    structural_equations = {}
    unique_functions = {}

    for edge in causal_graph.edges():
        for preds in causal_graph.predecessors(edge[1]):
            node = edge[1]
            if (node, 0) not in unique_functions:
                unique_functions[(node, 0)] = set()
            
            if (node, 1) not in unique_functions:
                unique_functions[(node, 1)] = set()

            # Between every node and its predecessor we have L and R actions
            unique_functions[(node, 0)].add(preds)
            unique_functions[(node, 1)].add(preds)

    print(unique_functions)

    for key in unique_functions:
        if key[1] in action_influence_dataset:
            x_data = []

            for x_feature in unique_functions[key]:
                x_data.append((np.array(action_influence_dataset[key[1]]['state'])[:,[x_feature]]).flatten()) # instead of x_feature we want X-feature AND key[0]

            x_feature_cols = [tf.feature_column.numeric_column(str(i)) for i in range(len(x_data))]  # we might want to change this

            y_data = np.array(action_influence_dataset[key[1]]['next_state'])[:,key[0]] # get new values of key[0]
            lr = tf.estimator.LinearRegressor(feature_columns=x_feature_cols, model_dir='scm_models/linear_regressor/'+str(key[0])+'_'+str(key[1]))
            structural_equations[key] = {'X': x_data,'Y': y_data,'function': lr,}
    
    return structural_equations

def generate_why_explanations(data_set, actual_state, actual_action, state_num_in_batch, causal_graph, structural_equations):
    optimal_state_set = []
    actual_state = {k: actual_state[k] for k in range(len(actual_state))}
    sink_nodes = get_sink_nodes(causal_graph)
    actual_action_edge_list = get_edges_of_actions(actual_action, causal_graph)
    all_actual_causal_chains_from_action = get_causal_chains(sink_nodes, actual_action_edge_list, causal_graph)
    # print(all_actual_causal_chains_from_action)
    action_chain_list = get_action_chains(actual_action, all_actual_causal_chains_from_action, causal_graph)
    # print(action_chain_list)

    why_exps = set()
    for i in range(len(all_actual_causal_chains_from_action)):
        optimal_state = dict(actual_state)
        print(optimal_state)
        for j in range(len(all_actual_causal_chains_from_action[i])):
            for k in range(len(all_actual_causal_chains_from_action[i][j])):
                optimal_state[all_actual_causal_chains_from_action[i][j][k]] = predict_node_scm(
                    data_set,
                    all_actual_causal_chains_from_action[i][j][k], action_chain_list[i][j][k], structural_equations)[state_num_in_batch]

        optimal_state_set.append(optimal_state)
        min_tuple_actual_state = get_minimally_complete_tuples(all_actual_causal_chains_from_action[i], actual_state)
        min_tuple_optimal_state = get_minimally_complete_tuples(all_actual_causal_chains_from_action[i], optimal_state)
        explanation = explanations.cartpole_generate_why_text_explanations(min_tuple_actual_state, min_tuple_optimal_state, actual_state, actual_action)
        print(explanation)
        print("\n")
        why_exps.add(explanation)

    return why_exps


def generate_counterfactual_explanations(actual_state, actual_action, counterfactual_action, state_num_in_batch, causal_graph, structural_equations):
    counterfactual_state_set = []
    actual_state = {k: actual_state[k] for k in range(len(actual_state))}
    sink_nodes = get_sink_nodes(causal_graph)
    counter_action_edge_list = get_edges_of_actions(counterfactual_action, causal_graph)
    actual_action_edge_list = get_edges_of_actions(actual_action, causal_graph)
    
    all_counter_causal_chains_from_action = get_causal_chains(sink_nodes, counter_action_edge_list, causal_graph)
    all_actual_causal_chains_from_action = get_causal_chains(sink_nodes, actual_action_edge_list, causal_graph)
    action_chain_list = get_action_chains(counterfactual_action, all_counter_causal_chains_from_action, causal_graph)
    
    for i in range(len(all_counter_causal_chains_from_action)):
        counterfactual_state = dict(actual_state)
        for j in range(len(all_counter_causal_chains_from_action[i])):
            for k in range(len(all_counter_causal_chains_from_action[i][j])):
                counterfactual_state[all_counter_causal_chains_from_action[i][j][k]] = predict_node_scm(
                    all_counter_causal_chains_from_action[i][j][k], action_chain_list[i][j][k], structural_equations)[state_num_in_batch]
        counterfactual_state_set.append(counterfactual_state)    
    
    contrastive_exp = set()
    for actual_chains in all_actual_causal_chains_from_action:
        for counter_chains in all_counter_causal_chains_from_action:
            for counter_states in counterfactual_state_set:
                contrast_tuple = get_minimal_contrastive_tuples(actual_chains, counter_chains, actual_state, counter_states)
                contrastive_exp.add(explanations.cartpole_generate_contrastive_text_explanations(contrast_tuple, actual_action))    

    for exp in contrastive_exp:
        print(exp)
        print("\n")
    # unique contrastive explanations
    return contrastive_exp 


def predict_node_scm(node, action, structural_equations):
    key = (node, action)

    pred = structural_equations[key]['function'].predict(input_fn=get_input_fn(structural_equations[key],                          
                num_epochs=1,                          
                n_batch = 128,                          
                shuffle=False))

    result = np.array([item['predictions'][0] for item in pred])

    return result
    
def test_predict_from_scm(structural_equations):
    result = predict_from_scm(structural_equations, [0, 0, 0, 0])
    print(result)
    
"""minimally complete tuple = (head node of action, immediate pred of sink nodes, sink nodes)"""
def get_minimally_complete_tuples(chains, state):
    head = set()
    immediate = set()
    reward = set()
    for chain in chains:
        if len(chain) == 1:
            reward.add((chain[0], state[chain[0]]))
        if len(chain) == 2:
            head.add((chain[0], state[chain[0]]))
            reward.add((chain[-1], state[chain[-1]]))
        if len(chain) > 2:    
            head.add((chain[0], state[chain[0]]))
            immediate.add((chain[-2], state[chain[-2]]))
            reward.add((chain[-1], state[chain[-1]]))
    minimally_complete_tuple = {
        'head': head,
        'immediate': immediate,
        'reward': reward
    }
    return minimally_complete_tuple    

def get_minimal_contrastive_tuples(actual_chain, counterfactual_chain, actual_state, counterfactual_state):

    actual_minimally_complete_tuple = get_minimally_complete_tuples(actual_chain, actual_state)
    counterfactual_minimally_complete_tuple = get_minimally_complete_tuples(counterfactual_chain, counterfactual_state)
    min_tuples = np.sum(np.array([list(k) for k in list(actual_minimally_complete_tuple.values())]))
    tuple_states = set([k[0] for k in min_tuples])

    counter_min_tuples = np.sum(np.array([list(k) for k in list(counterfactual_minimally_complete_tuple.values())]))
    counter_tuple_states = set([k[0] for k in counter_min_tuples])
    counter_tuple_states.difference_update(tuple_states)

    contrastive_tuple = {
                        'actual': {k: actual_state[k] for k in counter_tuple_states},
                        'counterfactual': {k: counterfactual_state[k] for k in counter_tuple_states},
                        'reward': {k[0]: k[1] for k in actual_minimally_complete_tuple['reward']}
                        }
    return contrastive_tuple


def get_causal_chains(sink_nodes, action_edge_list, causal_graph):
    # Action edge list contains all the edges corresponding to the action

    counter_action_head_set = set(np.array(action_edge_list)[:,1]) 
    all_causal_chains_from_action = []

    for action_head in counter_action_head_set:
        chains_to_sink_nodes = []
        for snode in sink_nodes:
            if action_head == snode:
                chains_to_sink_nodes.append([snode])
            else:
                chains_to_sink_nodes.extend((nx.all_simple_paths(causal_graph, source=action_head, target=snode)))
        all_causal_chains_from_action.append(chains_to_sink_nodes)

    return all_causal_chains_from_action    


def get_action_chains(action, chain_lists_of_action, causal_graph):
    action_chain_list = []
    for chain_list in chain_lists_of_action:
        action_chains = []
        for chain in chain_list:
            action_chain = []
            for i in range(len(chain)):
                if i == 0:
                    action_chain.append(action)  
                else:
                    action_chain.append(causal_graph.get_edge_data(chain[i-1], chain[i])[0]['action'])
            action_chains.append(action_chain)
        action_chain_list.append(action_chains)          
    return action_chain_list        


def get_edges_of_actions(action, causal_graph):
    return list(edge for edge in causal_graph.edges(data=True) if edge[2]['action'] == action)
   
def get_sink_nodes(causal_graph):
    return list((node for node, out_degree in causal_graph.out_degree_iter() if out_degree == 0 and causal_graph.in_degree(node) > 0 ))

def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=False):
    x_data = {str(k): data_set['X'][k] for k in range(len(data_set['X']))}

    return tf.estimator.inputs.pandas_input_fn(       
            x=pd.DataFrame(x_data),
            y = pd.Series(data_set['Y']),       
            batch_size=n_batch,          
            num_epochs=num_epochs,       
            shuffle=shuffle)

def get_predict_fn(data_set, num_epochs=None, n_batch = 128, shuffle=False):
    x_data = {str(k): np.array([data_set[k]]) for k in range(len(data_set))}

    return tf.estimator.inputs.pandas_input_fn(       
            x=pd.DataFrame(x_data),
            batch_size=n_batch,          
            num_epochs=num_epochs,       
            shuffle=shuffle)


# for episode in range(num_episodes):
#     env.reset()

#     for t in range(time_frame):
#         # Get random action (for now)
#         action = env.action_space.sample()

#         next_state, reward, done, terminated, info = env.step(action)
#         print(t, next_state, reward, done, info, action)
        
#         if done:
#             break