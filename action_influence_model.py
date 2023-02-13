import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd

class ActionInfluenceModel:
    def __init__(self, causal_graph, data_set):
        self.causal_graph = causal_graph
        self.structural_equations = self._initialise_structural_equations(data_set)

    def _initialise_structural_equations(self, data_set):
        state_set = []
        action_set = []
        next_state_set = []

        for (s, a, _, s_next) in data_set: 
            state_set.append(s)
            action_set.append(a)
            next_state_set.append(s_next)

        action_influence_dataset = self._generate_action_influence_dataset(
            state_set, 
            action_set, 
            next_state_set, 
            data_set
        )

        structural_equations = {}
        unique_functions = {}

        for edge in self.causal_graph.edges():
            for preds in self.causal_graph.predecessors(edge[1]):
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


    def train(self):        
        print("Starting SCM training...")
        self._train_structural_equations()
        print('Ending SCM training...')

    # Parameters:
    def _train_structural_equations(self):
        for key in self.structural_equations:
            self.structural_equations[key]['function'].train(input_fn=self.get_input_fn(self.structural_equations[key],                                       
                                                                                num_epochs=None,                                      
                                                                                n_batch = 128,                                      
                                                                                shuffle=False),                                      
                                                                                steps=1000)

    def _generate_action_influence_dataset(self, state_set, action_set, next_state_set, data_set):
        action_influence_dataset = {}

        for idx, action in enumerate(action_set):
            if action in action_influence_dataset:
                action_influence_dataset[action]['state'].append(state_set[idx])
                action_influence_dataset[action]['next_state'].append(next_state_set[idx])
            else:
                action_influence_dataset[action] = {'state' : [], 'next_state': []}
                action_influence_dataset[action]['state'].append(state_set[idx])
                action_influence_dataset[action]['next_state'].append(next_state_set[idx])

        return action_influence_dataset                                                               

    def predict_from_scm_given_action(self, structural_equations, s, a):
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

            pred = structural_equations[key]['function'].predict(input_fn=self.get_predict_fn(x_data,                          
                        num_epochs=1,                          
                        n_batch = 128,                          
                        shuffle=False))

            for item in pred:
                predicted_state[node] = item['predictions'][0]

        return predicted_state

    def predict_from_scm(self, structural_equations, s):
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

            pred = structural_equations[key]['function'].predict(input_fn=self.get_predict_fn(x_data,                          
                    num_epochs=1,                          
                    n_batch = 128,                          
                    shuffle=False))

            predict_y[key] = np.array([item['predictions'][0] for item in pred])

        return predict_y

    def process_explanations(self, structural_equations, state_set, action_set, next_state_set):
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


    def generate_why_explanations(self, data_set, actual_state, actual_action, state_num_in_batch, causal_graph, structural_equations):
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


    def generate_counterfactual_explanations(self, actual_state, actual_action, counterfactual_action, state_num_in_batch, causal_graph, structural_equations):
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


    def predict_node_scm(self, node, action, structural_equations):
        key = (node, action)

        pred = structural_equations[key]['function'].predict(input_fn=get_input_fn(structural_equations[key],                          
                    num_epochs=1,                          
                    n_batch = 128,                          
                    shuffle=False))

        result = np.array([item['predictions'][0] for item in pred])

        return result
        
    def test_predict_from_scm(self, structural_equations):
        result = predict_from_scm(structural_equations, [0, 0, 0, 0])
        print(result)
        
    """minimally complete tuple = (head node of action, immediate pred of sink nodes, sink nodes)"""
    def get_minimally_complete_tuples(self, chains, state):
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

    def get_minimal_contrastive_tuples(self, actual_chain, counterfactual_chain, actual_state, counterfactual_state):

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


    def get_causal_chains(self, sink_nodes, action_edge_list, causal_graph):
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


    def get_action_chains(self, action, chain_lists_of_action, causal_graph):
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


    def get_edges_of_actions(self, action, causal_graph):
        return list(edge for edge in causal_graph.edges(data=True) if edge[2]['action'] == action)
    
    def get_sink_nodes(self, causal_graph):
        return list((node for node, out_degree in causal_graph.out_degree_iter() if out_degree == 0 and causal_graph.in_degree(node) > 0 ))

    def get_input_fn(self, data_set, num_epochs=None, n_batch = 128, shuffle=False):
        x_data = {str(k): data_set['X'][k] for k in range(len(data_set['X']))}

        return tf.estimator.inputs.pandas_input_fn(       
                x=pd.DataFrame(x_data),
                y = pd.Series(data_set['Y']),       
                batch_size=n_batch,          
                num_epochs=num_epochs,       
                shuffle=shuffle)

    def get_predict_fn(self, data_set, num_epochs=None, n_batch = 128, shuffle=False):
        x_data = {str(k): np.array([data_set[k]]) for k in range(len(data_set))}

        return tf.estimator.inputs.pandas_input_fn(       
                x=pd.DataFrame(x_data),
                batch_size=n_batch,          
                num_epochs=num_epochs,       
                shuffle=shuffle)
            