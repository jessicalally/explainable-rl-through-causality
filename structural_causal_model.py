import explanation_templates as explanations
import tensorflow.compat.v1 as tf
import networkx as nx
import numpy as np
import pandas as pd
# import logging

# SCM with actions represented explicitly as a separate state variable


class StructuralCausalModel:
    def __init__(self, env, data_set, learned_causal_graph=None):
        self.env = env

        if learned_causal_graph is not None:
            self.causal_graph = learned_causal_graph
        else:
            self.causal_graph = env.causal_graph # true DAG

        self.structural_equations = self._initialise_structural_equations(
            data_set
        )

    def _convert_action_set_to_1_hot_encoding(self, action_set):
        action_set = action_set.astype(int)

        encoding = np.zeros((action_set.size, action_set.max() + 1))
        encoding[np.arange(action_set.size), action_set] = 1

        return encoding

    def _initialise_structural_equations(self, data_set):
        structural_equations = {}
        unique_functions = {}

        for edge in self.causal_graph.edges():
            node = edge[1]

            for predecessor in self.causal_graph.predecessors(edge[1]):
                if node not in unique_functions:
                    unique_functions[node] = set()

                unique_functions[node].add(predecessor)

        for node in unique_functions:
            predecessors = list(unique_functions[node])

            x_data = []
            for x_feature in predecessors:
                # instead of x_feature we want X-feature AND key[0]
                x_data.append((data_set[:, [x_feature]]).flatten())

            x_feature_cols = [
                tf.feature_column.numeric_column(
                    key=str(i)) for i in range(
                    len(x_data))]

            if node == self.env.state_space:
                y_data = data_set[:, node].astype(int)

                classifier = tf.estimator.DNNClassifier(
                    n_classes=self.env.action_space,
                    feature_columns=x_feature_cols,
                    model_dir='scm_models/' + str(self.env.name) + '/linear_classifier/' + str(node),
                    hidden_units=[64, 128, 64, 32],
                    dropout=0.2,
                )

                structural_equations[node] = {
                    'X': x_data,
                    'Y': y_data,
                    'function': classifier,
                    'type': 'action'}
            else:
                y_data = data_set[:, node]

                lr = tf.estimator.LinearRegressor(
                    feature_columns=x_feature_cols,
                    model_dir='scm_models/' + str(self.env.name) + '/linear_regressor/' + str(node))

                structural_equations[node] = {
                    'X': x_data, 'Y': y_data, 'function': lr, 'type': 'state'}

        return structural_equations

    def train(self):
        print("Starting SCM training...")
        self._train_structural_equations()
        print('Ending SCM training...')

    def _train_structural_equations(self):
        # logging.getLogger().setLevel(logging.INFO)

        for node in self.structural_equations:
            self.structural_equations[node]['function'].train(
                input_fn=self.get_input_fn(
                    self.structural_equations[node],
                    num_epochs=None,
                    n_batch=128,
                    shuffle=True),
                steps=1000)

    def get_input_fn(
            self,
            data_set,
            num_epochs=None,
            n_batch=128,
            shuffle=False):
        x_data = {str(k): data_set['X'][k] for k in range(len(data_set['X']))}

        return tf.estimator.inputs.pandas_input_fn(
            x=pd.DataFrame(x_data),
            y=pd.Series(data_set['Y']),
            batch_size=n_batch,
            num_epochs=num_epochs,
            shuffle=shuffle)

    def predict_from_scm(self, structural_equations, test_data):
        predict_y = {}

        for node in structural_equations:
            predecessors = self.causal_graph.predecessors(node)

            x_data = test_data[list(predecessors)]
            pred = structural_equations[node]['function'].predict(
                input_fn=self.get_predict_fn(
                    x_data, num_epochs=1, n_batch=128, shuffle=False))

            if structural_equations[node]['type'] == 'state':
                predict_y[node] = np.array(
                    [item['predictions'][0] for item in pred])
            else:
                assert (structural_equations[node]['type'] == 'action')

                predict_y[node] = np.array(
                    [np.argmax(item['probabilities']) for item in pred])

        return predict_y

    def get_predict_fn(
            self,
            data_set,
            num_epochs=None,
            n_batch=128,
            shuffle=False):

        x_data = {str(k): np.array([data_set[k]])
                  for k in range(len(data_set))}

        return tf.estimator.inputs.pandas_input_fn(
            x=pd.DataFrame(x_data),
            batch_size=n_batch,
            num_epochs=num_epochs,
            shuffle=shuffle)
    
    def process_explanations(
            self,
            data_set):
        print('Processing explanations...')
        why_explanations = {}
        why_not_explanations = {}

        for agent_step in range(0, 1, 10):
            print(str(agent_step) + "/" + str(len(data_set)))

            datapoint = data_set[agent_step]
            print(f'datapoint {datapoint}')
            action = datapoint[self.env.state_space]

            why_explanations[(agent_step,
                            action)] = {'state': datapoint[:self.env.state_space],
                                        'why_exps': self.generate_why_explanations(
                                            datapoint,
                                            self.env.causal_graph,
                                            self.structural_equations
                                        )}
            
            # Generate Why not B? counterfactual questions
            # Get all possible counterfactual actions
            poss_counter_actions = set(range(0, self.env.action_space)).difference({action})
            print(f'possible counter actions = {poss_counter_actions}')

            for counter_action in poss_counter_actions:
                why_not_explanations[(agent_step, action, counter_action)] = {'state': datapoint[:self.env.state_space], 
                                                        'why_not_exps': self.generate_counterfactual_explanations(self.env.causal_graph, self.structural_equations, datapoint, action, counter_action)}

        pd.DataFrame.from_dict(
            data=why_explanations,
            orient='index').to_csv(
            'why_explanations_taxi.csv',
            mode='a',
            header=False)
        
        pd.DataFrame.from_dict(
            data=why_not_explanations,
            orient='index').to_csv(
            'why_not_explanations_taxi.csv',
            mode='a',
            header=False)
        
        return why_explanations, why_not_explanations

    def generate_why_explanations(
            self,
            datapoint,
            causal_graph,
            structural_equations):
        why_exps = set()
        actual_state = datapoint[:self.env.state_space]
        actual_action = datapoint[self.env.state_space]
        actual_next_state = datapoint[self.env.state_space+1:]

        # Treat sink nodes as goal nodes
        sink_nodes = self.get_sink_nodes(causal_graph)
        print(f'sink nodes: {sink_nodes}')

        # Edges of action, doesn-t apply to this scm
        # actual_action_edge_list = self.get_edges_of_actions(
        #     actual_action, causal_graph)
        
        # all_actual_causal_chains_from_action = self.get_causal_chains(
        #     sink_nodes, actual_action_edge_list, causal_graph)

        # action_chain_list = self.get_action_chains(
        #     actual_action, all_actual_causal_chains_from_action, causal_graph)


        # TODO: How to generate causal chains :)

        # Get predecessors to the action variable - these are the head nodes
        action_node = self.env.state_space # TODO: add something to the environment to make this relationship clearer
        # TODO: actually are the head nodes just after the action?
        head_nodes = self.causal_graph.predecessors(action_node)

        # Get all causal chains between the predecessors and the sink nodes - these are the intermediate nodes
        # For a causal chain to exist, there needs to be a causal relationship that doesn't pass through the action variable
        # Since we assume the action depends on every state variable at time t, and affects every state variable at time t+1
        # Go through all the causal chains from the head nodes, skipping it if it contains the action variable
        causal_chains = self.get_causal_chains(head_nodes, sink_nodes, causal_graph)

        # Predict the values of state variables in these causal chains using the SCM at the action taken (Why A?)
        predicted_nodes = self.predict_from_scm(structural_equations, datapoint)

        # Generate minimally complete tuples

        optimal_transition = np.zeros(len(datapoint))
        noop_transition = np.zeros(len(datapoint))

        for node in range(self.env.state_space):
            optimal_transition[node] = datapoint[node]
            optimal_transition[node + self.env.state_space + 1] = datapoint[node]
            noop_transition[node] = datapoint[node]
            noop_transition[node + self.env.state_space + 1] = datapoint[node]

        # Set action
        optimal_transition[self.env.state_space] = actual_action
        noop_transition[self.env.state_space] = actual_action

        for causal_chain in causal_chains:
            # Get predicted values for all nodes in the causal chain
            for node in causal_chain[1:]:
                # All nodes excluding the head nodes
                optimal_transition[node] = predicted_nodes[node]

        # This should be similar to the actual next sate
        print(f'noop_transition = {noop_transition}')
        print(f'optimal state = {optimal_transition}')

        for causal_chain in causal_chains:
            min_tuple_optimal_transition = self.get_minimally_complete_tuples(
                causal_chain, optimal_transition)
            
            # What would have happened if no action had been performed
            min_tuple_noop_transition = self.get_minimally_complete_tuples(
                causal_chain, noop_transition)
                
                # TODO: we want to remove those tuples with state variables between t and t+1
                # that actually did not change, so we can have just those action explanations
                # for variables that changed!

                # Generate explanation text

            explanation = explanations.taxi_generate_why_text_explanations(
                min_tuple_noop_transition,
                min_tuple_optimal_transition, actual_action)
            
            print(explanation + "\n")
            why_exps.add(explanation)

        return why_exps
    
    def generate_counterfactual_explanations(self, causal_graph, structural_equations, datapoint, actual_action, counter_action):
        action_node = self.env.state_space
        sink_nodes = self.get_sink_nodes(causal_graph)
        head_nodes = self.causal_graph.predecessors(action_node)

        causal_chains = self.get_causal_chains(head_nodes, sink_nodes, causal_graph)

        optimal_transition = np.zeros(len(datapoint))
        counterfactual_transition = np.zeros(len(datapoint))

        # Set known values (from time t)
        for node in range(self.env.state_space):
            optimal_transition[node] = datapoint[node]
            # optimal_transition[node + self.env.state_space + 1] = datapoint[node]
            counterfactual_transition[node] = datapoint[node]

        # Set action
        optimal_transition[self.env.state_space] = actual_action
        counterfactual_transition[self.env.state_space] = counter_action

        predicted_optimal_nodes = self.predict_from_scm(structural_equations, optimal_transition)
        predicted_counterfactual_nodes = self.predict_from_scm(structural_equations, counterfactual_transition)

        for causal_chain in causal_chains:
            # Get predicted values for all nodes in the causal chain
            for node in causal_chain[1:]:
                # All nodes excluding the head nodes
                optimal_transition[node] = predicted_optimal_nodes[node]
                counterfactual_transition[node] = predicted_counterfactual_nodes[node]

        print(f'optimal transition {optimal_transition}')
        print(f'counter transition {counterfactual_transition}')
         
        contrastive_exp = set()

        contrastive_tuple = self.get_minimal_contrastive_tuples(causal_chain, optimal_transition, counterfactual_transition)
        explanation = explanations.taxi_generate_contrastive_text_explanations(contrastive_tuple, actual_action)
        print(f'explanation {explanation}')
        contrastive_exp.add(explanation)    

        return contrastive_exp   

    """minimally complete tuple = (head node of action, immediate pred of sink nodes, sink nodes)"""

    def get_minimally_complete_tuples(self, chain, state):
        head = set()
        immediate = set()
        reward = set()

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
    
    def get_minimal_contrastive_tuples(self, causal_chain, optimal_transition, counterfactual_transition):
        actual_minimally_complete_tuple = self.get_minimally_complete_tuples(causal_chain, optimal_transition)
        counterfactual_minimally_complete_tuple = self.get_minimally_complete_tuples(causal_chain, counterfactual_transition)

        min_tuples = np.sum(np.array([list(k) for k in list(actual_minimally_complete_tuple.values())]))
        tuple_states = set([k[0] for k in min_tuples])

        counter_min_tuples = np.sum(np.array([list(k) for k in list(counterfactual_minimally_complete_tuple.values())]))
        counter_tuple_states = set([k[0] for k in counter_min_tuples])
        counter_tuple_states.difference_update(tuple_states)

        contrastive_tuple = {
                            'actual': {k: optimal_transition[k] for k in counter_tuple_states},
                            'counterfactual': {k: counterfactual_transition[k] for k in counter_tuple_states},
                            'reward': {k[0]: k[1] for k in actual_minimally_complete_tuple['reward']}
                            }
        
        return contrastive_tuple

    # Get all causal chains between the predecessors and the sink nodes - these are the intermediate nodes
    # For a causal chain to exist, there needs to be a causal relationship that doesn't pass through the action variable
    # Since we assume the action depends on every state variable at time t, and affects every state variable at time t+1
    # Go through all the causal chains from the head nodes, skipping it if it contains the action variable
    def get_causal_chains(self, head_nodes, sink_nodes, causal_graph):
        action_node = self.env.state_space

        all_causal_chains = []

        # Get all causal chains in the graph that do not contain the action node
        for head_node in head_nodes:
            for sink_node in sink_nodes:
                all_chains_between_nodes = nx.all_simple_paths(
                                causal_graph,
                                source=head_node,
                                target=sink_node
                            )
                                
                all_chains_between_nodes = [chain for chain in all_chains_between_nodes if not action_node in chain]
                all_causal_chains.extend(all_chains_between_nodes)

        print(f"all causal chains {all_causal_chains}")

        return all_causal_chains

    def get_sink_nodes(self, causal_graph):
        return list((node for node, out_degree in causal_graph.out_degree() if out_degree == 0))
