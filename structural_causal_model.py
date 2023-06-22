import tensorflow.compat.v1 as tf
import networkx as nx
import numpy as np
import pandas as pd


class StructuralCausalModel:
    def __init__(
            self,
            env,
            rl_agent,
            data_set,
            learned_causal_graph=None,
            is_reward_scm=False,
            uses_true_dag=False):
        self.env = env
        self.rl_agent = rl_agent
        self.is_reward_scm = is_reward_scm
        self.uses_true_dag = uses_true_dag

        if learned_causal_graph is not None:
            self.causal_graph = learned_causal_graph
        else:
            # Use true DAG
            self.causal_graph = env.causal_graph

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
                x_data.append((data_set[:, [x_feature]]).flatten())

            x_feature_cols = [
                tf.feature_column.numeric_column(
                    key=str(i)) for i in range(
                    len(x_data))]

            if node == self.env.action_node and not self.is_reward_scm:
                # Use a DNNClassifier for modelling action causal relationships
                y_data = data_set[:, node].astype(int)

                classifier = tf.estimator.DNNClassifier(
                    n_classes=self.env.action_space,
                    feature_columns=x_feature_cols,
                    model_dir='scm_models/' +
                    f'{self.env.name}-{self.rl_agent.name}-{self.uses_true_dag}' +
                    '/linear_classifier/' +
                    str(node) +
                    str(
                        self.is_reward_scm),
                    hidden_units=[
                        64,
                        128,
                        64,
                        32],
                    dropout=0.2,
                )

                structural_equations[node] = {
                    'X': x_data,
                    'Y': y_data,
                    'function': classifier,
                    'type': 'action'}
            else:
                # Use a linear regressor for all other causal relationships
                y_data = data_set[:, node]

                lr = tf.estimator.LinearRegressor(
                    feature_columns=x_feature_cols,
                    model_dir='scm_models/' +
                    f'{self.env.name}-{self.rl_agent.name}-{self.uses_true_dag}' +
                    '/linear_regressor/' +
                    str(node) +
                    str(
                        self.is_reward_scm))

                structural_equations[node] = {
                    'X': x_data, 'Y': y_data, 'function': lr, 'type': 'state'}

        return structural_equations

    def train(self):
        print("Starting SCM training...")
        self._train_structural_equations()
        print('Ending SCM training...')

    def _train_structural_equations(self):
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

    def predict_from_scm(self, test_data, ignore_action=False):
        predict_y = {}

        for node in self.structural_equations:
            predecessors = self.causal_graph.predecessors(node)

            x_data = test_data[list(predecessors)]
            pred = self.structural_equations[node]['function'].predict(
                input_fn=self.get_predict_fn(
                    x_data, num_epochs=1, n_batch=128, shuffle=False))

            if self.structural_equations[node]['type'] == 'state':
                predict_y[node] = np.array(
                    [item['predictions'][0] for item in pred])
            else:
                assert (self.structural_equations[node]['type'] == 'action')

                if not ignore_action:
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

    # Generate all the causal chains between each pair of head and sink nodes
    def get_causal_chains(self, head_nodes, sink_nodes, causal_graph):
        action_node = self.env.state_space

        all_causal_chains = []

        for head_node in head_nodes:
            for sink_node in sink_nodes:
                all_chains_between_nodes = nx.all_simple_paths(
                    causal_graph,
                    source=head_node,
                    target=sink_node
                )

                # Ignore causal chains that contain the action: since we have
                # made the assumption that the action depends on all nodes in the
                # previous state, and the action influences all nodes in the next
                # state, the causal chains containing the action node do not
                # provide any additional useful information
                all_chains_between_nodes = [
                    chain for chain in all_chains_between_nodes if action_node not in chain]
                all_causal_chains.extend(all_chains_between_nodes)

        return all_causal_chains

    def get_sink_nodes(self, causal_graph):
        return list(
            (node for node, out_degree in causal_graph.out_degree() if out_degree == 0))
