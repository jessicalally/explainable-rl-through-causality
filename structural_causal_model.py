import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import logging

# SCM with actions represented explicitly as a separate state variable


class StructuralCausalModel:
    def __init__(self, env, data_set):
        self.causal_graph = env.causal_graph
        self.structural_equations = self._initialise_structural_equations(
            env, data_set
        )

    def _convert_action_set_to_1_hot_encoding(self, action_set):
        action_set = action_set.astype(int)

        encoding = np.zeros((action_set.size, action_set.max() + 1))
        encoding[np.arange(action_set.size), action_set] = 1

        return encoding

    def _initialise_structural_equations(self, env, data_set):
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

            if node == env.state_space:
                y_data = data_set[:, node].astype(int)

                classifier = tf.estimator.DNNClassifier(
                    n_classes=6,
                    feature_columns=x_feature_cols,
                    model_dir='scm_models/linear_classifier/' +
                    str(node),
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
                y_data = data_set[:, node]

                lr = tf.estimator.LinearRegressor(
                    feature_columns=x_feature_cols,
                    model_dir='scm_models/linear_regressor/' + str(node))

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
