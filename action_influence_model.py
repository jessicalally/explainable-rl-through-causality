import tensorflow.compat.v1 as tf
import numpy as np

class ActionInfluenceModel:
    def __init__(self, causal_graph):
        self.causal_graph = causal_graph
        self.structural_equations = self.initialise_structural_equations()

    def _initialise_structural_equations(self):
        state_set = []
        action_set = []
        next_state_set = []

        for (s, a, _, s_next) in data_set: 
            state_set.append(s)
            action_set.append(a)
            next_state_set.append(s_next)

        action_influence_dataset = {}

        for idx, action in enumerate(action_set):
            if action in action_influence_dataset:
                action_influence_dataset[action]['state'].append(state_set[idx])
                action_influence_dataset[action]['next_state'].append(next_state_set[idx])
            else:
                action_influence_dataset[action] = {'state' : [], 'next_state': []}
                action_influence_dataset[action]['state'].append(state_set[idx])
                action_influence_dataset[action]['next_state'].append(next_state_set[idx])

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

    # Parameters:
    # training_data: Tuples of the form (state, action, next_state)
    def train_structural_equations(self, training_data):
        for key in self.structural_equations:
            self.structural_equations[key]['function'].train(input_fn=self.get_input_fn(self.structural_equations[key],                                       
                                                                                num_epochs=None,                                      
                                                                                n_batch = 128,                                      
                                                                                shuffle=False),                                      
                                                                                steps=1000)