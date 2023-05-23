from collections import deque
import networkx as nx
import numpy as np
import pandas as pd
import torch


class ExplanationGenerator():
    def __init__(self, env, trained_scm, trained_reward_scm, trained_rl_agent):
        self.env = env
        self.scm = trained_scm
        self.reward_scm = trained_reward_scm
        self.rl_agent = trained_rl_agent

    # Why was [action] performed in this [state]?
    #
    # Parameters:
    # state [just datapoints at time t]
    # action chosen at time t
    def generate_why_explanation(self, state, action, pertubation):
        explanation = f'{self.env.actions[int(action)]}.\n'

        # These are all the nodes that influence the action decision - which is right because we want
        # to measure their feature importance in terms of choosing the action
        # But probably this should really all nodes of indegree=0 which are connected to the action
        # (since a node can affect a node that impacts the action decision)
        # Ahh but we've made an assumption that all nodes affect the action decision :O
        # So we want the predecessors with indegree=0?
        # Nah we want all the causal chains, its just that some of these start
        # at nodes with indegree/=0 but that's ok

        # These are all the nodes that have out-degree=0
        sink_nodes = self._get_sink_nodes(self.scm.causal_graph)
        action_node = self.env.action_node

        # Generate the causal chains for a single timestep
        head_nodes = [node for node in self.scm.causal_graph.predecessors(action_node)]
        print(f'head nodes {head_nodes}')
        print(f'sink nodes {sink_nodes}')

        one_step_causal_chains = self._get_one_step_causal_chains(
            head_nodes, sink_nodes, self.scm.causal_graph)
        
        print(f'one step causal chains {one_step_causal_chains}')
        # TODO: how do we combine the causal chains so that they all end up at the reward?
        # In our causal graph, we want to put a connection from each node at time t+1 back to time t
        # all_simple_paths has an optional cutoff parameter for cutting chains off at a certain length
        # But we want to do this by marking edges as visited instead? So that we don't use the same relationships
        # twice in explanation

        multistep_causal_chains = self._generate_multistep_causal_chains(
            one_step_causal_chains)
        print(f'multistep causal chains {multistep_causal_chains}')

        # Predict the values of all nodes using the trained structural
        # equations
        datapoint = np.zeros((self.scm.env.state_space * 2) + 1)
        for idx, val in enumerate(state):
            datapoint[idx] = val

        datapoint[self.scm.env.action_node] = action
        print(f"datapoint {datapoint}")
        predicted_nodes = self.scm.predict_from_scm(datapoint)
        print(f'predicted nodes {predicted_nodes}')

        # Get the causal chains with the head node of the most important feature - 
        # we do this in order of importance in case the most important feature has
        # no detected causal chains (although this is unlikely)
        importance_vector = self._estimate_q_function_feature_importance(
            state, pertubation)
        
        features_by_importance = np.flip(np.argsort(importance_vector))
        print(f'features ordered by importance {features_by_importance}')
        causal_chains = []

        for feature in features_by_importance:  
            # Get all causal chains with this feature as head - we want to use
            # these as explanation
            feature_causal_chains = [
                chain for chain in multistep_causal_chains if chain[0][0] == feature]
            
            print(f'feature {feature} : {feature_causal_chains}')
            
            if len(feature_causal_chains) > 0:
                causal_chains = feature_causal_chains
                break

        if len(causal_chains) == 0:
            return "Error: no appropiate causal chains found"

        # Get all nodes that are immediately affected by the current action
        imm_nodes = {chain[0][1] for chain in causal_chains}
        print(f'imm nodes {imm_nodes}')

        # Get diff between current node value and predicted node value for
        # the next node in the causal chain

        for idx, imm_node in enumerate(imm_nodes):
            curr_node_value = datapoint[imm_node -
                                        (self.scm.env.state_space + 1)]
            predicted_node_value = predicted_nodes[imm_node]
            diff = predicted_node_value - curr_node_value
            # TODO: can the diff be 0
            direction = 'increase' if diff > 0 else 'decrease'

            if idx == 0:
                explanation += f'To {direction} the value of {self.env.features[imm_node]} (from {curr_node_value:3.3f} to {predicted_node_value[0]:3.3f}) '
            else:
                explanation += f'and {direction} the value of {self.env.features[imm_node]} (from {curr_node_value:3.3f} to {predicted_node_value[0]:3.3f}) '

        explanation += 'in the next time step.\n Because: '

        # TODO: do feature influence the reward jointly or independently? How might we be able to tell?
        # Can we look at the weightings of the linear regression model?

        # TODO: problems occur with the explanations with links between nodes at the same time step

        predecessors_to_reward = [
            node for node in 
            self.reward_scm.causal_graph.predecessors(self.env.state_space)
        ]

        for causal_chain in causal_chains:
            print(f'causal chain {causal_chain}')
            for i, step in enumerate(causal_chain):
                print(f'step {step}')
                # TODO: replace with idx to reward node
                for j, node in enumerate(step):
                    # Always skip j == 0
                    if j == 0:
                        continue
                    # TODO: it may be easier to change every node of the same feature to the same value at some point
                    if node % (self.env.state_space + 1) in predecessors_to_reward:
                        # This must be the last node of the step, and the last step in the chain
                        explanation += (f'\n{self.env.features[node]} influences the reward.'.capitalize())
                        break
                    if j == 1 and i == 0:
                        explanation += (f'\n{self.env.features[node]} influences '.capitalize())
                    else:
                        explanation += f'{self.env.features[node]}, which influences '

        # TODO: do we add something here about the predicted impact on the reward?

        # pd.DataFrame.from_dict(
        #     data=set(explanation),
        #     orient='index').to_csv(
        #     f'why_explanations_{self.scm.env.name}_{self.rl_agent.name}.csv',
        #     mode='a',
        #     header=False)

        return explanation

    # Why was [counter_action] not taken at [state]?
    #
    # Parameters:
    # state [just datapoints at time t]
    # actual action chosen at time t
    # counterfactual action to be taken at time t

    def generate_why_not_explanation(self, state, action, counter_action):
        explanation = f'Because it is more desirable to do {self.env.actions[int(action)]}.\n'

        # These are all the nodes that have out-degree=0
        sink_nodes = self._get_sink_nodes(self.scm.causal_graph)
        action_node = self.env.action_node

        # Generate the causal chains for a single timestep
        head_nodes = self.scm.causal_graph.predecessors(action_node)
        print(f'head nodes {head_nodes}')
        one_step_causal_chains = self._get_one_step_causal_chains(
            head_nodes, sink_nodes, self.scm.causal_graph)
        print(f'one step causal chains {one_step_causal_chains}')

        causal_chains = self._generate_multistep_causal_chains(
            one_step_causal_chains)
        print(f'multistep causal chains {causal_chains}')

        # # Predict the values of all nodes using the trained structural
        # # equations
        # datapoint = np.zeros((self.scm.env.state_space * 2) + 1)
        # for idx, val in enumerate(state):
        #     datapoint[idx] = val

        # datapoint[self.scm.env.state_space] = action
        # print(f"datapoint {datapoint}")
        # predicted_nodes = self.scm.predict_from_scm(datapoint)
        # print(f'predicted nodes {predicted_nodes}')

        # # Predict the values of all nodes with the counterfactual action using
        # # the trained SCM
        # counter_datapoint = np.zeros((self.scm.env.state_space * 2) + 1)
        # for idx, val in enumerate(state):
        #     counter_datapoint[idx] = val

        # counter_datapoint[self.scm.env.state_space] = counter_action
        # print(f"counter datapoint {counter_datapoint}")
        # predicted_counter_nodes = self.scm.predict_from_scm(counter_datapoint)
        # print(f'predicted_counter_nodes {predicted_counter_nodes}')

        # most_importance_feature = self._estimate_q_function_feature_importance(
        #     state)
        # print(f"most important feature {most_importance_feature}")

        # print(f'multistep causal chains {causal_chains}')

        # # Get all causal chains with this feature as head - we want to use
        # # these as explanation
        # causal_chains = [
        #     chain for chain in causal_chains if chain[0][0] == most_importance_feature]
        # print(f'relevant causal chains {causal_chains}')

        # # Get all nodes that are immediately affected by the current action
        # imm_nodes = {chain[0][1] for chain in causal_chains}
        # print(f'imm nodes {imm_nodes}')

        # # Get diff between current node value and predicted node value for
        # # the next node in the causal chain

        # for idx, imm_node in enumerate(imm_nodes):
        #     curr_node_value = datapoint[imm_node -
        #                                 (self.scm.env.state_space + 1)]
        #     predicted_node_value = predicted_nodes[imm_node][0]
        #     predicted_counter_node_value = predicted_counter_nodes[imm_node][0]
        #     diff = predicted_node_value - curr_node_value
        #     # TODO: can the diff be 0
        #     direction = 'increase' if diff > 0 else 'decrease'

        #     if idx == 0:
        #         explanation += f'In order to {direction} the value of {self.env.features[imm_node]} (from {curr_node_value:3.3f} to {predicted_node_value:3.3f}) (counterfactual {predicted_counter_node_value:3.3f}) '
        #         # TODO: add the rather than case?
        #     else:
        #         explanation += f'and {direction} the value of {self.env.features[imm_node]} (from {curr_node_value:3.3f} to {predicted_node_value:3.3f}) (counterfactual {predicted_counter_node_value:3.3f}) '
        #         # TODO: add the rather than case?

        # # TODO: we would like to learn which variables affect the reward, and
        # # by roughly how much
        # explanation += 'in the next time step.\n Because:'

        # for causal_chain in causal_chains:
        #     print(f'causal chain {causal_chain}')
        #     for idx, step in enumerate(causal_chain):
        #         print(f'step {step}')
        #         # TODO: replace with index to reward node
        #         if step[-1] == self.env.reward_node:
        #             if idx == 0:
        #                 explanation += f'\n{self.env.features[step[-1]]} influences the reward.'
        #             else:
        #                 explanation += f'{self.env.features[step[-1]]}, which influences the reward.'
        #         elif idx == 0:
        #             explanation += f'\n{self.env.features[step[-1]]} influences '
        #         else:
        #             explanation += f'{self.env.features[step[-1]]}, which influences '

        return explanation

        # pd.DataFrame.from_dict(
        #     data=set(explanation),
        #     orient='index').to_csv(
        #     f'why_not_explanations_{self.scm.env.name}_{self.rl_agent.name}.csv',
        #     mode='a',
        #     header=False)

    def _get_sink_nodes(self, causal_graph):
        return list(
            (node for node, out_degree in causal_graph.out_degree() if out_degree == 0))

    # Generates all the causal chains for a single timestep

    def _get_one_step_causal_chains(
            self,
            head_nodes,
            sink_nodes,
            causal_graph):
        action_node = self.scm.env.action_node

        all_causal_chains = []

        # Get all causal chains in the graph that do not contain the action
        # node
        for head_node in head_nodes:
            for sink_node in sink_nodes:
                all_chains_between_nodes = nx.all_simple_paths(
                    causal_graph,
                    source=head_node,
                    target=sink_node
                )

                # We want all the causal chains that don't contain the action node, as these
                # nodes have a causal effect on future nodes?
                all_chains_between_nodes = [
                    chain for chain in all_chains_between_nodes if action_node not in chain]
                
                # Take all chains where the successor node is in the future
                all_chains_between_nodes = [
                    chain for chain in all_chains_between_nodes if chain[1] > self.env.action_node
                ]
                
                all_causal_chains.extend(all_chains_between_nodes)

        return all_causal_chains

    def _generate_multistep_causal_chains(self, one_step_causal_chains):
        multi_step_causal_chains = []
        
        for chain in one_step_causal_chains: 
            subchains = self._get_subchains_that_influence_reward(chain)
            for subchain in subchains:
                multi_step_causal_chains.append([subchain])

            else:
                q = deque([[chain]])

                while len(q) > 0:
                    curr_chain = q.pop()
                    print(f"curr chain {curr_chain}")

                    # Find all chains that begin with the last node
                    poss_next_chains = [
                        next_chain for next_chain in one_step_causal_chains
                        if next_chain[0] == chain[-1] % (self.scm.env.state_space + 1)
                        and next_chain not in curr_chain # prevents cycles - we don't want to duplicate chains since this information is already given to the user
                    ]

                    # If there are no possible unused chains then this chain will
                    # just be removed from the deque
                    for poss_next_chain in poss_next_chains:
                        new_chain = curr_chain
                        new_chain.append(poss_next_chain)

                        subchains = self._get_subchains_that_influence_reward(chain)
                        for subchain in subchains:
                            multi_step_causal_chains.append([subchain])

                        else:
                            q.append(new_chain)

        return multi_step_causal_chains
    
    def _get_subchains_that_influence_reward(self, chain):
        predecessors_to_reward = [
            node for node in 
            self.reward_scm.causal_graph.predecessors(self.env.state_space)
        ]

        subchains = []

        for i in range(1, len(chain)):
            if (chain[i] % (self.env.state_space + 1)) in predecessors_to_reward:
                subchains.append(chain[:i+1]) # TODO: i or i+1

        return subchains

    def _estimate_q_function_feature_importance(self, state, pertubation):
        min_pertubation = 0.001
        # TODO: should the pertubation be relative to the state feature bounds as well?? So that we change each feature by the same percentage
        # Without accounting for the state values, pertubation almost always causes the pole angular velocity to be the most importat state variable
        # and sometimes the pole angle
        # State feature bounds might not be possible because some of the bounds are infinite
        # TODO: what about doing it relative to the current state variable value?
        # pertubation should be 0.01 for continuous features, and smallest unit
        # for discrete features, as in paper
        q_values = self.rl_agent.get_q_values(state)
        action = np.argmax(q_values)
        state_tensor = torch.DoubleTensor(state).unsqueeze(0)

        importance_vector = np.full(state_tensor.shape[1], q_values[action])
        print(f'importance vector {importance_vector}')

        # Apply small pertubation to each state variable, and recalculate the
        # q_values
        for i in range(len(state)):
            pertubated_state = state
            # Applying a 1% pertubation
            pertubated_state_value = max(min_pertubation, pertubated_state[i] * (1.0 + pertubation))
            pertubated_state[i] = pertubated_state_value
            print(f'pertubated state {pertubated_state}')
            updated_q_values = self.rl_agent.get_q_values(pertubated_state)
            print(f'updated q values {updated_q_values}')
            importance_vector[i] = (
                abs(updated_q_values[action] - importance_vector[i]) / pertubation)

        print(f"importance vector {importance_vector}")

        return importance_vector

    # It is not really possible to estimate the magnitude of the change when
    # the actions are discrete.
    # So it is just returning the index of any state variable that changes the
    # action

    def _estimate_action_feature_importance(self, state, pertubation=0.1):
        action = self.rl_agent.get_optimal_action(state)
        importance_vector = np.zeros(state.shape)

        # Apply small pertubation to each state variable, and recalculate the
        # q_values
        for i in range(len(state)):
            pertubated_state = state
            pertubated_state[i] += pertubation
            new_action = self.rl_agent.get_optimal_action(pertubated_state)

            importance_vector[i] = 1 if new_action != action else 0

        print(f'action importance vector {importance_vector}')

        # Picks a state variable arbitrarily that affects the chosen action
        return np.where(importance_vector == 1)
