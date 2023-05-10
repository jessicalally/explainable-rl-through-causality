from collections import deque
import networkx as nx
import numpy as np

class ExplanationGenerator():
    def __init__(self, env, trained_scm, trained_rl_agent):
        self.env = env
        self.scm = trained_scm
        self.rl_agent = trained_rl_agent

    # Why was [action] performed in this [state]?
    #
    # Parameters:
    # state [just datapoints at time t]
    # action chosen at time t
    def generate_why_explanation(self, state, action):
        explanation = f'Do {self.env.actions[int(action)]}.\n'
        
        # These are all the nodes that influence the action decision - which is right because we want 
        # to measure their feature importance in terms of choosing the action
        # But probably this should really all nodes of indegree=0 which are connected to the action
        # (since a node can affect a node that impacts the action decision)
        # Ahh but we've made an assumption that all nodes affect the action decision :O
        # So we want the predecessors with indegree=0? 
        # Nah we want all the causal chains, its just that some of these start at nodes with indegree/=0 but that's ok

        # These are all the nodes that have out-degree=0
        sink_nodes = self._get_sink_nodes(self.scm.causal_graph)
        action_node = self.scm.env.state_space

        # Generate the causal chains for a single timestep
        head_nodes = self.scm.causal_graph.predecessors(action_node)
        one_step_causal_chains = self._get_one_step_causal_chains(head_nodes, sink_nodes, self.scm.causal_graph)
        print(f'one step causal chains {one_step_causal_chains}') 
        # TODO: how do we combine the causal chains so that they all end up at the reward?
        # In our causal graph, we want to put a connection from each node at time t+1 back to time t
        # all_simple_paths has an optional cutoff parameter for cutting chains off at a certain length
        # But we want to do this by marking edges as visited instead? So that we don't use the same relationships
        # twice in explanation

        causal_chains = self._generate_multistep_causal_chains(one_step_causal_chains)
        print(f'multistep causal chains {causal_chains}')               

        # Predict the values of all nodes using the trained structural equations
        datapoint = np.zeros((self.scm.env.state_space * 2) + 1)
        for idx, val in enumerate(state):
            datapoint[idx] = val
        
        datapoint[self.scm.env.state_space] = action
        print(f"datapoint {datapoint}")
        predicted_nodes = self.scm.predict_from_scm(datapoint)
        print(f'predicted nodes {predicted_nodes}')

        # TODO: measure relative feature importance of all nodes at time t towards the action decision
        # We need access here to the Q-value function of the RL agent algorithm, so something
        # from the RL agent must be passed into this class as well as the SCM
        most_importance_feature = self._estimate_most_important_feature(state)
        print(f"most important feature {most_importance_feature}")

        print(f'multistep causal chains {causal_chains}') 

        # Get all causal chains with this feature as head - we want to use these as explanation
        causal_chains = [chain for chain in causal_chains if chain[0][0] == most_importance_feature]
        print(f'relevant causal chains {causal_chains}')

        # Get all nodes that are immediately affected by the current action
        imm_nodes = {chain[0][1] for chain in causal_chains}
        print(f'imm nodes {imm_nodes}')

        # Get diff between current node value and predicted node value for
        # the next node in the causal chain

        for idx, imm_node in enumerate(imm_nodes):
            curr_node_value = datapoint[imm_node - (self.scm.env.state_space + 1)]
            predicted_node_value = predicted_nodes[imm_node]
            diff = predicted_node_value - curr_node_value
            # TODO: can the diff be 0
            direction = 'increase' if diff > 0 else 'decrease'

            if idx == 0:
                explanation += f'To {direction} the value of {self.env.features[imm_node]} (from {curr_node_value:5.1f} to {predicted_node_value[0]:5.1f}) '
            else:
                explanation += f'and {direction} the value of {self.env.features[imm_node]} (from {curr_node_value:5.1f} to {predicted_node_value[0]:5.1f}) '
            
        explanation += 'in the next time step.\n Because:'

        visited_steps = []
        
        for causal_chain in causal_chains:
            print(f'causal chain {causal_chain}')
            for step in causal_chain:
                if step not in visited_steps:
                    visited_steps.append(step)
                    print(f'step {step}')
                    if step[-1] in self.scm.env.reward_nodes:
                        explanation += f'{self.env.features[step[-1]]} influences the reward.'

                    else:
                        explanation += f'{self.env.features[step[-1]]} influences '

        print(explanation)

        # pd.DataFrame.from_dict(
        #     data=explanation, # TODO: or wrap in curly set notation
        #     orient='index').to_csv(
        #     f'why_explanations_{self.scm.env.name}.csv',
        #     mode='a',
        #     header=False)
        

    def _get_sink_nodes(self, causal_graph):
        return list((node for node, out_degree in causal_graph.out_degree() if out_degree == 0))
    
    
    # Generates all the causal chains for a single timestep
    def _get_one_step_causal_chains(self, head_nodes, sink_nodes, causal_graph):
        action_node = self.scm.env.state_space

        all_causal_chains = []

        # Get all causal chains in the graph that do not contain the action node
        for head_node in head_nodes:
            for sink_node in sink_nodes:
                all_chains_between_nodes = nx.all_simple_paths(
                                causal_graph,
                                source=head_node,
                                target=sink_node
                            )

                # We want all the causal chains that don't contain the action node, as these
                # nodes have a causal effect on future nodes?
                # TODO: actually not sure, maybe we want all chains? leave like this for now   
                all_chains_between_nodes = [chain for chain in all_chains_between_nodes if not action_node in chain]
                all_causal_chains.extend(all_chains_between_nodes)

        return all_causal_chains
    

    def _generate_multistep_causal_chains(self, one_step_causal_chains):
        multi_step_causal_chains = []

        for chain in one_step_causal_chains:
            if chain[-1] in self.scm.env.reward_nodes:
                multi_step_causal_chains.append([chain])
            
            else:
                q = deque([[chain]])
                
                while len(q) > 0:
                    curr_chain = q.pop()
                    print(f"curr chain {curr_chain}")

                    # Find all chains that begin with the last node
                    poss_next_chains = [
                        next_chain for next_chain in one_step_causal_chains
                        # TODO: this isn't quite right as theres a mix here between 0-5 and 1-6 etc
                        if next_chain[0] == chain[-1] - (self.scm.env.state_space + 1)
                        and next_chain not in curr_chain # as we don't want to reuse chains in the same chain, since the information
                        # is already given to the user
                    ]

                    # If there are no possible unused chains then this chain will 
                    # just be removed from the deque
                    for poss_next_chain in poss_next_chains:
                        new_chain = curr_chain
                        new_chain.append(poss_next_chain)

                        if poss_next_chain[-1] in self.scm.env.reward_nodes:
                            multi_step_causal_chains.append(new_chain)

                        else:
                            q.append(new_chain)

        return multi_step_causal_chains
    

    def _estimate_most_important_feature(self, state):
        # TODO: implement
        # 0.01 for continuous features, and smallest unit for discrete features, as in paper
        # pertubation = 0.01
        # q_values = self.rl_agent.get_q_values()
        # action_values = q_values(state)

        # importance_vector = np.zeros(len(state))

        # for idx, val in enumerate(state):
        #     importance_vector[idx] = abs(())


        # curr_state = q_values(state)
        

        # Apply small pertubation to each state variable, and recalculate the q_values
        return 1

    
    