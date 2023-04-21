import copy
import tensorflow.compat.v1 as tf
import networkx as nx
import numpy as np
import pandas as pd
import explanation_templates as explanations
import rl_algorithms.q_learning
import gym


class Cartpole:
    def __init__(self):
        self.state_space = 4
        self.action_space = 2
        self.env = gym.make('CartPole-v1')

        causal_graph = nx.from_numpy_matrix(np.array([
            [1, 0, 0, 0],  # 0 = cart position
            [1, 1, 0, 0],  # 1 = cart velocity
            [0, 0, 1, 0],  # 2 = pole angle
            [0, 0, 1, 1],  # 3 = pole angular velocity
        ]), create_using=nx.MultiDiGraph())

        self.causal_graph = nx.MultiDiGraph()

        self.action_matrix = np.array([
            [[0, 1], [0, 1], [0, 1], [0, 1]],
            [[0, 1], [0, 1], [0, 1], [0, 1]],
            [[0, 1], [0, 1], [0, 1], [0, 1]],
            [[0, 1], [0, 1], [0, 1], [0, 1]],
        ])

        for edge in causal_graph.edges():
            self.causal_graph.add_edge(
                edge[0], edge[1], action=0)  # Push cart to left
            self.causal_graph.add_edge(
                edge[0], edge[1], action=1)  # Push cart to right

        # 0 = push cart to left, 1 = push cart to right
        self.action_set = (0, 1)
