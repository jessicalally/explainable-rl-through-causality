import networkx as nx
import numpy as np
import gym

# TODO: separate environments into separate files


class Environment(object):
    name = None
    state_space = None
    action_space = None
    env = None
    true_dag = None
    action_node = 4
    reward_node = 9

    # TODO: currently needed for VarLiNGAM DAG plots, find a better way to
    # generalise for all methods
    labels = None
    features = None
    actions = None
    forbidden_edges = []
    required_edges = []

    def __init__(self):
        self.causal_graph = nx.from_numpy_matrix(
            self.true_dag, create_using=nx.MultiDiGraph())

    @staticmethod
    def get_env(env):
        if env == 'cartpole':
            return Cartpole()
        elif env == 'lunarlander':
            return LunarLander()
        elif env == 'mountaincar':
            return MountainCar()
        elif env == 'taxi':
            return Taxi()
        elif env == 'starcraft':
            return Starcraft()
        else:
            raise ValueError(f'{env} is an unsupported environment')

##### Cartpole #####


class Cartpole(Environment):
    name = "cartpole"
    state_space = 4
    action_space = 2
    env = gym.make('CartPole-v1')

    true_dag = np.array([
        [0, 0, 0, 0, 1, 1, 0, 0, 0],  # 0 = pos t
        [0, 0, 0, 0, 1, 1, 1, 0, 0],  # 1 = velocity t
        [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 2 = pole angle t
        [0, 0, 0, 0, 1, 0, 0, 1, 1],  # 3 = angular velocity t
        [0, 0, 0, 0, 0, 1, 1, 1, 1],  # 4 = action t
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 = pos t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6 = velocity t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7 = pole angle t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8 = angular t+1
    ])

    reward_true_dag = np.array([
        [0, 0, 0, 0, 1], # 0 = pos
        [0, 0, 0, 0, 0], # 1 = velocity
        [0, 0, 0, 0, 1], # 2 = pole angle
        [0, 0, 0, 0, 0], # 3 = angular velocity
        [0, 0, 0, 0, 0], # 4 = reward
    ])

    action_node = 4

    labels = [
        'pos(t)',
        'velo(t)',
        'angle(t)',
        'ang-velo(t)',
        'action(t)',
        'reward(t)',
        'pos(t-1)',
        'velo(t-1)',
        'angle(t-1)',
        'ang-velo(t-1)',
        'action(t-1)',
        'reward(t-1)']

    features = {
        0: 'cart position',
        1: 'cart velocity',
        2: 'pole angle',
        3: 'pole angular velocity',
        4: 'action',
        5: 'cart position',
        6: 'cart velocity',
        7: 'pole angle',
        8: 'pole angular velocity',
        9: 'reward',
    }

    actions = {
        0: 'Push cart to left',
        1: 'Push cart to right'
    }

    # Assumption: we cannot have any causal relationships that go backwards in
    # time
    forbidden_edges = [
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (5, 0),
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 4),
        (6, 0),
        (6, 1),
        (6, 2),
        (6, 3),
        (6, 4),
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
        (7, 4),
        (8, 0),
        (8, 1),
        (8, 2),
        (8, 3),
        (8, 4),
    ]

    # We only want the edges from state variables to the reward, so forbid all other edges
    forbidden_edges_reward = [
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
    ]

    # Assumption: all past state variables affect action choice and action
    # affects all future state variables
    required_edges = [
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (4, 8),
    ]


##### Lunar Lander #####

class LunarLander(Environment):
    name = "lunarlander"
    state_space = 8
    action_space = 4

    env = gym.make(
        "LunarLander-v2",
    )

    actions = {
        0: "Do nothing",
        1: "Fire left orientation engine",
        2: "Fire main engine",
        3: "Fire right orientation engine"
    }

    features = {
        0: 'x-coord',
        1: 'y-coord',
        2: 'x-velocity',
        3: 'y-velocity',
        4: 'angle',
        5: 'angular velocity',
        6: 'left leg in contact with ground',
        7: 'right leg in contact with ground',
        8: 'action',
        9: 'x-coord',
        10: 'y-coord',
        11: 'x-velocity',
        12: 'y-velocity',
        13: 'angle',
        14: 'angular velocity',
        15: 'left leg in contact with ground',
        16: 'right leg in contact with ground',
    }

    # Rewar equation taken from code
    # reward = -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
    # - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
    # - 100 * abs(state[4])
    # + 10 * state[6]
    # + 10 * state[7]
    action_node = 8

    true_dag = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 0 = x coord t
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 1 = y coord t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # 2 = x velocity t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],  # 3 = y velocity t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # 4 = angle t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],  # 5 = angular velocity t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 6 = left leg in contact with ground t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 7 = right leg in contact with ground t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 8 = action t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # 9 = x coord t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # 10 = y coord t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11 = x velocity t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12 = y velocity t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13 = angle t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14 = angular velocity t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15 = left leg in contact with ground t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 16 = right leg in contact with ground t+1
    ])

    reward_true_dag = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # 0 = x coord t
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # 1 = y coord t
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # 2 = x velocity t
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # 3 = y velocity t
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # 4 = angle t
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # 5 = angular velocity t
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # 6 = left leg t
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # 7 = right leg t
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # 8 = reward
    ])

    labels = [
        'x-coord(t)',
        'y-coord(t)',
        'x-velo(t)',
        'y-velo(t)',
        'ang(t)',
        'ang-velo(t)',
        'left-leg(t)',
        'right-leg(t)',
        'action(t)',
        'x-coord(t-1)',
        'y-coord(t-1)',
        'x-velo(t-1)',
        'y-velo(t-1)',
        'ang(t-1)',
        'ang-velo(t-1)',
        'left-leg(t-1)',
        'right-leg(t-1)',
        'action(t-1)']
    
    forbidden_edges_reward = [
       (8, 0),
       (8, 1),
       (8, 2),
       (8, 3),
       (8, 4),
       (8, 5),
       (8, 6),
       (8, 7),
       (8, 8),
    ]

    # Assumption: we cannot have any causal relationships that go backwards in
    # time
    forbidden_edges = [
        (9, 0),
        (9, 1),
        (9, 2),
        (9, 3),
        (9, 4),
        (9, 5),
        (9, 6),
        (9, 7),
        (9, 8),
        (9, 9),
        (10, 0),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
        (10, 5),
        (10, 6),
        (10, 7),
        (10, 8),
        (10, 9),
        (11, 0),
        (11, 1),
        (11, 2),
        (11, 3),
        (11, 4),
        (11, 5),
        (11, 6),
        (11, 7),
        (11, 8),
        (11, 9),
        (12, 0),
        (12, 1),
        (12, 2),
        (12, 3),
        (12, 4),
        (12, 5),
        (12, 6),
        (12, 7),
        (12, 8),
        (12, 9),
        (13, 0),
        (13, 1),
        (13, 2),
        (13, 3),
        (13, 4),
        (13, 5),
        (13, 6),
        (13, 7),
        (13, 8),
        (13, 9),
        (13, 0),
        (14, 0),
        (14, 1),
        (14, 2),
        (14, 3),
        (14, 4),
        (14, 5),
        (14, 6),
        (14, 7),
        (14, 8),
        (14, 9),
        (15, 0),
        (15, 1),
        (15, 2),
        (15, 3),
        (15, 4),
        (15, 5),
        (15, 6),
        (15, 7),
        (15, 8),
        (15, 9),
        (16, 0),
        (16, 1),
        (16, 2),
        (16, 3),
        (16, 4),
        (16, 5),
        (16, 6),
        (16, 7),
        (16, 8),
        (16, 9),
    ]

    # Assumption: all past state variables affect action choice and action
    # affects all future state variables
    required_edges = [
        (0, 8),
        (1, 8),
        (2, 8),
        (3, 8),
        (4, 8),
        (5, 8),
        (6, 8),
        (7, 8),

        (8, 9),
        (8, 10),
        (8, 11),
        (8, 12),
        (8, 13),
        (8, 14),
        (8, 15),
        (8, 16),
    ]

    def __init__(self):
        self.causal_graph = nx.from_numpy_matrix(
            self.true_dag, create_using=nx.MultiDiGraph())


#### Mountain Car ####

# Transition dynamics for Mountain Car
# (https://gymnasium.farama.org/environments/classic_control/mountain_car/)

# velocityt+1 = velocityt + (action - 1) * force - cos(3 * positiont) * gravity
# positiont+1 = positiont + velocityt+1

class MountainCar(Environment):
    name = "mountaincar"
    state_space = 2
    action_space = 3
    env = gym.make('MountainCar-v0')

    true_dag = np.array([
        [0, 0, 1, 1, 1],  # 0 = pos t
        [0, 0, 1, 0, 1],  # 1 = velocity t
        [0, 0, 0, 0, 1],  # 2 = action t
        [0, 0, 0, 0, 0],  # 3 = pos t + 1
        [0, 0, 0, 1, 0],  # 4 = velocity t + 1
    ])

    reward_true_dag = np.array([
        [0, 0, 1], # 0 = pos t
        [0, 0, 0], # 1 = velocity t
        [0, 0, 0], # 2 = reward
    ])

    labels = [
        'pos(t)',
        'velocity(t)',
        'action(t)',
        'pos(t-1)',
        'velocity(t-1)',
        'action(t-1)'
    ]

    actions = {
        0: "Accelerate to the left",
        1: "Don’t accelerate",
        2: "Accelerate to the right"
    }

    features = {
        0: 'car position',
        1: 'car velocity',
        2: 'action',
        3: 'car position',
        4: 'car velocity',
        5: 'reward'
    }

    action_node = 2

    forbidden_edges_reward = [
        (2, 0),
        (2, 1),
        (2, 2),
    ]

    # Assumption: we cannot have any causal relationships that go backwards in
    # time
    forbidden_edges = [
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (3, 2),
        (4, 0),
        (4, 1),
        (4, 2),
    ]

    # Assumption: all past state variables affect action choice and action
    # affects all future state variables
    required_edges = [
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 4)
    ]

    def __init__(self):
        self.causal_graph = nx.from_numpy_matrix(
            self.true_dag, create_using=nx.MultiDiGraph())


#### Taxi ####

class Taxi(Environment):
    name = "taxi"
    state_space = 4
    action_space = 6
    env = gym.make('Taxi-v3')

    true_dag = np.array([
        [0, 0, 0, 0, 1, 1, 0, 0, 0],  # 0 = taxi row t
        [0, 0, 0, 0, 1, 0, 1, 0, 0],  # 1 = taxi column t
        [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 2 = passenger location t
        [0, 0, 0, 0, 1, 0, 0, 0, 1],  # 3 = destination t
        [0, 0, 0, 0, 0, 1, 1, 1, 1],  # 4 = action t
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 = taxi row t + 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6 = taxi column t + 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7 = passenger location t + 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8 = destination t + 1
    ])

    reward_true_dag = np.array([
        [0, 0, 0, 0, 0], # 0 = taxi row t
        [0, 0, 0, 0, 0], # 1 = taxi column t
        [0, 0, 0, 0, 1], # 2 = passenger location t
        [0, 0, 0, 0, 1], # 3 = destination t
        [0, 0, 0, 0, 0], # 4 = reward t
    ])


    labels = [
        'row(t)',
        'column(t)',
        'pass(t)',
        'dest(t)',
        'action(t)',
        'row(t-1)',
        'column(t-1)',
        'pass(t-1)',
        'dest(t-1)',
        'action(t-1)'
    ]

    actions = {
        0: "Move south (down)",
        1: "Move north (up)",
        2: "Move east (right)",
        3: "Move west (left)",
        4: "Pickup passenger",
        5: "Drop off passenger"
    }

    features = {
        0: 'taxi row',
        1: 'taxi column',
        2: 'passenger location',
        3: 'destination',
        4: 'action',
        5: 'taxi row',
        6: 'taxi column',
        7: 'passenger location',
        8: 'destination',
        9: 'action',
        10: 'reward'
    }

    action_node = 4

    forbidden_edges_reward = [
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ]

    # Assumption: we cannot have any causal relationships that go backwards in
    # time
    forbidden_edges = [
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (5, 0),
        (5, 1),
        (5, 2),
        (5, 3),
        (6, 0),
        (6, 1),
        (6, 2),
        (6, 3),
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
        (8, 0),
        (8, 1),
        (8, 2),
        (8, 3),
    ]

    # Assumption: all past state variables affect action choice and action
    # affects all future state variables
    required_edges = [
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (4, 8),
    ]

    # 0 = move south, 1 = move north, 2 = move east, 3 = move west,
    # 4 = pickup passenger, 5 = drop off passenger
    action_set = (0, 1, 2, 3, 4, 5)

    learnt_causal_graph = None

    def __init__(self):
        self.causal_graph = nx.from_numpy_matrix(
            self.true_dag, create_using=nx.MultiDiGraph())


#### Starcraft ####

class Starcraft(Environment):
    name = "starcraft"
    state_space = 9
    action_space = 5 # includes a do-nothing action
    env = None  # TODO

    true_dag = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 0 = worker supply number t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # 1 = supply depot number t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # 2 = barracks number t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1],  # 3 = enemy location t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1],  # 4 = ally unit number t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],  # 5 = ally unit health t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # 6 = ally unit location t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 7 = destroyed units t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 8 = destroyed buildings t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 9 = action t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10 = worker supply number t + 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11 = supply depot number t + 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12 = barracks number t + 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13 = enemy location t + 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14 = ally unit number t + 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15 = ally unit health t + 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 16 = ally unit location t + 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 17 = destroyed units t + 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 18 = destroyed buildings t + 1
    ])

    reward_true_dag = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 = worker supply number t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 = supply depot number t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 = barracks number t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3 = enemy location t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 = ally unit number t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 = ally unit health t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6 = ally unit location t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 7 = destroyed units t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 8 = destroyed buildings t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9 = reward t
    ])

    labels = [
        'W(t)',
        'S(t)',
        'B(t)',
        'El(t)',
        'An(t)',
        'Ah(t)',
        'Al(t)',
        'Du(t)',
        'Db(t)'
        'A(t)',
        'W(t-1)',
        'S(t-1)',
        'B(t-1)',
        'El(t-1)',
        'An(t-1)',
        'Ah(t-1)',
        'Al(t-1)',
        'Du(t-1)',
        'Db(t-1)'
        'A(t-1)',
    ]

    forbidden_edges_reward = [
        (9, 0),
        (9, 1),
        (9, 2),
        (9, 3),
        (9, 4),
        (9, 5),
        (9, 6),
        (9, 7),
        (9, 8),
        (9, 9),
    ]

    # Assumption: we cannot have any causal relationships that go backwards in
    # time
    forbidden_edges = [
        (9, 0),
        (9, 1),
        (9, 2),
        (9, 3),
        (9, 4),
        (9, 5),
        (9, 6),
        (9, 7),
        (9, 8),
        (10, 0),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 4),
        (10, 5),
        (10, 6),
        (10, 7),
        (10, 8),
        (11, 0),
        (11, 1),
        (11, 2),
        (11, 3),
        (11, 4),
        (11, 5),
        (11, 6),
        (11, 7),
        (11, 8),
        (12, 0),
        (12, 1),
        (12, 2),
        (12, 3),
        (12, 4),
        (12, 5),
        (12, 6),
        (12, 7),
        (12, 8),
        (13, 0),
        (13, 1),
        (13, 2),
        (13, 3),
        (13, 4),
        (13, 5),
        (13, 6),
        (13, 7),
        (13, 8),
        (14, 0),
        (14, 1),
        (14, 2),
        (14, 3),
        (14, 4),
        (14, 5),
        (14, 6),
        (14, 7),
        (14, 8),
        (15, 0),
        (15, 1),
        (15, 2),
        (15, 3),
        (15, 4),
        (15, 5),
        (15, 6),
        (15, 7),
        (15, 8),
        (16, 0),
        (16, 1),
        (16, 2),
        (16, 3),
        (16, 4),
        (16, 5),
        (16, 6),
        (16, 7),
        (16, 8),
        (17, 0),
        (17, 1),
        (17, 2),
        (17, 3),
        (17, 4),
        (17, 5),
        (17, 6),
        (17, 7),
        (17, 8),
        (18, 0),
        (18, 1),
        (18, 2),
        (18, 3),
        (18, 4),
        (18, 5),
        (18, 6),
        (18, 7),
        (18, 8),
    ]

    # Assumption: all past state variables affect action choice and action
    # affects all future state variables
    required_edges = [
        (0, 9),
        (1, 9),
        (2, 9),
        (3, 9),
        (4, 9),
        (5, 9),
        (6, 9),
        (7, 9),
        (8, 9),
        (9, 10),
        (9, 11),
        (9, 12),
        (9, 13),
        (9, 14),
        (9, 15),
        (9, 16),
        (9, 17),
        (9, 18),
    ]

    def __init__(self):
        self.causal_graph = nx.from_numpy_matrix(
            self.true_dag, create_using=nx.MultiDiGraph())
