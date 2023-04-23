import numpy as np
import gym


class Environment(object):
    state_space = None
    action_space = None
    env = None
    true_dag = None
    # TODO: currently needed for VarLiNGAM DAG plots, find a better way to
    # generalise for all methods
    labels = None
    forbidden_edges = []
    required_edges = []

    @staticmethod
    def get_env(env):
        if env == 'cartpole':
            return Cartpole()
        elif env == 'lunarlander':
            return LunarLander()
        elif env == 'mountaincar':
            return MountainCar()
        else:
            raise ValueError(f'{env} is an unsupported environment')

##### Cartpole #####


class Cartpole(Environment):
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
        [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 8 = angular t+1
    ])

    labels = [
        'pos(t)',
        'velo(t)',
        'angle(t)',
        'ang-velo(t)',
        'action(t)',
        'pos(t-1)',
        'velo(t-1)',
        'angle(t-1)',
        'ang-velo(t-1)',
        'action(t-1)']

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
    state_space = 8
    action_space = 4

    env = gym.make(
        "LunarLander-v2",
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
    )

    true_dag = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 0 = x coord t
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 1 = y coord t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # 2 = x velocity t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],  # 3 = y velocity t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # 4 = angle t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            1, 1, 0, 0],  # 5 = angular velocity t
        # 6 = left leg in contact with ground t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        # 7 = right leg in contact with ground t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 8 = action t
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # 9 = x coord t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # 10 = y coord t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11 = x velocity t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12 = y velocity t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13 = angle t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0],  # 14 = angular velocity t+1
        # 15 = left leg in contact with ground t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # 16 = right leg in contact with ground t+1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
        (16, 9)
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


#### Mountain Car ####

# Transition dynamics for Mountain Car
# (https://gymnasium.farama.org/environments/classic_control/mountain_car/)

# velocityt+1 = velocityt + (action - 1) * force - cos(3 * positiont) * gravity
# positiont+1 = positiont + velocityt+1

class MountainCar(Environment):
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

    labels = [
        'pos(t)',
        'velocity(t)',
        'action(t)',
        'pos(t-1)',
        'velocity(t-1)',
        'action(t-1)'
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
