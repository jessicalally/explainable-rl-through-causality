from .rl_agent import RLAgent

# This is just for compatibility with the other classes - the actual algorithm
# can be found in [https://github.com/prashanm/StarCraft-II-causal-explanations]


class A2C(RLAgent):
    def __init__(self, env):
        self.name = "a2c"
        self.env = env
