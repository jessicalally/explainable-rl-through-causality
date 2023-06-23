# Explainable Reinforcement Learning Through Causality
A structural causal model (SCM) approach to explainable reinforcement learning, developed as part of my MEng Final Year Project at Imperial College London.

This project extends previous work by Madumal et al. [[1]](#1) to develop and implement a fully domain-agnostic explanation generation approach. By detecting the underlying causal relationships, and with our novel explanation generation algorithm, we ensure that this tool is generalisable to a range of RL environments.

## RL Algorithms
Some of the RL algorithms presented in this codebase were adapted from the following sources:
| RL Algorithm | Source | License |
| --- | --- | --- |
| `SARSA` | [dennybritz/reinforcement-learning](https://github.com/dennybritz/reinforcement-learning) | MIT License |
| `Policy Gradient` | [bentrevett/pytorch-rl/tree/master](https://github.com/bentrevett/pytorch-rl/tree/master) | MIT License |

## References
<a id="1">[1]</a> 
Madumal P, Miller T, Sonenberg L, Vetere F. (2020). 
Explainable reinforcement learning through a causal lens.
Proceedings of the AAAI Conference on Artificial Intelligence, 34(3), 2493-2500.
