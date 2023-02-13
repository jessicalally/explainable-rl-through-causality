import environments.cartpole as c
import evaluation
from rl_algorithms.q_learning import QLearning
import gym

state_space = 4
action_space = 2
env = gym.make('CartPole-v1')

# TODO: remove file eventually, as we will move scm stuff to separate file
# Just to fix circular import currently

q_learning = QLearning(env, state_space, action_space)
q_table, data_set, bins = q_learning.train()
trained_structural_equations = c.train_scm(data_set)

# TODO: this should be some sort of wrapper on a generalised trained rl agent
test_data = q_learning.generate_test_data(env, q_table)
accuracy = evaluation.task_prediction(test_data, trained_structural_equations)
print("Accuracy="+str(accuracy))
fidelity = evaluation.evaluate_fidelity(test_data, trained_structural_equations)
print("Fidelity="+str(fidelity))