import numpy as np

# Task prediction method from Madumal et al.
# TODO: need to be careful between discrete and continuous states, since we're
# learning on discrete bins

# TODO: restructure this for the new causal graph format?


def task_prediction(test_data, action_influence_model):
    trained_structural_equations = action_influence_model.structural_equations
    num_correct_action_predictions = 0
    total_predictions = 0
    print(len(test_data))

    for i in range(10):
        print(i)
        (state, action, next_state) = test_data[i]

        diff_with_actual_value = {}
        total_diffs_per_action = {}

        for key in trained_structural_equations:
            predict_next_states = action_influence_model.predict_from_scm(
                trained_structural_equations, state)
            predicted_value = predict_next_states[key]
            actual_value = next_state[key[0]]
            diff_with_actual_value[key] = abs(predicted_value - actual_value)

            if key[1] in total_diffs_per_action:
                total_diffs_per_action[key[1]] += diff_with_actual_value[key]
            else:
                total_diffs_per_action[key[1]] = diff_with_actual_value[key]

        # print(total_diffs_per_action)
        predicted_action = min(
            total_diffs_per_action,
            key=total_diffs_per_action.get)
        # print(predicted_action)
        # print(action)

        if action == predicted_action:
            num_correct_action_predictions += 1

        total_predictions += 1

    accuracy = 100 * (num_correct_action_predictions / total_predictions)

    return accuracy


# Fidelity: Measures the prediction accuracy of the trained causal model
def evaluate_fidelity(scm, test_data, num_test_points=100):
    print("Evaluating prediction accuracy of the trained SCM...")
    trained_structural_equations = scm.structural_equations

    # Evaluates how accurately the causal model predicts the chosen action
    num_correct_action_predictions = 0

    # Evaluates how accurately the causal model predicts state variables
    avg_mse = 0

    rnd_indices = np.random.choice(len(test_data), num_test_points)
    test_data = test_data[rnd_indices]

    for test_case in range(num_test_points):
        predicted_nodes = scm.predict_from_scm(
            test_data[test_case],
        )

        # Evaluates fidelity of predicted state values
        total_diff = 0.0

        for node in trained_structural_equations:
            predicted_value = predicted_nodes[node][0]
            actual_value = test_data[test_case][node]

            # print(f'node: {node}, predicted: {predicted_value}, actual: {actual_value}')

            if trained_structural_equations[node]['type'] == 'state':
                diff = (abs(actual_value - predicted_value)) ** 2
                total_diff += diff
            elif trained_structural_equations[node]['type'] == 'action':
                if predicted_value == actual_value:
                    num_correct_action_predictions += 1

        mse = total_diff / len(trained_structural_equations)
        avg_mse = ((avg_mse * test_case) + mse) / (test_case + 1)

    avg_correct_action_predictions = num_correct_action_predictions / num_test_points

    return avg_mse, avg_correct_action_predictions

# Performance: Measures the time taken to train the model

# Faithfulness: Measures how faithful explanations are to the RL agent's
# decision-making
