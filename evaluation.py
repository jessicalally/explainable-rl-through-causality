import numpy as np
import math

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
def evaluate_fidelity(scm, test_data, num_test_points=100, REWARD_DAG=False):
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

            if REWARD_DAG:
                diff = (abs(actual_value - predicted_value)) ** 2
                total_diff += diff
            else:
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

def evaluate_scm(scm, test_data, num_test_points=1000):
    print("Evaluating NMRSE and action prediction accuracy...")
    trained_structural_equations = scm.structural_equations

    # Evaluates how accurately the causal model predicts the chosen action
    num_correct_action_predictions = 0

    sum_mse_per_feature = {}
    avg_prediction_per_feature = {}

    # Take 100 points randomly from the test dataset
    rnd_indices = np.random.choice(len(test_data), num_test_points)
    test_data = test_data[rnd_indices]


    for test_case in range(num_test_points):
        predicted_nodes = scm.predict_from_scm(
            test_data[test_case],
        )

        # For each structural equation
        # Average the RMSE over 1000 datapoints
        # Average the predictions to find the NRMSE
        # Then average the NRMSEs to find an overall metric for the model

        for node in trained_structural_equations:
            predicted_value = predicted_nodes[node][0]
            actual_value = test_data[test_case][node]

            if trained_structural_equations[node]['type'] == 'state':
                squared_diff = (abs(actual_value - predicted_value)) ** 2

                if node in sum_mse_per_feature:
                    sum_mse_per_feature[node] += squared_diff
                    avg_prediction_per_feature[node] += abs(predicted_value)
                else:
                    sum_mse_per_feature[node] = squared_diff
                    avg_prediction_per_feature[node] = abs(predicted_value)

            elif trained_structural_equations[node]['type'] == 'action':
                if predicted_value == actual_value:
                    num_correct_action_predictions += 1

    print(f"sum mse per feature {sum_mse_per_feature}")
    rmse_per_feature = {}

    for node in sum_mse_per_feature:
        rmse_per_feature[node] = math.sqrt(sum_mse_per_feature[node] / num_test_points)
        avg_prediction_per_feature[node] = avg_prediction_per_feature[node] / num_test_points

    print(f"mean prediction per feature {avg_prediction_per_feature}")
    print(f"RMSE {rmse_per_feature}")

    nrmse_per_feature = {}

    for node in rmse_per_feature:
        nrmse_per_feature[node] = rmse_per_feature[node] / abs(avg_prediction_per_feature[node])

    print(f"NRMSE per feature {nrmse_per_feature}")
    avg_nrmse = sum(nrmse_per_feature.values()) / len(nrmse_per_feature)
    avg_correct_action_predictions = num_correct_action_predictions / num_test_points

    return avg_nrmse, avg_correct_action_predictions

