import numpy as np
import math

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
        rmse_per_feature[node] = math.sqrt(
            sum_mse_per_feature[node] / num_test_points)
        avg_prediction_per_feature[node] = avg_prediction_per_feature[node] / \
            num_test_points

    print(f"mean prediction per feature {avg_prediction_per_feature}")
    print(f"RMSE {rmse_per_feature}")

    nrmse_per_feature = {}

    for node in rmse_per_feature:
        nrmse_per_feature[node] = rmse_per_feature[node] / \
            abs(avg_prediction_per_feature[node])

    print(f"NRMSE per feature {nrmse_per_feature}")
    avg_nrmse = sum(nrmse_per_feature.values()) / len(nrmse_per_feature)
    avg_correct_action_predictions = num_correct_action_predictions / num_test_points

    return avg_nrmse, avg_correct_action_predictions
