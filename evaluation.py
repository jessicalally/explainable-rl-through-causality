# Task prediction method from Madumal et al.
# TODO: need to be careful between discrete and continuous states, since we're
# learning on discrete bins
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
def evaluate_fidelity(test_data, action_influence_model):
    trained_structural_equations = action_influence_model.structural_equations

    # Evaluates how accurately the causal model predicts the chosen action
    # TODO: we need to be able to predict actions directly from the model to
    # do this
    # num_correct_action_predictions = 0

    # Evaluates how accurately the causal model predicts the next state (%)
    avg_predicted_next_state_diff = 0

    for test_case in range(10):
        print(test_case)
        (state, action, next_state) = test_data[test_case]

        predicted_next_states = action_influence_model.predict_from_scm(
            trained_structural_equations,
            state
        )

        # Evaluates fidelity of predicted states
        total_predicted_state_diff = 0.0

        for key in trained_structural_equations:
            if key[1] == action:
                predicted_next_state = predicted_next_states[key]
                actual_next_state = next_state[key[0]]
                diff = (abs(predicted_next_state - actual_next_state)
                        ) / abs(actual_next_state)
                total_predicted_state_diff += diff

        avg_predicted_next_state_diff = (
            (avg_predicted_next_state_diff * test_case) + total_predicted_state_diff) / (test_case + 1)

        # Evaluates fidelity of predicted actions
        # if action == predicted_action:
        #     num_correct_action_predictions += 1

    fidelity = 100 - (avg_predicted_next_state_diff * 100)

    return fidelity

# Performance: Measures the time taken to train the model

# Faithfulness: Measures how faithful explanations are to the RL agent's
# decision-making
