import environments.cartpole as c # TODO: split structural equation work into new file

# Runs the trained RL agent to generate test data over x timesteps, then 
# takes y datapoints randomly to form the test data
# def generate_test_data(trained_rl_agent):
    # num_test_datapoints = 10
    # test_data = [] # Tuples of (s, a, s')

    # for i in range(num_test_datapoints):
    #     # Get current state
    #     state = _

    #     # Get action chosen by RL agent
    #     action = _

    #     # Get next state
    #     next_state = _

    #     # Add to test_data set
    #     test_data[i] = (state, action, next_state)
    
    # return test_data

# Task prediction method from Madumal et al.
# TODO: need to be careful between discrete and continuous states, since we're
# learning on discrete bins
def task_prediction(test_data, trained_structural_equations):
    num_correct_action_predictions = 0
    print(len(test_data))

    for i in range(1):
        print("i")
        (state, action, next_state) = test_data[i]

        diff_with_actual_value = {}
        total_diffs_per_action = {}

        # Task Prediction method from Madumal et al.
        for key in trained_structural_equations:
            predict_next_states = c.predict_from_scm(trained_structural_equations, state)
            predicted_value = predict_next_states[key]
            print(predicted_value)
            actual_value = next_state[key[0]]
            print(actual_value)
            diff_with_actual_value[key] = abs(predicted_value - actual_value)

            if key[1] in total_diffs_per_action:
                total_diffs_per_action[key[1]] += diff_with_actual_value[key]
            else:
                total_diffs_per_action[key[1]] = diff_with_actual_value[key]
            
        print(total_diffs_per_action)
        predicted_action = min(total_diffs_per_action, key=total_diffs_per_action.get)

        if action == predicted_action:
            num_correct_action_predictions += 1


    accuracy = num_correct_action_predictions / len(test_data)

    return accuracy


# Fidelity: Measures the prediction accuracy of the trained causal model
def evaluate_fidelity(test_data, trained_structural_equations):
    # Evaluates how accurately the causal model predicts the chosen action
    # TODO: we need to be able to predict actions directly from the model to 
    # do this 
    # num_correct_action_predictions = 0

    # Evaluates how accurately the causal model predicts the next state (%)
    avg_predicted_next_state_diff = 0

    for i in range(1):
        print("i")
        (state, action, next_state) = test_data[i]

        predicted_next_states = c.predict_from_scm(
            trained_structural_equations, 
            state
        )

        # Evaluates fidelity of predicted states
        total_predicted_state_diff = 0

        for key in trained_structural_equations:
            if key[1] == action:
                predicted_next_state = predicted_next_states[key]
                actual_next_state = next_state[key[0]]
                diff = (abs(predicted_next_state - actual_next_state)) / actual_next_state

                for d in diff:
                    total_predicted_state_diff += d

                # TODO: is this sometimes negative and why
                print(total_predicted_state_diff)
        
        avg_predicted_next_state_diff = ((avg_predicted_next_state_diff * (i - 1)) + total_predicted_state_diff) / max(1, i)
        
        # Evaluates fidelity of predicted actions
        # if action == predicted_action:
        #     num_correct_action_predictions += 1


    fidelity = 100 - (avg_predicted_next_state_diff * 100) # Concert to percentage

    return fidelity

# Performance: Measures the time taken to train the model

# Faithfulness: Measures how faithful explanations are to the RL agent's decision-making