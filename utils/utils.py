import pandas as pd
import numpy as np
import dill as pickle


def convert_data_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, header=False, index=False)


def convert_csv_to_npy(filename, save_filename):
    data = np.loadtxt(filename, delimiter=',')
    
    with open(save_filename, 'wb') as f:
        np.save(f, data)


def convert_csv_to_npy_pickle(filename, save_filename):
    data = np.loadtxt(filename, delimiter=',')
    
    with open(save_filename, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    convert_csv_to_npy_pickle("starcraft_causal_discovery.csv", "causalstarcraft_a2c.pickle")
    convert_csv_to_npy_pickle("starcraft_reward_causal_discovery.csv", "rewardstarcraft_a2c.pickle")