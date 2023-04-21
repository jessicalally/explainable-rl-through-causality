import pandas as pd
import numpy as np


def convert_data_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, header=False, index=False)


def convert_csv_to_npy(filename, save_filename):
    data = np.loadtxt(filename, delimiter=',')
    
    with open(save_filename, 'wb') as f:
        np.save(f, data)
