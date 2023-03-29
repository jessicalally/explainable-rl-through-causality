import numpy as np

def convert_npy_to_csv(npy_filename, csv_filename):
    data = np.load(npy_filename, allow_pickle=True)
    np.savetxt(csv_filename, data, fmt='%s')

def convert_csv_to_npy(csv_filename, npy_filename):
    data = np.genfromtxt(csv_filename, delimiter=',')
    print(data)
    np.save(npy_filename, data, allow_pickle=True)

if __name__ == '__main__':
    convert_csv_to_npy("data.csv", "data.npy")
    convert_csv_to_npy("DAG.csv", "DAG.npy")
    # convert_npy_to_csv("DAG.npy", "DAG.csv")
    # convert_npy_to_csv("data.npy", "data.csv")