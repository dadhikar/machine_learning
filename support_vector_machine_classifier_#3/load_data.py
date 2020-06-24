import os
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np 
cwd = os.getcwd()
file_dir = cwd + os.sep + 'data_sets'
print(cwd)
# file_list = os.listdir(cwd + os.sep + 'data_sets')
# print(file_list)

def load_mat_data(file_dir, file_name):
    """
    file_dir = path for data stored in matlab file format
    filename = name of the data file
    Returns
    -------
    directory with variables names as key

    """
    data = loadmat(file_dir + os.sep + file_name)
    return data

def plot_data():
    """
    plot data 

    Returns
    -------
    None.

    """
    data = load_mat_data(file_dir, 'data3.mat')
    X = data["X"]
    y = data["y"].ravel()
    key_list = []
    for keys in data.keys():
        key_list.append(keys)
    plt.scatter(X[np.where(y==1), 0], X[np.where(y==1),1], c='r')
    plt.scatter(X[np.where(y==0), 0], X[np.where(y==0),1], c='g')
    plt.title('Linearly inseparable data')
    plt.xlabel('X1 [ab. unit]')
    plt.ylabel('X2 [ab. unit]')
    # plt.savefig('data3_info', dpi=150, format='png')
    plt.show()

if __name__ == "__main__":
    
    # execute only if run as a script
    plot_data()