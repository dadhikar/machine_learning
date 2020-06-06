import os
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
import numpy as np 
from load_data import load_mat_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from plot_decision_boundary import plot_decision_boundary


# current working directory
cwd = os.getcwd()
# directory for the data file
file_dir = cwd + os.sep + 'data_sets'

data = load_mat_data(file_dir, 'data2.mat')

X = data['X']
y = np.ravel(data['y'])
y = y.reshape(len(y), 1).ravel()



# splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200,
                                                   random_state=0)

C_list = [0.01, 0.1, 1, 10, 50, 100]
gamma_list = [0.01, 0.1, 1, 10, 50, 100]
nx = 2
ny = 3
fig, ax = plt.subplots(nx, ny)
ax = ax.flatten()


for i in range(len(gamma_list)):
    clf = SVC(C=10, kernel='rbf', degree=3, gamma=gamma_list[i], coef0=0.0,
              shrinking=True, probability=False, tol=0.001, cache_size=200,
              class_weight=None, verbose=False, max_iter=-1,
              decision_function_shape='ovr', break_ties=False,
              random_state=None)

    clf.fit(X_train, y_train)
    
    # plotting decision boundary
    x1_min, x1_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    x2_min, x2_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                   np.arange(x2_min, x2_max, 0.01))
    x1_grid = xx1.ravel()
    x2_grid = xx2.ravel()
    # x1x2_grid contains each possible combination of x and y value
    x1x2_grid = np.array([x1_grid, x2_grid]).T
    y_predict = clf.predict(x1x2_grid)
    y_predict = y_predict.reshape(xx1.shape)
    # plotting the decision boundary
    colors = ['magenta', 'skyblue', 'seagreen', 'C3', 'C4', 'C5']
    markers = ["o", "v", "H", "*", "h"]
    cmap = ListedColormap(colors[:len(np.unique(y))], N=None)
    ax[i].contourf(xx1, xx2, y_predict, cmap=cmap, alpha=0.5)
    ax[i].set_title(r"C:10, gamma:{}". format(gamma_list[i]))
    # plotting test samples
    for idx, cl in enumerate(np.unique(y_test)):
        ax[i].scatter(x=X_test[y_test == cl, 0],
                    y=X_test[y_test == cl, 1], c=colors[idx],
                    marker=markers[idx], s=100, alpha=1,
                    edgecolor='b', label=cl)
    ax[i].legend(loc='best')
    plt.show()    
