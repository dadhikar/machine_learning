"""
Created on Tue May  5 15:37:55 2020 @author: dadhikar
"""
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True


def plot_decision_boundary(X, y, X_test, y_test, classifier):
    """
    """
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                           np.arange(x2_min, x2_max, 0.01))
    x1_grid = xx1.ravel()
    x2_grid = xx2.ravel()
    # x1x2_grid contains each possible combination of x and y value
    x1x2_grid = np.array([x1_grid, x2_grid]).T
    y_predict = classifier.predict(x1x2_grid)
    y_predict = y_predict.reshape(xx1.shape)
    # plotting the decision boundary
    colors = ['magenta', 'skyblue', 'seagreen', 'C3', 'C4', 'C5']
    markers = ["o", "v", "H", "*", "h"]
    cmap = ListedColormap(colors[:len(np.unique(y))], N=None)
    plt.contourf(xx1, xx2, y_predict, cmap=cmap, alpha=0.5)
    # plotting test samples
    for idx, cl in enumerate(np.unique(y_test)):
        plt.scatter(x=X_test[y_test == cl, 0], y=X_test[y_test == cl, 1],
                    c=colors[idx], marker=markers[idx], s=100, alpha=1,
                    edgecolor='b', label=cl)
