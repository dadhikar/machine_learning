# author: Dasharath
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

#----------Perceptron_Learning------------------------------------------------------
def Perceptron_train(X, y, epoch_n, eta_lr):
    """
    Training data, X is a 2D array with shape = [samples_size, feature_size].
    Target data, y is a 1D array. 
    we will be running weight update epoch_n number of times.
    eta_lr is the learning rate (0.0, 1.0).
    """
    # initialize random weights of size = feature_size
    rng = np.random.default_rng()
    w = rng.standard_normal(size = 1 + X.shape[1])
    train_misclassifications = []
    for _ in range(epoch_n):
        # calculating z = W^T*X, net input
        z =  w[0] + np.dot(X, w[1:])    # w[0] is the bias unit
        # predicting the class level if z >=0 class 1 otherwise -1
        class_predict = np.where(z>=0.0, 1, -1)
        train_misclassified = 0 
        for train_x, target_y, predict in zip(X, y, class_predict):
            error = (target_y - predict)
            train_misclassified += int(error != 0.0)   
            w[0] += eta_lr*error
            w[1:] += eta_lr*error*train_x 
        train_misclassifications.append(train_misclassified)          
    return train_misclassifications, w, class_predict

def Perceptron_decision_boundary(w_learned, X):
    """
    w_learned - weights learned during the training process
    X_test  - test data
    y_test  - target output to test
    """
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
    x1_grid = xx1.ravel()
    x2_grid = xx2.ravel()
    x1x2_grid = np.array([x1_grid, x2_grid]).T  # x1x2_grid contains each possible combination of x and y value
    z =  w_learned[0] + np.dot(x1x2_grid, w_learned[1:])
    class_prediction = []
    for z_ in z:
        if z_ >= 0.0:
            class_prediction.append(1)
        else:
            class_prediction.append(-1)
    class_prediction = np.asarray(class_prediction) 
    class_prediction = class_prediction.reshape(xx1.shape)                      
    return xx1, xx2, class_prediction 
    
def Perceptron_test(w_learned, X_test):
    """
    w_learned - weights learned during the training process
    X_test  - test data
    y_test  - target output to test
    """
    test_predict = []
    for i in range(X_test.shape[0]):
        z = w_learned[0] + np.dot(X_test[i], w_learned[1:])
        if z >= 0.0:
            test_predict.append(1)
        else:
            test_predict.append(-1)             
    return test_predict
#--------------------------------------------------------------------------------

#-------------------Adaline_Learning---------------------------------------------



