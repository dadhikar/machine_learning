# author: Dasharath
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

#feature scaling 
def feature_scale(X):
    X_n = np.copy(X)
    X_n[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_n[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    return X_n

#----------Perceptron_Learning------------------------------------------------------
def perceptron_train(X_train, y_train, epoch_train, eta_train):
    """
    Training data, X is a 2D array with shape = [samples_size, feature_size].
    Target data, y is a 1D array. 
    we will be running weight update epoch_n number of times.
    eta_lr is the learning rate (0.0, 1.0).
    """
    # initialize random weights of size = feature_size
    rng = np.random.default_rng()
    w = rng.standard_normal(size = 1 + X_train.shape[1])
    train_misclassifications = []
    for _ in range(epoch_train):
        # calculating z = W^T*X, net input
        z =  w[0] + np.dot(X_train, w[1:])    # w[0] is the bias unit
        # predicting the class level if z >=0 class 1 otherwise -1
        class_predict = np.where(z>=0.0, 1, -1)
        train_misclassified = 0 
        for train_x, target_y, predict in zip(X_train, y_train, class_predict):
            error = (target_y - predict)
            train_misclassified += int(error != 0.0)   
            w[0] += eta_train*error
            w[1:] += eta_train*error*train_x 
        train_misclassifications.append(train_misclassified)          
    return train_misclassifications, w, class_predict

    
def perceptron_test(w_learned, X_test, y_test):
    """
    w_learned - weights learned during the training process
    X_test  - test data
    y_test  - target output to test
    """
    z =  w_learned[0] + np.dot(X_test, w_learned[1:]) 
    predict = np.where(z>=0.0, 1, -1)
    error= y_test - predict          
    return error
def perceptron_decision_boundary(w_learned, X, y):
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
    z = w_learned[0] + np.dot(x1x2_grid, w_learned[1:])
    predict = np.where(z>=0, 1, -1)
    predict = predict.reshape(xx1.shape) 
    # plotting the decision boundary 
    colors = ['C0','C1','C2','C3']
    markers = ['D', 's', 'v']
    cmap = ListedColormap(colors, N=None)
    plt.contourf(xx1, xx2, predict, cmap = cmap)
    #plotting test samples 
    for idx, cl in enumerate(np.unique(y)):  
        plt.scatter(x= X[y== cl, 0], y= X[y== cl, 1], c= colors[idx], 
                       marker= markers[idx], edgecolor='k', label= cl)
    #plt.xlim(X[:, 0].min(), X[:, 0].max())
    #plt.ylim(X[:, 1].min(), X[:, 1].max())               
    print('*'*30)
    x = input('Please enter first feature positional index >> ')
    y = input('Please enter second feature positional index >> ')
    print('*'*30)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.show()  
#-----------------------------------------------------------------------------------
#----------ADAptive LInear NEuron(Adaline)------------------------------------------

def adaline_train_batch_descent(X_train, y_train, epoch_train, eta_train):
    """
    Batch Gradient Descent
    """
    # initialize random weights of size = feature_size
    rng = np.random.default_rng()
    w_ = rng.standard_normal(size = 1 + X_train.shape[1])
    cost = []
    for _ in range(epoch_train):
        # calculating z = W^T*X, net input
        z_ = w_[0] + np.dot(X_train, w_[1:])        
        error = (y_train - z_)
        cost_fn = 0.5*(error**2).sum()    #cost function 
        cost.append(cost_fn)  
        w_[0] += eta_train*error.sum()
        w_[1:] += eta_train*np.dot(X_train.T, error)
    class_predict = np.where((w_[0] + np.dot(X_train, w_[1:])) >=0.0, 1, -1)
    return cost, w_, class_predict
    # Stochastic Gradient Descent
def adaline_train_stochastic_descent(X_train, y_train, epoch_train, eta_train, shuffle=True):
    """
    performance of the algorithms can be enhanced by presenting the
    training data in a random order. To achieve, we shuffle the 
    training data using permutation function in numpy.random module
    """
    # initialize random weights of size = feature_size
    rng = np.random.default_rng()
    w_ = rng.standard_normal(size = 1 + X_train.shape[1])
    cost = []  
    for _ in range(epoch_train):
        if shuffle:
            rng = np.random.default_rng()
            i = rng.permutation(len(y_train))
            X =  X_train[i]
            y =  y_train[i]
            _cost = []
            for x_train, y_train in zip(X, y):
                z_ = w_[0] + np.dot(x_train, w_[1:])
                error = y_train - z_
                _cost.append(0.5*error**2)
                w_[0] += eta_train*error
                w_[1:] += eta_train*np.dot(error, x_train)   
            avg_cost = sum(_cost)/len(_cost)
            cost.append(avg_cost)      
    class_predict = np.where(w_[0] + np.dot(x_train, w_[1:]) >=0.0, 1, -1)
    return cost, w_, class_predict
    

   

