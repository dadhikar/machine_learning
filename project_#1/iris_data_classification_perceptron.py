#author: Dasharath Adhikari
import os
import sys
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron 
from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

filepath = "/Users/dadhikar/Box Sync/GitHub_Repository/machine_learning/data"
# Information about the  Iris data
# Number of Instances: 150 (50 in each of three classes)
# 0. sepal length in cm
# 1. sepal width in cm
# 2. petal length in cm
# 3. petal width in cm
# 4. class: -- Iris Setosa -- Iris Versicolour -- Iris Virginica
df = pd.read_csv(filepath + os.sep+ "iris.data",  skiprows=0, header=None)
#print(df.head())
#sys.exit()
# feature matrix
X = df.iloc[:, [2, 3]] 
# target variable 
class_label = {'Iris-setosa': 0, 
               'Iris-versicolor': 1,
               'Iris-virginica': 2}
y = df.iloc[:, 4].map(class_label).values
# standardization of the feature matrix
std_sc = StandardScaler(copy=True, with_mean=True, with_std=True)
#X_new = std_sc.fit_transform(X)
#Compute the mean and std to be used for later scaling
std_sc.fit(X)  
#print(std_sc.mean_)
#print(std_sc.var_)
#Perform standardization by centering and scaling and return X
X_std = std_sc.transform(X) #standardized feature matrix

# random splitting of train and test data
# splitting date for training and test
X_train, X_test, y_train, y_test = train_test_split(X_std, y, train_size=0.8, random_state= 1, 
                                                     shuffle=True,  stratify= y)

# training the perceptron algorithm
pc_lc = Perceptron(max_iter=50, eta0=0.1, random_state=1)
pc_lc.fit(X_train, y_train)
# predicting the performance on test data
y_predict = pc_lc.predict(X_test)
print(r"The accuracy score of the classification : {} %". format(round(accuracy_score(y_test, y_predict)*100, 2)))


def perceptron_decision_boundary(X, y, X_test, y_test, classifier):
    """
    """
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
    x1_grid = xx1.ravel()
    x2_grid = xx2.ravel()
    x1x2_grid = np.array([x1_grid, x2_grid]).T  # x1x2_grid contains each possible combination of x and y value
    y_predict = classifier.predict(x1x2_grid)
    y_predict = y_predict.reshape(xx1.shape) 
    # plotting the decision boundary 
    colors = ['C0','C1','C2','C3', 'C4', 'C5']
    markers = ["o", "v", "*", "H", "h" ]
    cmap = ListedColormap(colors, N=None)
    plt.contourf(xx1, xx2, y_predict, cmap = cmap, alpha=0.5)
    #plotting test samples 
    for idx, cl in enumerate(np.unique(y_test)):  
        plt.scatter(x= X_test[y_test== cl, 0], y= X_test[y_test== cl, 1], c= colors[idx], 
                       marker= markers[idx], s=100, alpha = 1, edgecolor='b', label= cl)              
    print('*'*30)
    x = input('Please enter first feature positional index >> ')
    y = input('Please enter second feature positional index >> ')
    print('*'*30)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
      

# plotting the decision boundary 
perceptron_decision_boundary(X_std, y, X_test, y_test, pc_lc)
plt.title(r"Accuracy score of the classcification : {} %". format(round(accuracy_score(y_test, y_predict)*100, 2)))
#plt.xlim(X_std[:, 0].min()-1, X_std[:, 0].max()+1)
#plt.ylim(X_std[:, 1].min()-1, X_std[:, 1].max()+1) 
plt.show()