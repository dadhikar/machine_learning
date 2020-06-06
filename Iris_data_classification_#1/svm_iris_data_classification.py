"""
Created on Tue May  5 19:21:42 2020 @author: dadhikar
"""
# author: Dasharath Adhikari
import os
import sys
import pandas as pd
# import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from plot_decision_boundary import plot_decision_boundary
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

filepath = "/Users/dadhikar/Box Sync/GitHub_Repository/machine_learning/data"
# Information about the  Iris data
# Number of Instances: 150 (50 in each of three classes)
iris_data = {0: 'sepal length', 1: 'sepal width', 2: 'petal length',
             3: 'petal width'}
class_label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

df = pd.read_csv(filepath + os.sep + "iris.data",  skiprows=0, header=None)
# target variable
y = df.iloc[:, 4].map(class_label).values
# print(df.head())
# sys.exit()
# feature matrix
print('*'*30)
first_feature = int(input('Please enter first feature >> '))
second_feature = int(input('Please enter second feature >> '))
print('*'*30)
X = df.iloc[:, [first_feature, second_feature]]
# standardization of the feature matrix
std_sc = StandardScaler(copy=True, with_mean=True, with_std=True)
# X_new = std_sc.fit_transform(X)
# Compute the mean and std to be used for later scaling
std_sc.fit(X)
# Perform standardization by centering and scaling and return X
X_std = std_sc.transform(X)  # standardized feature matrix

# random splitting of train and test data
# splitting date for training and test
X_train, X_test, y_train, y_test = train_test_split(X_std, y, train_size=0.8,
                                                    random_state=1,
                                                    shuffle=True,  stratify=y)
# support vector machine classification
svm = SVC(C=1, kernel='rbf', degree=3, gamma=1.0, coef0=0.0,
          shrinking=True, probability=False, tol=0.0001, cache_size=200,
          class_weight=None, verbose=False, max_iter=-1,
          decision_function_shape='ovr', break_ties=False, random_state=1)
# fit the SVM model according to the given training data.
svm.fit(X_train, y_train, sample_weight=None)
# perform classification on samples in X_test.
svm.predict(X_test)
# return the mean accuracy on the given test data and labels.
test_accuracy = svm.score(X_test, y_test)
print(r'The predication accuaracy is {}%'. format(round(test_accuracy*100), 2))

id = {0: 'sepal length [ab. unit]', 1: 'sepal width [ab. unit]',
      2: 'petal length [ab. unit]', 3: 'petal width [ab. unit]'}

# plotting the decision boundary
plot_decision_boundary(X_std, y, X_test, y_test, classifier=svm)
# print('*'*30)
# x = input('Please enter first feature positional index >> ')
# y = input('Please enter second feature positional index >> ')
# print('*'*30)
plt.xlabel(id[first_feature], fontsize=15)
plt.ylabel(id[second_feature], fontsize=15)
plt.legend()
plt.title(r"Accuracy score of SVM: {} %".
          format(round(test_accuracy*100, 2)), fontsize=15)
plt.show()