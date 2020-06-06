# author: Dasharath Adhikari
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Logistic regression
# C, Inverse of regularization strength; must be a positive float
lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=100,
                        fit_intercept=True, intercept_scaling=1,
                        class_weight=None, random_state=None, solver='lbfgs',
                        max_iter=100, multi_class='auto', verbose=0,
                        warm_start=False, n_jobs=None, l1_ratio=None)

# fit the model according to the given training data
lr.fit(X_train, y_train, sample_weight=None)
# print(lr.coef_[0])
# print(lr.intercept_)
# print(lr.n_iter_)
# sys.exit()
# predict class labels for samples in X.
predict_class = lr.predict(X_test)
# predicting classification accuracy
list1 = []
for pr, test in zip(predict_class, y_test):
    if pr != test:
        list1.append(pr)
predict_accuracy = 1 - len(list1)/len(y_test)
print(r'Prediction accuracy is:{}'.format(predict_accuracy))

# predict logarithm of probability estimates.
# lr.predict_log_proba(X_test[:5, :])
# lr.predict_proba(X_test)

id = {0: 'sepal length [ab. unit]', 1: 'sepal width [ab. unit]',
      2: 'petal length [ab. unit]', 3: 'petal width [ab. unit]'}

# plotting the decision boundary
plot_decision_boundary(X_std, y, X_test, y_test, classifier=lr)
# print('*'*30)
# x = input('Please enter first feature positional index >> ')
# y = input('Please enter second feature positional index >> ')
# print('*'*30)
plt.xlabel(id[first_feature], fontsize=15)
plt.ylabel(id[second_feature], fontsize=15)
plt.legend()
plt.title(r"Logistic Regression accuracy: {} %".
          format(round(predict_accuracy*100, 2)), fontsize=15)
plt.show()
