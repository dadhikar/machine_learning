import os
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np 
from load_data import load_mat_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from plot_decision_boundary import plot_decision_boundary


# current working directory
cwd = os.getcwd()
# directory for the data file
file_dir = cwd + os.sep + 'data_sets'

data = load_mat_data(file_dir, 'data1.mat')

X = data['X']
y = np.ravel(data['y'])

#sys.exit()

# splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30,
                                                   random_state=0)

clf = SVC(C=10, kernel='linear', degree=3, gamma='scale', coef0=0.0,
          shrinking=True, probability=False, tol=0.001, cache_size=200,
          class_weight=None, verbose=False, max_iter=-1,
          decision_function_shape='ovr', break_ties=False, random_state=None)

# training the SVM on the data
clf.fit(X_train, y_train)
#  classification performance on test data
y_test_predict = clf.predict(X_test).ravel()
error = np.where(y_test_predict != y_test)
# print(clf.coef_)
# print(clf.intercept_)
w0 = clf.intercept_
w1 = clf.coef_.ravel()[0]
w2 = clf.coef_.ravel()[1]
X1 = X[:, 0].ravel()
X2 = -(w0 + w1*X1)/w2
plt.plot(X1, X2, c='k')
plt.scatter(X[np.where(y==1), 0], X[np.where(y==1),1], c='r')
plt.scatter(X[np.where(y==0), 0], X[np.where(y==0),1], c='g')
plt.title('SVM on Linearly separable data')
plt.xlabel('X1 [ab. unit]')
plt.ylabel('X2 [ab. unit]')
#plt.savefig('SVM_on_data1', dpi=150, format='png')


#print(clf.get_params(deep=True))

plot_decision_boundary(X, y, X, y, classifier=clf)
plt.show()