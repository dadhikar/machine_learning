import os
import sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np 
from load_data import load_mat_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10,
                                                   random_state=0)



clf_lin = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001,
                    C=1.0, multi_class='ovr', fit_intercept=True,
                    intercept_scaling=1, class_weight=None, verbose=0,
                    random_state=None, max_iter=1000)

# training the SVM on the data
clf_lin.fit(X_train, y_train)

#  classification performance on test data
y_test_predict = clf_lin.predict(X_test).ravel()
error = np.where(y_test_predict != y_test)

# print(clf.coef_)
# print(clf.intercept_)
w0 = clf_lin.intercept_
w1 = clf_lin.coef_.ravel()[0]
w2 = clf_lin.coef_.ravel()[1]
X1 = X[:, 0].ravel()
X2 = -(w0 + w1*X1)/w2
plt.plot(X1, X2, c='k')
#plt.scatter(X[np.where(y==1), 0], X[np.where(y==1),1], c='r')
#plt.scatter(X[np.where(y==0), 0], X[np.where(y==0),1], c='g')
plt.title('SVM on Linearly separable data')
plt.xlabel('X1 [ab. unit]')
plt.ylabel('X2 [ab. unit]')
#plt.savefig('SVM_on_data1', dpi=150, format='png')


#print(clf.get_params(deep=True))

plot_decision_boundary(X, y, X_test, y_test, classifier=clf_lin)
plt.show()