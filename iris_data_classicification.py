# author: Dasharath Adhikari
import os
import sys
import pandas as pd 
import numpy as np 
from perceptron_learning_algorithm import Perceptron_train, Perceptron_test, Perceptron_decision_boundary
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
y = df.iloc[0:100, 4] # select target variable
y = y.values
y = np.where(y == "Iris-versicolor", -1, 1)
X= df.iloc[0:100, [1, 3]].values # select features 
#print(X)
#----------Perceptron_Learning------------------------------------------------------
epoch = 50
learning_rate = 0.01
train_misclassifications, weight_learned, train_class_predict = Perceptron_train(X, y, epoch_n= epoch, eta_lr= learning_rate)
test_predict = Perceptron_test(weight_learned, X)

fig, ax = plt.subplots(1,2, figsize=(8, 5), sharey=False,  dpi= 150)
ax[0].plot(1+np.arange(epoch), train_misclassifications, 'o', c='r', label = 'Training performance')
ax[1].plot(1+np.arange(len(y)), test_predict, '.', c='b', label = 'Test performance')
ax[0].set_xlabel('Training epoch')
ax[0].set_ylabel('Number of misclasscifications')
ax[1].set_xlabel('Testing sample')
ax[1].set_ylabel('Class label')
ax[1].set_ylim(-1.5, 1.5)
ax[0].legend(loc= 'upper right', fontsize= 8) 
ax[1].legend(loc= 'upper right', fontsize= 8)
plt.show()
#sys.exit()

# plotting the decision boundary 
xx1, xx2, class_prediction = Perceptron_decision_boundary(weight_learned, X)
colors = ['skyblue','violet','g','r']
markers = ['D', 's', 'v']
cmap = ListedColormap(colors, N=None)
plt.contourf(xx1, xx2, class_prediction, cmap = cmap)
#plt.xlim(X[:, 0].min(), X[:, 0].max())
#plt.ylim(X[:, 1].min(), X[:, 1].max())
for i in range(X.shape[0]):
    test_predict = []
    z = weight_learned[0] + np.dot(X[i], weight_learned[1:])
    if z >= 0.0:
        marker = markers[0]
        color = colors[2]
        test_predict.append(1)
    else:
        marker = markers[1]
        color = colors[3]
        test_predict.append(-1)
    plt.scatter(X[i][0], X[i][1], c= color, marker= marker, s=100, edgecolors='k')              
print('*'*30)
x = input('Please enter first feature positional index >> ')
y = input('Please enter second feature positional index >> ')
print('*'*30)
plt.xlabel(x)
plt.ylabel(y)
#plt.legend()
plt.show()
