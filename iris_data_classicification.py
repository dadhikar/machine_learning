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
# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class: -- Iris Setosa -- Iris Versicolour -- Iris Virginica
df = pd.read_csv(filepath + os.sep+ "iris.data",  skiprows=0, header=None)
#print(df.head())
# selecting all setosa, versicular and virginica class labels 
y = df.iloc[0:150, 4]
#print(y.values)
# y is the target variable, setting its value in binary notation 1 or -1 
y = y.values
y = y[0:100]  # selecting setosa and versicolours only
y = np.where(y == 'Iris-setosa', -1, 1)
#for idx, cl in enumerate(np.unique(y)):
#    print(idx, cl)
#sys.exit()
# extract sepal-length and petal-length and create a feature matrix X
X= df.iloc[0:100, [2, 3]].values
#print(X)



#----------Perceptron_Learning------------------------------------------------------
epoch = 20
learning_rate = 0.01
train_misclassifications, weight_learned, train_class_predict = Perceptron_train(X, y, epoch_n= epoch, eta_lr= learning_rate)
test_predict = Perceptron_test(weight_learned, X)

fig, ax = plt.subplots(1,2, figsize=(8, 5), sharey=False,  dpi= 150)
ax[0].plot(1+np.arange(epoch), train_misclassifications, 'o--', c='r', label = 'Training performance')
ax[1].plot(1+np.arange(len(y)), test_predict, 'D--', c='b', label = 'Test performance')
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
colors = ['skyblue', 'greenyellow', 'm']
markers = ['D', 's', 'v']

cmap = ListedColormap(colors, N=None)
plt.contourf(xx1, xx2, class_prediction, cmap = cmap)
#plt.xlim(X[:, 0].min(), X[:, 0].max())
#plt.ylim(X[:, 1].min(), X[:, 1].max())

test_predict = []
for i in range(X.shape[0]):
    z = weight_learned[0] + np.dot(X[i], weight_learned[1:])
    if z >= 0.0:
        marker = 's'
        color = 'b'
        test_predict.append(1)
    else:
        marker = 'D'
        color = 'g'
        test_predict.append(-1)
    plt.scatter(X[i][0], X[i][1], c= color, marker= marker, s=100, edgecolors='r')    
    #class_predict = np.where(z>=0.0, 1, -1)
    #error_test = y_test - class_predict
    #test_misclassified.append(error)              

# plot class samples
#for idx, cl in enumerate(np.unique(y)):
#    plt.scatter(x= X[y==cl, 0], y= X[y==cl, 1], c= colors[idx], marker= markers[idx], edgecolors='k')

print('*'*30)
x = input('Please enter first feature positional index >> ')
y = input('Please enter second feature positional index >> ')
print('*'*30)
plt.xlabel(x)
plt.ylabel(y)
#plt.legend()
plt.show()
