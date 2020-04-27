# author: Dasharath Adhikari
import os
import sys
import pandas as pd 
import numpy as np 
from one_layer_neuron_learning_algorithm import feature_scale
from one_layer_neuron_learning_algorithm import perceptron_train, perceptron_test, perceptron_decision_boundary
from one_layer_neuron_learning_algorithm import adaline_train_batch_descent, adaline_train_stochastic_descent
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
y = np.where(y == "Iris-setosa", -1, 1)
X = df.iloc[0:100, [0, 2]].values # select features 
#feature scaling
X_n = feature_scale(X) 

#---------------Adaline_Learning------------------------------------------------------
#cost, w_, class_predict = adaline_train_batch_descent(X_n, y, 20, 0.01)
cost, w_, class_predict = adaline_train_stochastic_descent(X_n, y, epoch_train= 20, eta_train= 0.01)
fig, ax = plt.subplots(1,1, figsize=(4, 4), sharey=False,  dpi= 100)
ax.plot(1+np.arange(len(20)), cost, 'o', c='r', label = 'Training performance')
#ax.plot(1+np.arange(len(y)), class_predict, '.', c='b', label = 'Test performance')
ax.set_xlabel('Training epoch')
ax.set_ylabel('Error Function')
#ax[1].set_xlabel('Testing sample')
#ax[1].set_ylabel('Class label')
#ax[1].set_ylim(-1.5, 1.5)
ax.legend(loc= 'upper right', fontsize= 8) 
#ax[1].legend(loc= 'upper right', fontsize= 8)
plt.title("Adaline Algorithms Learning")
plt.show()
sys.exit()

#----------Perceptron_Learning------------------------------------------------------
epoch = 20
learning_rate = 0.01
train_misclassifications, w_learned, train_class_predict = perceptron_train(X_n, y, epoch_train= epoch, eta_train= learning_rate)
test_error = perceptron_test(w_learned= w_learned, X_test= X_n, y_test= y)
fig, ax = plt.subplots(1,2, figsize=(10, 4), sharey=False,  dpi= 100)
ax[0].plot(1+np.arange(epoch), train_misclassifications, 'o', c='r', label = 'Training performance')
ax[1].plot(1+np.arange(len(y)), test_error, 'go', label = 'Test performance')
ax[0].set_xlabel('Training epoch')
ax[0].set_ylabel('Number of misclasscifications')
ax[1].set_xlabel('Testing sample [count]')
ax[1].set_ylabel('Testing Errors')
ax[1].set_ylim(-1.5, 1.5)
ax[0].legend(loc= 'upper right', fontsize= 8) 
ax[1].legend(loc= 'upper right', fontsize= 8)
plt.title("Perceptron Algorithms Learning")
plt.show()

# plotting the decision boundary 
perceptron_decision_boundary(w_learned, X_n, y)
#-------------------------------------------------------------------------------------

