# author: Dasharath Adhikari
import os
import sys
import pandas as pd 
import numpy as np 
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
#sys.exit()
# selecting all setosa, versicular and virginica class labels 
#y = df.iloc[0:150, 4]
#print(y.values)
#sys.exit()
# extract sepal-length and petal-length and create a feature matrix X
X= df.iloc[0:150, [0, 1, 2, 3]].values
sl = X[0:150, 0]
sw = X[0:150, 1]
pl = X[0:150, 2]
pw = X[0:150, 3]
features = [sl, sw, pl, pw]
feature_names  = ['sepal length', 'sepal width', 'petal length', 'petal width']

#print(len(sl))
#sys.exit()
print('*'*30)
i1 = int(input('Please enter first feature positional index >> '))
i2 = int(input('Please enter second feature positional index >> '))
print('*'*30)
plt.plot(features[i1][0:50], features[i2][0:50], 'or', label= 'Setosa')
plt.plot(features[i1][50:100], features[i2][50:100], 'pg', label= 'Vercicular')
plt.plot(features[i1][100:150], features[i2][100:150], 'Db', label= 'Setosa')
plt.xlabel(r'{} (cm)'.format(feature_names[i1]))
plt.ylabel(r'{} (cm)'.format(feature_names[i2]))
plt.title('Iris dataset')
plt.legend()
plt.show()
