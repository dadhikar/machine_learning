# author: Dasharath Adhikari
import os
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True


def sigmoid(w, x):
    return 1/(1 + np.exp(-w*x))

x = np.arange(-5, 5, 0.1)
w = np.arange(1, 21, 2)
for w_ in w:
    plt.plot(x, sigmoid(w_, x), alpha=0.9, markersize=5, label= r'w = {}'.format(w_))
plt.vlines(x=0, ymin= sigmoid(w_, x).min(), ymax= sigmoid(w_, x).max(), colors='k', linestyles='--')
plt.hlines(y = 0.5, xmin= x.min(), xmax= x.max(), colors='k', linestyles='--')    
plt.xlabel('x')
plt.ylabel('Sigmoid function')
plt.text(0.6, 0.2, r'$\frac{1}{1 + e^{(-w*x})}$', size=20, rotation=0,
                           bbox=dict(boxstyle="round", ec='k', fc='g'))
plt.title('Approximation to the step-function')
plt.xlim(xmin= x.min(), xmax= x.max())
plt.ylim(ymin= sigmoid(w_, x).min()-.02, ymax= sigmoid(w_, x).max()+0.02)
plt.legend(loc='best')
plt.show()