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

def logistic_cost_fcn(y, phi_z):
    """
    calulating cost function for 
    predicting class label for a single
    target varaible
    """
    if y == 1:
        f1 = -np.log(phi_z)
    elif y == 0:
        f1 = -np.log(1 - phi_z) 
    else:
        print('Can not predict')
    return f1 

z = np.arange(-5, 5, 0.1)
phi_z = sigmoid(1, z)
plt.plot(phi_z, logistic_cost_fcn(1, phi_z), alpha=0.9, markersize=5, label= r'True class label = {}'.format(1))
plt.plot(phi_z, logistic_cost_fcn(0, phi_z), alpha=0.9, markersize=5, label= r'True class label = {}'.format(0)) 
#plt.vlines(x=0, ymin=logistic_cost_fcn(1, phi_z).min(), ymax= logistic_cost_fcn(1, phi_z).max(), colors='k', linestyles='--')
#plt.vlines(x=1, ymin= logistic_cost_fcn(1, phi_z).min(), ymax= logistic_cost_fcn(1, phi_z).max(), colors='k', linestyles='--') 
plt.xlabel('Sigmoid Output')
#plt.xlabel('Net Input [z]')
plt.ylabel('Cost Function')
plt.title('One Target Value Logistic Cost Function')
plt.legend(loc='best')
plt.grid(which='major', axis= 'both', color='k', linestyle='--', linewidth=0.5, alpha=0.8 )
plt.tight_layout()
plt.show()