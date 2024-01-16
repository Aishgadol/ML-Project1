"""
implementing simple linear regression
we only try to fit our given data, without validations or test
A - 2d matrix sized n x d to represent training samples
b - array of size n which represents target value for corresponding samples
"""
#import libraries
import numpy as np
import matplotlib.pyplot as plt

#the lineaar reg calculation function
def Linreg_sol(X,y):
    res=np.linalg.inv(X.T @ X) @ X.T @ y
    return res

data = np.array([[0.4, 3.4], [0.95, 5.8], [0.16, 2.9], [0.7, 3.6], [0.59, 3.27], [0.11, 1.89], [0.05, 4.5]])
#first visualization
def show_points():
    plt.scatter(data[:, 0], data[:, 1], color='blue', label='Data')
    plt.show()

