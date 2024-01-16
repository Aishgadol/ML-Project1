"""
implementing simple linear regression
we only try to fit our given data, without validations or test
A - 2d matrix sized n x d to represent training samples
b - array of size n which represents target value for corresponding samples
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt


# the lineaar reg calculation function
def Linreg_sol(X, y):
    res = np.linalg.inv(X.T @ X) @ X.T @ y
    return res


data = np.array([[0.4, 3.4], [0.95, 5.8], [0.16, 2.9], [0.7, 3.6], [0.59, 3.27], [0.11, 1.89], [0.05, 4.5]])


# first visualization
def show_points():
    plt.scatter(data[:, 0], data[:, 1], color='blue', label='Data')
    plt.show()


def plotSolution(w, x, y, headline):
    plt.plot(x, y)
    plt.title(headline)
    plt.scatter(data[:, 0], data[:, 1], color='blue', label='Data')
    plt.show()


'''
Here we split the data to A,b
we also center data to mean zero, by removing the mean of A from A and removing the mean of b from b
we flatten w cuz the return type is 2d and we want to work with 1d
also fixed print of linear equation (w --> w[0])
'''
A = data[:, 0]
b = data[:, 1]
x = np.arange(min(A), max(A), 0.01)
# turning A,b from size (n,) to (n,1)
A = A.reshape(-1, 1)
b = b.reshape(-1, 1)
# means of raw data
mean = np.array([np.mean(A), np.mean(b)])
A_centered = A - mean[0]
b_centered = b - mean[1]
# finding the w with zero_mean data
w_zeroMean = Linreg_sol(A_centered, b_centered)
# w here is 2d array which we want to make 1d
w_zeroMean = w_zeroMean.flatten()
# Restore the original line. if y'=wx' (after removing bias) than y-u_y = w(x-u_x), isolate y.
print(f'The linear line when data has zero mean is y={w_zeroMean[0]:.2f}*(x-{mean[0]:.2f})+{mean[1]:.2f}')
y = w_zeroMean * (x - mean[0]) + mean[1]
# plotting the first line
plotSolution(w_zeroMean, x, y, "Line plot with zero mean data:")

# we're going to standarize the data by deducting mean from original data
# and then dividing by standard diviation, without scikit-learn
std = np.array([np.std(A), np.std(b)])
stdzd_A = (A - mean[0]) / std[0]
stdzd_b = (b - mean[1]) / std[1]
w_afterStandarization = Linreg_sol(stdzd_A, stdzd_b).flatten()
y = w_afterStandarization * (x - mean[0]) * std[1] / std[0] + mean[1]
print(
    f'The linear line after standarizing data is y=({w_afterStandarization[0]:.2f}*((x-{mean[0]:.2f})/{std[0]:.2f})*{std[1]:.2f}+{mean[1]:.2f})')
plotSolution(w_afterStandarization, x, y, "Line plot with standarized data:")

'''
we define outliers as points that are located one standard deviation 
above or below the best fit line
we will find and print the outliers in original dataset
'''

bestFitLineY = w_afterStandarization * (A - mean[0]) * std[1] / std[0] + mean[1]
outliers = np.array([data[i] for i in range(len(data)) if np.absolute(b[i] - bestFitLineY[i]) >= std[1]])
outliers_indices = np.array([i for i in range(len(data)) if data[i] in outliers])
print(f'Found {len(outliers)} total outliers:\n{outliers}')

# running linear reg again, without outliers to get a better regression model
new_data = np.delete(data, outliers_indices, axis=0)
newA = new_data[:, 0]
newb = new_data[:, 1]
new_mean = np.array([np.mean(newA), np.mean(newb)])
new_std = np.array([np.std(newA), np.std(newb)])
newA_stdzd = (newA - new_mean[0]) / new_std[0]
newb_stdzd = (newb - new_mean[1]) / new_std[1]
newA_stdzd = newA_stdzd.reshape(-1, 1)
newb_stdzd = newb_stdzd.reshape(-1, 1)
new_w = Linreg_sol(newA_stdzd, newb_stdzd).flatten()
y = new_w * (x - new_mean[0]) * new_std[1] / new_std[0] + new_mean[1]
plotSolution(new_w, x, y, "Line plot with outliers removed and standarized data:")

