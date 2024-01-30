import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw1/wine.data.csv')
#print(df.shape)
#df.head(5)

def getMeansDict(data):
    samples_in_class={label : data[y_train==label] for label in cat}
    return {label:np.mean(samples_in_class[label],axis=0) for label in cat}
def getCovsDict(data):
    samples_in_class={label : data[y_train==label] for label in cat}
    return {label:np.cov(samples_in_class[label],rowvar=False) for label in cat}
def getVarsDiagDict(data):
    samples_in_class={label : data[y_train==label] for label in cat}
    return {label : np.diag(np.var(samples_in_class[label],axis=0)) for label in cat}



def plotDensities():
    df.plot(kind='density', subplots=True, layout=(4, 4), figsize=(18, 15), sharex=False)
    plt.show()
#seperating dataframe to data and labels
data=df.drop('Class',axis=1).to_numpy()
labels=df['Class'].to_numpy()
#cat=categories
cat=set(labels)
#train test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=25)
# counting and calculating priors
priors=np.bincount(y_train)[1:]/len(y_train)
means=getMeansDict(X_train)
#normal covariance, not naive
class_cov=getCovsDict(X_train)

#naive gaus 'covariance' (diagonal of variances)
naive_cov=getVarsDiagDict(X_train)
def classify_point_gaussian_bayes(x):
    prob_per_class = {label: -0.5 * ((x - means[label]).T @ np.linalg.inv(class_cov[label]) @ (x-means[label]))
                             - 0.5 * np.log(
      np.linalg.det(class_cov[label])) + np.log(priors[label-1]) for label in cat}
    return max(prob_per_class,key=prob_per_class.get)

def classify_point_gaussian_naive_bayes(x):
    prob_per_class = {label: -0.5 * ((x - means[label]).T @ np.linalg.inv(naive_cov[label]) @ (x - means[label]))
                             - 0.5 * np.log(np.linalg.det(naive_cov[label])) + np.log(priors[label - 1]) for label in cat}
    return max(prob_per_class, key=prob_per_class.get)
print("Unscaled data:")
#testing the gaussian bayes function
def testClassifiers(test_data):
    res = []
    for idx, test_point in enumerate(test_data):
      res.append(classify_point_gaussian_bayes(test_point) == y_test[idx])
    print(f'Test accuracy for gaussian bayes is {res.count(True)/len(res)}')
    #testing the gaussian naive bayes function
    res = []
    for idx, test_point in enumerate(test_data):
      res.append(classify_point_gaussian_naive_bayes(test_point) == y_test[idx])
    print(f'Test accuracy for gaussian naive bayes is {res.count(True)/len(res)}')

testClassifiers(X_test)

'''
def plotCov():
    cmap=plt.get_cmap('coolwarm')
    fig, axs = plt.subplots(len(cat), 1, figsize=(6, 4 * len(cat)))
    for label in cat:
        axs[label-1].imshow(class_cov[label], cmap=cmap, interpolation='nearest',vmin=-1, vmax=1)
        axs[label-1].set_title(f'Covariance Matrix - Label {label}')
        axs[label-1].axis('off')

    plt.tight_layout()
    plt.show()


plotCov()
'''

print(f'\nTesting scaled data but with same means,covs, and variance as original data:')
#scaling the data with StandardScaler
from sklearn.preprocessing import StandardScaler
std_scaler=StandardScaler()
std_scaler.fit(X_train)
X_train_std=std_scaler.transform(X_train)
X_test_std=std_scaler.transform(X_test)
testClassifiers(X_test_std)
print("\ntesting scaled data but with updated (after scaling) means,covs, variances:")
means=getMeansDict(X_train_std)
class_cov=getCovsDict(X_train_std)
naive_cov=getVarsDiagDict(X_train_std)
testClassifiers(X_test_std)