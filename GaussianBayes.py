import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw1/wine.data.csv')
#print(df.shape)
#df.head(5)


def plotDensities():
    df.plot(kind='density', subplots=True, layout=(4, 4), figsize=(18, 15), sharex=False)
    plt.show()
data=df.drop('Class',axis=1).to_numpy()
labels=df['Class'].to_numpy()
cat=set(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=25)
priors=np.bincount(y_train)[1:]/len(y_train)
samples_in_class={label : X_train[y_train==label] for label in cat}
means={label :np.zeros((len(data[0]))) for label in cat}
for label in cat:
    means[label]=np.mean(samples_in_class[label],axis=0)

test_means={label :np.mean(samples_in_class[label],axis=0)  for label in cat}
#normal covariance, not naive
class_cov={label:np.cov(samples_in_class[label],rowvar=False) for label in cat}

#naive gaus 'covariance' (diagonal of variances)
naive_cov={label: np.diag(np.var(samples_in_class[label], axis=0))for label in cat}

x=X_test[0]
print(f' shapes: ((x - means[label]).T).shape={((x - means[1]).T).shape}\nnp.linalg.inv(class_cov[label])={np.linalg.inv(class_cov[1]).shape}'
      f'\n(x-means[label]).shape={(x-means[1]).shape} ')
def classify_point_gaussian_bayes(x):
    prob_per_class = {label: -0.5 * ((x - means[label]).T @ np.linalg.inv(class_cov[label]) @ (x-means[label]))
                             - 0.5 * np.log(
      np.linalg.det(class_cov[label])) + np.log(priors[label-1]) for label in cat}
    return max(prob_per_class,key=prob_per_class.get)

def classify_point_gaussian_naive_bayes(x):
    prob_per_class = {label: -0.5 * ((x - means[label]).T @ np.linalg.inv(naive_cov[label]) @ (x - means[label]))
                             - 0.5 * np.log(np.linalg.det(naive_cov[label])) + np.log(priors[label - 1]) for label in cat}
    return max(prob_per_class, key=prob_per_class.get)

res = []
for idx, test_point in enumerate(X_test):
  res.append(classify_point_gaussian_bayes(test_point) == y_test[idx])
print(f'Test accuracy for gaussian bayes is {res.count(True)/len(res)}')

res = []
for idx, test_point in enumerate(X_test):
  res.append(classify_point_gaussian_naive_bayes(test_point) == y_test[idx])
print(f'Test accuracy for gaussian naive bayes is {res.count(True)/len(res)}')
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