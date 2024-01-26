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
#print(df.drop('Class',axis=1))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=25)
priors=np.bincount(y_train)[1:]/len(y_train)
samples_in_class={label : X_train[y_train==label] for label in cat}
means={label :np.zeros((len(data[0]))) for label in cat}
for label in cat:
    means[label]=np.mean(samples_in_class[label],axis=0)

test_means={label :np.mean(samples_in_class[label],axis=0)  for label in cat}
for i in cat:
    print(means[i]-test_means[i])
#normal covariance, not naive
class_cov={label:np.cov(samples_in_class[label],rowvar=False) for label in cat}

#naive gaus 'covariance' (diagonal of variances)
naive_cov={label: np.diag(np.var(samples_in_class[label], axis=0))for label in cat}

for label in cat:
    print(class_cov[label])
'''
def classify_point_gaussian_bayes(x):
  prob_per_class = {label: -0.5 * ((x - means[label]).T) for label in cat}


def classify_point_gaussian_naive_bayes(x):

'''

def plotCov():
    cmap=plt.get_cmap('coolwarm')
    fig, axs = plt.subplots(len(cat), 1, figsize=(6, 4 * len(cat)))
    for label in cat:
        axs[label-1].imshow(naive_cov[label], cmap=cmap, interpolation='nearest',vmin=-1, vmax=1)
        axs[label-1].set_title(f'Covariance Matrix - Label {label}')
        axs[label-1].axis('off')

    plt.tight_layout()
    plt.show()


plotCov()