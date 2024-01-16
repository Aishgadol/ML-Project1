
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

'''
we want to detect type of stars into 6 diff types by measuring properties
NASA gave us dataset (tutor is nasa appearntly), including temperature,color, spectral class and more
we aim to compare difference metric to dtermine which one is best for this data
'''
#load data and printing first 3 rows
url="https://sharon.srworkspace.com/ml/datasets/hw1/Stars.csv"
data=pd.read_csv(url,sep=",")
data=data.dropna()
#print(data.head(3))

#converting categorical features to discrete values
#note: Type (prediction) is int
colors = data['Color'].unique()
for idx, color in enumerate(colors):
  data['Color'] = data['Color'].replace({color: idx})

spec_class = data['Spectral_Class'].unique()
for idx, spec in enumerate(spec_class):
  data['Spectral_Class'] = data['Spectral_Class'].replace({spec: idx})

#checkin the correlation matrix
corr_mat=data.corr()
#print(corr_mat)

#splittin the data into 80% train 20% test with random state=21
X = data.drop('Type', axis=1)
y = data['Type']
num_features=len(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21,stratify=y)
X_train=X_train.to_numpy()
X_test=X_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()
#this isnt the most efficient way, but its the most thorough approach
def Euclidian(test,train):
    distances=np.zeros((test.shape[0],train.shape[0]))
    #i represents each row (each sample in test)
    #j represnts each col (each sample in train)
    for i in range(test.shape[0]):
        for j in range(train.shape[0]):
            dist=0
            for k in range(num_features):
                dist+=pow((test[i,k]-train[j,k]),2)
            distances[i, j] = dist
            dist=0
    return np.sqrt(distances)

#manhattan distance d(x,y) is sum(i;0 -> num_features) abs(x_i-y_i), simpler, not sure on usage
def Manhattan(test,train):
    distances=np.zeros((test.shape[0],train.shape[0]))
    for i in range(test.shape[0]):
        for j in range(train.shape[0]):
            sum=0
            for k in range(num_features):
                sum+=abs(test[i,k]-train[j,k])
            distances[i,j]=sum
    return distances

#mahalanobis distance, useful because its directly connected to the distribution and affected by cov matrix
def Mahalanobis(test, train):
    distances = np.zeros((test.shape[0], train.shape[0]))
    covariance_matrix_data = np.cov(train, rowvar=False)
    # Calculate the Mahalanobis distances
    for i in range(test.shape[0]):
        for j in range(train.shape[0]):
            diff =  test[i] - data[j]
            distances[i, j] = np.sqrt(np.dot(np.dot(diff, np.linalg.inv(covariance_matrix_data)), diff.T))
    return distances

#here we will implmenet the KNN_classify function, returns array sized as the number of test samples
#because we're going to classify each point in test set
def kNN_classify(train,labels,test,k,metric='Euclidian'):
    arguments = (test,train)
    distances = eval(f'{metric}(*arguments)')#returns np[][] |test| X |train| by the given metric.
    #counts will hold the predictions of k neighbors
    counts=np.zeros((num_features))
    results=np.zeros((X_test.shape[0]))
    for row_index, row in enumerate(distances):
        #this argsort is pure cheating i swear
        smallest_k_indices=np.argsort(row)[:k]
        for i in smallest_k_indices:
            counts[labels[i]]+=1
        results[row_index]=np.argmax(counts)
        counts = np.zeros((num_features))
    return results


#testing

metrics=['Euclidian','Manhattan','Mahalanobis']
# Apply kNN classification using your custom function
for k_value in range(1,10):
    for metric in metrics:
        predictions_custom = kNN_classify(X_train, y_train, X_test, k_value,metric=metric)

        # Use scikit-learn's KNeighborsClassifier for comparison
        knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
        knn_classifier.fit(X_train, y_train)
        predictions_sklearn = knn_classifier.predict(X_test)

        # Compare the results
        accuracy_custom = accuracy_score(y_test, predictions_custom)
        accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)
        print(f'k is: {k_value} , metric is: {metric}')
        print(f'Accuracy using custom kNN function: {accuracy_custom:.7f}')
        print(f'Accuracy using scikit-learn KNeighborsClassifier: {accuracy_sklearn:.7f}\n')
