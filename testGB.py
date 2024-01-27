from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd


df=pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw1/wine.data.csv')
#print(df.shape)
#df.head(5)
data=df.drop('Class',axis=1).to_numpy()
labels=df['Class'].to_numpy()
cat=set(labels)
#print(df.drop('Class',axis=1))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=25)
gnb=GaussianNB()
y_pred=gnb.fit(X_train,y_train).predict(X_test)

res = []
for idx, test_point in enumerate(X_test):
  res.append(y_pred[idx] == y_test[idx])
print(f'Test accuracy for gaussian bayes is {res.count(True)/len(res)}')
