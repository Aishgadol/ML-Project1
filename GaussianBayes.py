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
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=25)