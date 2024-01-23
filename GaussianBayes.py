import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw1/wine.data.csv')
#print(df.shape)
#df.head(5)

def plotDensities():
    df.plot(kind='density', subplots=True, layout=(4, 4), figsize=(18, 15), sharex=False)
    plt.show()

