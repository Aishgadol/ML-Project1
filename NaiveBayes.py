import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
url='https://sharon.srworkspace.com/ml/datasets/hw1/cyber_train.csv'
df=pd.read_csv(url)
tempTxt=df.drop('sentiment',axis=1).to_numpy()
texAll=[row[0].split() for index,row in enumerate(tempTxt)]

