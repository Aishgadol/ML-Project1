import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
def readTrainData(file_name):
    df=pd.read_csv(file_name,header=None)
    tempTxt=df.drop(0,axis=1).to_numpy()
    texAll=[row[0].split() for index,row in enumerate(tempTxt)]
    voc=[word for line in texAll for word in line]
    lbAll=[lbl for lbl in df[0]]
    voc=set(voc)
    cat=set(lbAll)
    return texAll, lbAll, voc, cat


#this is work in progress:
def learn_NB_text():
    # Implement here
    documents,labels,voc,cat=readTrainData('https://sharon.srworkspace.com/ml/datasets/hw1/cyber_train.csv')
    P=np.zeros(len(cat))
    for index,category in enumerate(cat):
        count=0
        for label in labels:
            if(category==label):
                count+=1
    P[index]=count/len(labels)
    return Pw, P


#testing
docs,labels,voc,cat=readTrainData('https://sharon.srworkspace.com/ml/datasets/hw1/cyber_train.csv')
P=np.zeros(len(cat))
for index, category in enumerate(cat):
    count = sum(1 for label in labels if category == label)
    P[index] = count / len(labels)
Pw=np.zeros((len(cat),len(voc)))
label_word_count={label:{word:0 for word in voc} for label in cat}
wordsInCat=np.zeros(len(cat))
#in this part we are counting number of total words per category in cat
for index, category in enumerate(cat):
    for label_index,label in enumerate(labels):
        if(category == label):
            wordsInCat[index]+=len(docs[label_index])
#here we will count number of appearnces of each word in each categoery
for label in cat:
    for row_index,row in enumerate(docs):
        if label==labels[row_index]:
            for word in row:
                label_word_count[label][word]+=1

#testing to see counting number of words per succeeded
print(sum(len(row) for row in docs))
for label in cat:
    print(f' for label  : {label}, total words per label:{sum(label_word_count[label].values())}')
print(sum(label_word_count[label][word] for label in cat for word in voc))
