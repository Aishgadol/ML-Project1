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

def learn_NB_text():
    docs,labels,voc,cat=readTrainData('https://sharon.srworkspace.com/ml/datasets/hw1/cyber_train.csv')
    laplace=0.2
    d=len(cat)
    priorDict={label:0 for label in cat}
    for category in cat:
        count = sum(1 for label in labels if category == label)
        priorDict[category] = count / len(labels)
    condDict={label:{word:laplace for word in voc} for label in cat}
    words_in_cat={label:0 for label in cat} #in this part we are counting number of total words per category in cat
    for category in cat:
        for label_index,label in enumerate(labels):
            if(category == label):
                words_in_cat[label]+=len(docs[label_index])
    #here we will count number of appearnces of each word in each categoery
    for label in cat:
        for row_index,row in enumerate(docs):
            if label==labels[row_index]:
                for word in row:
                    condDict[label][word]+=1
    for label in cat:
        for word in voc:
            condDict[label][word] = (condDict[label][word])/(words_in_cat[label]+laplace*d)

    return condDict,priorDict

def ClassifyNB_test(condDict,priorDict):
    docs,testLabels, falseVoc,cat=readTrainData('https://sharon.srworkspace.com/ml/datasets/hw1/cyber_test.csv')
    for label in cat:
        priorDict[label] = np.log(priorDict[label])
        for word in condDict[label]:
            condDict[label][word] = np.log(condDict[label][word])
    #default probability in case test docs has words we didnt see in train docs
    default_prob=np.log(1.0/len(falseVoc))
    hits=0
    for row_index,row in enumerate(docs):
        probsPerRow={label:0 for label in cat}
        for unique_cat in cat:
            for word in row:
                if(word in condDict[unique_cat]):
                    probsPerRow[unique_cat]+=condDict[unique_cat][word]
                else:
                    probsPerRow[unique_cat]+=default_prob
            probsPerRow[unique_cat]+=priorDict[label]
        bestLabel=""
        maxProb=probsPerRow[testLabels[0]]
        for label in cat:
            if(probsPerRow[label]>maxProb):
                bestLabel=label
                maxProb=probsPerRow[label]
        if(bestLabel==testLabels[row_index]):
            hits+=1
    print(f'Succes rate: {hits/len(docs)}')
    return

condDict,priorDict=learn_NB_text()
ClassifyNB_test(condDict,priorDict)

