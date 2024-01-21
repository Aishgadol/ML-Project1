
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
def readTrainData(file_name):
    df=pd.read_csv(file_name,header=None)
    tempTxt=df.drop(0,axis=1).to_numpy()
    texAll=[row[0].split() for index,row in enumerate(tempTxt)]
    voc=[word for line in texAll for word in line]
    lbAll=[lbl for lbl in df[0]]
    voc=set(voc)
    cat=set(lbAll)
    return texAll, lbAll, voc, cat


def naiveBayes_Learn():
    # read the data/labels
    data, labels, unique_words, unique_labels = readTrainData('https://sharon.srworkspace.com/ml/datasets/hw1/cyber_train.csv')
    unique_labels = sorted(unique_labels)
    laplace = 0.25
    # here we create the prior array
    # rememebr: prior is number of label apperances in text div by total number of labels in text
    priorDict = {label: 0.0 for label in unique_labels}
    for label in labels:
        priorDict[label] += 1
    for label in unique_labels:
        priorDict[label] = priorDict[label] / len(labels)
    # here we create the conditional dict of dicts
    #first we count num of appearances per word in each label
    #apply lapalace smoothing with alpha=1 so each word 'appeared' once
    condDict = {label: {word: laplace for word in unique_words} for label in unique_labels}
    totalWordCountPerLabel={label:laplace*len(unique_words)  for label in unique_labels}
    for label,doc in zip(labels,data):
        for word in doc:
            condDict[label][word]+=1
            totalWordCountPerLabel[label]+=1
    for label in unique_labels:
        for word in unique_words:
            condDict[label][word]=condDict[label][word]/totalWordCountPerLabel[label]
    return condDict,priorDict

def naiveBayes_Test(condDict, priorDict):
    # read the data/labels
    data, labels, unique_words, unique_labels = readTrainData('https://sharon.srworkspace.com/ml/datasets/hw1/cyber_test.csv')
    hits = 0
    unique_labels = sorted(unique_labels)
    # set up a default probability for unknown words
    default_prob = np.log(1.0 / len(unique_words))
    # go through each document and compute the predicted label
    # turn all probs to log so we can sum and not multiply
    for label in unique_labels:
        for word in condDict[label]:
            condDict[label][word] = np.log(condDict[label][word])
    for label in unique_labels:
        priorDict[label] = np.log(priorDict[label])
    for label, doc in zip(labels, data):
        # reset dict to hold sum of logs for each label,doc combo in test.txt
        # start off with the prior, which was converted to log
        probsPerDoc = priorDict.copy()
        for eachLabel in unique_labels:
            for word in doc:
                if word in condDict[eachLabel]:
                    probsPerDoc[eachLabel] += condDict[eachLabel][word]
                else:
                    probsPerDoc[eachLabel] += default_prob
        maxLabel = max(probsPerDoc, key=probsPerDoc.get)
        if maxLabel == label:
            hits += 1
    print("\nSuccess rate: ", hits / len(data))
    return

Pw,P=naiveBayes_Learn()
naiveBayes_Test(Pw, P)