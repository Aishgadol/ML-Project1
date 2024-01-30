import numpy as np
import pandas as pd

def readTrainData(file_name):
    df=pd.read_csv(file_name,header=None)
    tempTxt=df.drop(0,axis=1).to_numpy()
    texAll=[row[0].split() for index,row in enumerate(tempTxt)]
    voc=[word for line in texAll for word in line]
    lbAll=[lbl for lbl in df[0]]
    voc=set(voc)
    cat=set(lbAll)
    return texAll, lbAll, sorted(voc), sorted(cat)

def learn_NB_text():
    docs,labels,voc,cat=readTrainData('cyber_train.csv')
    laplace=0.35
    priorDict={label:0.0 for label in cat}
    for label in labels:
        priorDict[label]+=1
    for category in cat:
        priorDict[category] = priorDict[category] / len(labels)
    condDict={label:{word:laplace for word in voc} for label in cat}
    words_in_cat={label:laplace*len(voc) for label in cat} #in this part we are counting number of total words per category in cat
    #here we will count number of appearnces of each word in each categoery
    for row_index, row in enumerate(docs):
        for word in row:
            condDict[labels[row_index]][word]+=1
            words_in_cat[labels[row_index]]+=1
    for label in cat:
        for word in voc:
            condDict[label][word] = (condDict[label][word])/(words_in_cat[label])
    return condDict,priorDict


def ClassifyNB_test(condDict,priorDict):
    docs,testLabels, falseVoc,cat=readTrainData('cyber_test.csv')
    for label in cat:
        priorDict[label] = np.log(priorDict[label])
        for word in condDict[label]:
            condDict[label][word] = np.log(condDict[label][word])
    #default probability in case test docs has words we didnt see in train docs
    default_prob=np.log(1.0/len(falseVoc))
    hits=0
    for row_index,row in enumerate(docs):
        probsPerRow={label:0.0 for label in cat}
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
    print(f' Success rate: {hits/len(docs)}')
    return hits/len(docs)
#need to improve accuracy
condDict,priorDict = learn_NB_text()
ClassifyNB_test(condDict,priorDict)
