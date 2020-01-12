# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:52:53 2019

@author: cyrzon
"""
import pandas as pd

#import spacy 
#from spacy import displacy
# nlp = spacy.load('en')

import random
import sklearn.metrics as metrics

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

################################################################################
# tf-idf vectorizer
Tfidf_Vectorizer = TfidfVectorizer( min_df=0.03, max_df=0.97, ngram_range = (2,2))
TfidfTrain = Tfidf_Vectorizer.fit_transform(trainHeadlines)

TfidfModel = LogisticRegression(solver = 'lbfgs')
TfidfModel = TfidfModel.fit(TfidfTrain, train["Label"])

TfidfTest = Tfidf_Vectorizer.transform(testHeadlines)
TfidfPreds = TfidfModel.predict(TfidfTest)

metrics.accuracy_score(test["Label"], TfidfPreds)
# 0.5423280423280423 # with stop words removed and n = 3
# 0.5687830687830688 # with stop words kept, n = 2, min = .03, max .97

nonRandmClassifiers = [
    LogisticRegression(solver = 'lbfgs'),
    KNeighborsClassifier(2),
    SVC(kernel="rbf", probability=True)
    ]

randmClassifiers = [
    MLPClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200)
    ]

#dense_features = ngramTrain.toarray()
#dense_test = test_features.toarray()

TfidfAccuracy=[]
TfidfModel=[]

for classifier in randmClassifiers:
    for x in range(0,10):
        random.seed(x)
        fit = classifier.fit(TfidfTrain,train['Label'])
        pred = fit.predict(TfidfTest)
        prob = fit.predict_proba(TfidfTest)[:,1]
        
        accuracy = metrics.accuracy_score(pred, test['Label'])
        TfidfAccuracy.append(accuracy)
        TfidfModel.append(classifier.__class__.__name__)
        
        print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))

rndmModelPerf = pd.DataFrame({
        'Model': TfidfModel,
        'Acc': TfidfAccuracy
        })
