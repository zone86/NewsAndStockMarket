# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:54:32 2019

@author: cyrzon
"""

import pandas as pd

#import spacy 
#from spacy import displacy
# nlp = spacy.load('en')

import random
import sklearn.metrics as metrics

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# "n-gram" model where n = 2
# create model
ngramVectorizer = CountVectorizer(ngram_range=(2,2))
ngramTrain = ngramVectorizer.fit_transform(trainHeadlines)

ngramModel = LogisticRegression(solver = 'lbfgs')
ngramModel = ngramModel.fit(ngramTrain, train["Label"])
 
ngramTest = ngramVectorizer.transform(testHeadlines)
ngramPredictions = ngramModel.predict(ngramTest)

metrics.accuracy_score(test["Label"], ngramPredictions)
# 0.5079365079365079 # with stop words removed and n = 3
# 0.5634920634920635 # with stop words kept and n = 2
# 0.5714285714285714 # with stop words and .,?! kept and n = 2

ngramWords = ngramVectorizer.get_feature_names()
ngramCoeffs = ngramModel.coef_.tolist()[0]
ngramCoeffDf = pd.DataFrame({'Word' : ngramWords, 
                        'Coefficient' : ngramCoeffs})
ngramCoeffDf = ngramCoeffDf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
ngramCoeffDf.head(10)
ngramCoeffDf.tail(10)


# other models: ridge and lasso, svm, neural net
Classifiers = [
    LogisticRegression(penalty = 'l2', solver = 'lbfgs', max_iter = 100),
    KNeighborsClassifier(5),
    MLPClassifier(),
    SVC(kernel="rbf", probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200)
    ]

#dense_features = ngramTrain.toarray()
#dense_test = test_features.toarray()

Accuracy=[]
Model=[]

for classifier in Classifiers:
    random.seed(1)
    fit = classifier.fit(ngramTrain,train['Label'])
    pred = fit.predict(ngramTest)
    prob = fit.predict_proba(ngramTest)[:,1]
    
    accuracy = metrics.accuracy_score(pred, test['Label'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    
    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))

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
