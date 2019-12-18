import pandas as pd

#import spacy 
#from spacy import displacy
# nlp = spacy.load('en')

import random
import sklearn.metrics as metrics

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# add trainHeadlines to the list for CountVectorizer and center the data
basicVectorizer = CountVectorizer()
basictrain = basicVectorizer.fit_transform(trainHeadlines)

# logistic regression: basic "bag of words" model
# create model
basicMod = LogisticRegression(solver = "lbfgs", max_iter = 100)
basicMod = basicMod.fit(basictrain, train["Label"])

# test model, assess performance and view coefficients
basicTest = basicVectorizer.transform(testHeadlines)
basicPredictions = basicMod.predict(basicTest)

metrics.accuracy_score(test["Label"], basicPredictions)
# 0.3968253968253968 # with stop words removed
# 0.42857142857142855 # with stop words kept
# 0.4312169312169312 # with stop words and .,?! kept

basicWords = basicVectorizer.get_feature_names()
basicCoeffs = basicMod.coef_.tolist()[0]
coeffDf = pd.DataFrame({'Word' : basicWords, 
                        'Coefficient' : basicCoeffs})
coeffDf = coeffDf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
coeffDf.head(10)

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

    
# hashing vectorizer
hashingVectorizer = HashingVectorizer(n_features=50)
hashingTrain = hashingVectorizer.fit_transform(trainHeadlines)

hashingModel = LogisticRegression(solver = 'lbfgs')
hashingModel = hashingModel.fit(hashingTrain, train["Label"])

hashingTest = hashingVectorizer.transform(testHeadlines)
hashingPreds = hashingModel.predict(hashingTest)

metrics.accuracy_score(test["Label"], hashingPreds)