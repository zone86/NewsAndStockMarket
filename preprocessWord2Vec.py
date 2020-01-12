# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 13:27:14 2020

@author: cyrzon
"""
import pandas as pd
import numpy as np
import os

import gensim
import pdb

from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from gensim.models import word2vec

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


os.chdir('C:\\Users\\nija\\Documents\\NewsAndStockMarket')

combinedNewsAndDJIA = pd.read_csv('Combined_News_DJIA.csv')
combinedNewsAndDJIA = combinedNewsAndDJIA.replace(np.nan, '', regex=True)

dateCounts = combinedNewsAndDJIA.pivot_table(index=['Date'], aggfunc='size')

# create train and test data frame
train = combinedNewsAndDJIA[combinedNewsAndDJIA['Date'] < '2015-01-01']
test = combinedNewsAndDJIA[combinedNewsAndDJIA['Date'] > '2014-12-31']


def news_to_words(news, removeStopwords = False):
    '''
    Function to convert news title to strings of words.
    The input is a Dataframe (news title), and
    the output is an array (preprocessed news).
    Based on scikit-learn the Bag of Words model
    :param news: a Dataframe of news
    :return: headlines: word list for everyday news
    '''
    # Removing punctuations
    news.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)

    # Renaming column names for ease of access
    list = [i for i in range(len(news.columns))]
    newIndex = [str(i) for i in list]
    news.columns = newIndex

    # Show all the stopwords
    stops = set(stopwords.words("english"))

    # Convertng headlines to lower case
    for index in newIndex:
        news[index] = news[index].str.lower()

    sentences = []
    for row in range(0, len(news.index)):
        headline = []
        for x in news.iloc[row, 0:len(news.columns)]:
            headline = headline + str(x).strip().lstrip('b').split()

        if removeStopwords:
            headline = [w for w in headline if not w in stops]

        sentences.append(headline)

    return sentences


def makeFeatureVec(words, model, num_features):
    '''
    # Function to average all of the word vectors in a given paragraph
    :param words: a list of words
    :param model: the word2vec model (after training)
    :param num_features: the dimension of each word vector
    :return: a vector of the average of all of the word vectors in a given paragraph
    '''

    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    
    #pdb.set_trace()

    # Loop over each word in the news and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
            
    

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)

    return featureVec


def getAvgFeatureVecs(news, model, num_features):
    '''
    Given a set of news (each one a list of words), calculate
    the average feature vector for each one and return a 2D numpy array.
    Initialize a counter
    :param news: word list
    :param model: the word2vec model (after training)
    :param num_features: the dimension of each word vector
    :return: everyday news vector (2D array)
    '''
    counter = 0

    # Preallocate a 2D numpy array, for speed
    newsFeatureVecs = np.zeros((len(news),num_features),dtype="float32")
    # Loop through the reviews
    for row in news:
       #pdb.set_trace()
        # Call the function (defined above) that makes average feature vectors
       newsFeatureVecs[counter] = makeFeatureVec(row, model, num_features)
       # Increment the counter
       counter = counter + 1

    return newsFeatureVecs


trainData = train.iloc[:, 2:(len(combinedNewsAndDJIA.columns) - 1)]
trainNews = news_to_words(trainData, removeStopwords=True)

testData = test.iloc[:, 2:(len(combinedNewsAndDJIA.columns) - 1)]
testNews =  news_to_words(testData, removeStopwords=True)

w2v_model = Word2Vec(min_count = 10,
                     size = 300,
                     workers = 3, 
                     window = 2, 
                     sg = 0)

w2v_model.build_vocab(trainNews, progress_per=10000)

trainVecs = getAvgFeatureVecs(trainNews, w2v_model, num_features=300)
testVecs = getAvgFeatureVecs(testNews, w2v_model, num_features=300)

 


w2v_LogModel = LogisticRegression(solver = 'lbfgs')
w2v_LogModel = w2v_LogModel.fit(trainVecs, train["Label"])

predictions_logMod = w2v_LogModel.predict(testVecs)
metrics.accuracy_score(test["Label"], predictions_logMod)


ngramTest = ngramVectorizer.transform(testHeadlines)
ngramPredictions = ngramModel.predict(ngramTest)
