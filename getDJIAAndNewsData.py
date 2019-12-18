# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 19:27:33 2019

@author: cyrzon
"""
import pandas as pd
import numpy as np
import os
import re
from nltk.corpus import stopwords

os.chdir('C:\\Users\\nija\\Documents\\NewsAndStockMarket')

combinedNewsAndDJIA = pd.read_csv('Combined_News_DJIA.csv')
combinedNewsAndDJIA = combinedNewsAndDJIA.replace(np.nan, '', regex=True)

dateCounts = combinedNewsAndDJIA.pivot_table(index=['Date'], aggfunc='size')

# create train and test data frame
train = combinedNewsAndDJIA[combinedNewsAndDJIA['Date'] < '2015-01-01']
test = combinedNewsAndDJIA[combinedNewsAndDJIA['Date'] > '2014-12-31']

# pre process the data: fix contractions, remove stopwords, remove unwanted punctuation
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

# function to clean text. models perform better with stop words set to false
def clean_text(text, remove_stopwords = False):
    
    # convert to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'0,0', '00', text) 
    #text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'[_"\-;%()|+&=*%:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
            
    return text

# clean train and test headlines
cleaned_headlines_train = []
for i in range(0,len(train)):
    row_headlines = train.iloc[i,2:27]
    clean_row_headlines = []
    for j in range(0,25):
        string = clean_text(row_headlines[j])
        clean_row_headlines.append(''.join(str(x) for x in string))
        
    cleaned_headlines_train.append(clean_row_headlines)
    
    
trainDf = pd.DataFrame(cleaned_headlines_train) 

trainHeadlines = []

for row in range(0,len(trainDf.index)):
    trainHeadlines.append(' '.join(
            str(x)
            for x in trainDf.iloc[row,0:25]) )
    
cleaned_headlines_test = []
for i in range(0,len(test)):
    row_headlines = test.iloc[i,2:27]
    clean_row_headlines = []
    for j in range(0,25):
        string = clean_text(row_headlines[j])
        clean_row_headlines.append(''.join(str(x) for x in string))
        
    cleaned_headlines_test.append(clean_row_headlines)
    
    
testDf = pd.DataFrame(cleaned_headlines_test) 

testHeadlines = []

for row in range(0,len(testDf.index)):
    testHeadlines.append(''.join(
            str(x) 
            for x in testDf.iloc[row,0:25]) )