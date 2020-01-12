# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:22:36 2019

@author: cyrzon
"""

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

ngramVectorizer = CountVectorizer(ngram_range=(2,2))
ngramTrain = ngramVectorizer.fit_transform(trainHeadlines)

MLPSearch = {
        'hidden_layer_sizes': Integer(40,1000), 
        'alpha': Real(0.0001, 0.9, prior = 'log-uniform'),
        'learning_rate': Categorical(['constant','adaptive'])
        }

optMLP = BayesSearchCV(MLPClassifier(),
              MLPSearch,
              n_iter = 20,
              cv = 3)

optMLP.fit(ngramTrain, train["Label"])
print("val. score: %s" % opt.best_score_)
opt.best_estimator_

