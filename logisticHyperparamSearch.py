# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:28:50 2019

@author: cyrzon
"""
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize


logSearch = {
        'C': Real(1e-6, 1, prior = 'log-uniform'), 
        'penalty': Categorical(['l1','l2']),
        'max_iter': Integer(100,200)
        }

opt = BayesSearchCV(LogisticRegression(),
              logSearch,
              n_iter = 20,
              cv = 3)

opt.fit(ngramTrain, train["Label"])
print("val. score: %s" % opt.best_score_)
opt.best_estimator_


SVMSearch = {
        'C': Real(1e-6, 1, prior = 'log-uniform'), 
        'gamma': Real(1e-6, 1e+1, prior = 'log-uniform'),
        'degree': Integer(1,5),
        'kernel': Categorical(['linear','poly','rbf'])
        }

optSVC = BayesSearchCV(SVC(),
              SVMSearch,
              n_iter = 20,
              cv = 3)

optSVC.fit(ngramTrain, train["Label"])
print("val. score: %s" % optSVC.best_score_)
opt.best_estimator_

