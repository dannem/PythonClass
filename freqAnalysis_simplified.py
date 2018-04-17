# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 18:20:24 2018

@author: Dan
"""

folder='C:/Users/Dan/Documents/GitHub/PythonClass'

import os
os.chdir(folder)
os.getcwd()
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import auxfuns as af
import timeit
import pandas as pd

#%% loading data
folder='C:/Users/Dan/Documents/MATLAB'
os.chdir(folder)

data=scipy.io.loadmat('S09_fft.mat')
data=data['S09_fft']
data=np.transpose(data.reshape(64*40,80*16,order='F'))

#data=scipy.io.loadmat('S09_fft_2D.mat')
#data=data['data']
#dataM=np.transpose(data)

labels=np.tile(np.arange(1,81),16)
#%% scaling
from sklearn.preprocessing import maxabs_scale
data = maxabs_scale(data)

#%% SVM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,  cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, mean_squared_error
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.svm import SVC
import itertools
start = time.time()
chi_k = 'all'
scores_all=dict()
scores=list()
for i in itertools.combinations(np.arange(1,81), 2):
     print(i)
     X=data[np.logical_or(labels==i[0],labels==i[1]),:]
     y=labels[np.logical_or(labels==i[0],labels==i[1])]
     # Setup the pipeline steps: steps
     steps = [('clf', SVC(kernel='linear'))]
#     steps = [('dim_red', SelectKBest(chi2, chi_k)),
#               ('clf', SVC(C=1))]        
     # Create the pipeline: pipeline
     pipeline = Pipeline(steps)
     
#      Create train and test sets

     cv_scores = cross_val_score(pipeline, X, y, cv=16, scoring='accuracy')
     scores_all.update({i:np.mean(cv_scores)})
     scores.append(np.mean(cv_scores))
     # Fit the pipeline to the training set: knn_scaled
     #knn = pipeline.fit(X_train,y_train)
     #prediction = knn.predict(X_test) 
     # Compute and print metrics
     #print('Accuracy with Scaling: {}'.format(knn.score(X_test,y_test)))
#     print('Accuracy with Scaling: {}'.format(cv_scores))
print(np.mean(scores))
end = time.time()
print(end-start)