# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:58:53 2018

@author: Dan
"""
#folder='C:/Users/Dan/Documents/GitHub/PythonClass'

import os
#os.chdir(folder)
#os.getcwd()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import auxfuns as af
import timeit
import pandas as pd

#%% Importing data
# loading data
data=af.loadDataDN('S09_fft.mat')

#%% testing Standard Deviation
# across electrodes
els_std=np.mean(data,axis=2)
els_std=np.std(els_std,axis=2)
fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.pcolor(els_std,vmin=0, vmax=0.2)
plt.show()
els_std=np.mean(els_std,axis=1)
bad_els=pd.DataFrame(els_std)
bad_els=bad_els.sort_values([0]).index.values

#across words
std_wrds=np.mean(data,axis=3)
std_wrds=np.std(std_wrds,axis=2)
fig = plt.gcf()
fig.set_size_inches(7, 7)
plt.pcolor(std_wrds,vmin=0, vmax=0.2)
plt.show()
std_wrds=np.mean(std_wrds,axis=1)

fig=plt.gcf()
fig.set_size_inches(7,7)
plt.scatter(els_std,std_wrds)
plt.xlabel('Blocks std')
plt.ylabel('Words std')
for i in range(64):
    plt.annotate(i, (els_std[i],std_wrds[i]))

#%% Defining parameters
els=np.array([24,23,22,25,26,27,59,60,61,62,63,64])-1
domain=np.array(np.arange(75,375))
domain=np.arange(0,40)
els=np.arange(0,64)
els=np.delete(np.arange(0,64),29,axis=0)
#%% Preparing for classification
dataIn,labels=af.foldMatDN(data,els=els,domain=domain)
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
#%% saving matrix to matlab
import numpy, scipy.io
scipy.io.savemat('C:/Users/Dan/Documents/GitHub/PythonClass/arrdata.mat', mdict={'arr': data})
