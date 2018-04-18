# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:00:57 2018

@author: Dan
"""
#%% Importing packages
# Import package

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import auxfuns as af
import timeit

# loading data
data=af.loadDataDN('S09_fft.mat')
# turning into 2D matrix
els=np.array([24,23,22,25,26,27,59,60,61,62,63,64])-1
domain=np.array(np.arange(75,375))
dataIn,labels=af.foldMatDN(data,els=els,domain=domain)

#%% scaling data
from sklearn.preprocessing import StandardScaler, scale, maxabs_scale, minmax_scale,Normalizer, MaxAbsScaler
#scl = MaxAbsScaler() #Scale each feature by its maximum absolute value
#scl = Normalizer() #Scale input vectors individually to unit norm
#scl = StandardScaler() #Center to the mean and component wise scale to unit variance.
#data = maxabs_scale(dataIn) #Scale each feature to the [-1, 1] range
#data = scale(dataIn) #Center to the mean and component wise scale to unit variance.
#data=minmax_scale(dataIn,feature_range=(0, 1), axis=0)# scaling between 0 and 1


#%% pca
from sklearn.decomposition import PCA
pca = PCA(n_components=200) 


#%% ica
from sklearn.decomposition import FastICA
ica = FastICA(n_components=480)


#%% NMF
from sklearn.decomposition import NMF
model = NMF(n_components=400)


#%% Preprocessing
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scl)
data = pipeline.fit_transform(dataIn)

#%% Classifier SVM
# Import the necessary modules

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
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
#%% GridSearch for functions
from sklearn.preprocessing import StandardScaler, scale, maxabs_scale
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,  cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, mean_squared_error
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.svm import SVC
import itertools

def svc_param_selection(X, y, nfolds):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

scores_all=dict()
scores=list()
for i in itertools.combinations(np.arange(1,21), 2):
     X=data[np.logical_or(labels==i[0],labels==i[1]),:]
     y=labels[np.logical_or(labels==i[0],labels==i[1])]
     X=scale(X)
     scores.append(svc_param_selection(X, y, 8))
     print(i)
res=list()
for i in scores:
     for j in i.keys():
          print(i[j])
          print(res)
          res.append(i[j])

     
#%% Classifier Logistic Regression
# Import the necessary modules
from sklearn.preprocessing import StandardScaler, scale, maxabs_scale
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,  cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, mean_squared_error
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_selection import chi2, SelectKBest, SelectFromModel, f_classif
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression, ElasticNet  
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB #good
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import itertools

chi_k = 480
combs=list(itertools.combinations(np.arange(1,81), 2))
scores_all=dict()
scores=list()
for i in itertools.combinations(np.arange(1,81), 2):
     print(i)
     X=data[np.logical_or(labels==i[0],labels==i[1]),:]
     y=labels[np.logical_or(labels==i[0],labels==i[1])]
#     y=(y > np.min(y)).astype(int)
#      b
#     print(X)
     # Setup the pipeline steps: steps
     steps = [('feature_selection', SelectFromModel(LinearSVC())),
               ('classification', LogisticRegression())]
#     steps = [('dim_red', SelectKBest(score_func=chi2, k=460)),
#               ('clf', LogisticRegression())]        
     # Create the pipeline: pipeline
     pipeline = Pipeline(steps)
     
     # Create train and test sets

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

#%% Classifier 
# Import the necessary modules
from sklearn.preprocessing import StandardScaler, scale, maxabs_scale
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,  cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, mean_squared_error
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_selection import chi2, SelectKBest, SelectFromModel
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression, ElasticNet  
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB #good
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import itertools

chi_k = 'all'
combs=list(itertools.combinations(np.arange(1,81), 2))
scores_all=dict()
scores=list()
for i in itertools.combinations(np.arange(1,81), 2):
     print(i)
     X=data[np.logical_or(labels==i[0],labels==i[1]),:]
     y=labels[np.logical_or(labels==i[0],labels==i[1])]
     y=(y > np.min(y)).astype(int)
     X=scale(X)
     # Setup the pipeline steps: steps
     steps = [('classification', LinearDiscriminantAnalysis())]
#     steps = [('dim_red', SelectKBest(chi2, chi_k)),
#               ('clf', SVC(C=1))]        
     # Create the pipeline: pipeline
     pipeline = Pipeline(steps)
     
     # Create train and test sets

     cv_scores = cross_val_score(pipeline, X, y, cv=8, scoring='accuracy')
     scores_all.update({i:np.mean(cv_scores)})
     scores.append(np.mean(cv_scores))
     # Fit the pipeline to the training set: knn_scaled
     #knn = pipeline.fit(X_train,y_train)
     #prediction = knn.predict(X_test) 
     # Compute and print metrics
     #print('Accuracy with Scaling: {}'.format(knn.score(X_test,y_test)))
#     print('Accuracy with Scaling: {}'.format(cv_scores))
print(np.mean(scores))
