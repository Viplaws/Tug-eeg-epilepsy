# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 20:25:39 2019

@author: subasi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import scipy.io as sio
# descriptive statistics
import scipy as sp
import pywt
import scipy.stats as stats
#waveletname = 'sym5'
waveletname='db1'
mat_contents = sio.loadmat('final.mat')
sorted(mat_contents.keys())

EpilepticZone_Interictal=mat_contents['Epilesy']
Epileptic_Ictal=mat_contents['No_epilepsy']
temp_list = []

######################################################
#coeff = pywt.wavedec(Normal_Eyes_Open[:,1], waveletname, level)
###############################
"""Extract The Coeeficients"""
Length = 1024;    # Length of signal
Nofsignal=400; #Number of Signal
numrows = len(No_epilepsy)   
numrows =48     #Number of features extracted from DWT decomposition 
numcols = len(No_epilepsy[0]) 
#rms = np.sqrt(np.mean(y**2))

Extracted_Features=np.ndarray(shape=(2*numcols,numrows), dtype=float, order='F')

"""NORMAL EEG SIGNAL"""  

for i in range(numcols):
    coeff = pywt.wavedec(No_epilepsy[:,i], waveletname, level=6)
    cA6,cD6,cD5,cD4, cD3, cD2, cD1=coeff
    
    Extracted_Features[i,0]=sp.mean(abs(cD1[:]))
    Extracted_Features[i,1]=sp.mean(abs(cD2[:]))
    Extracted_Features[i,2]=sp.mean(abs(cD3[:]))
    Extracted_Features[i,3]=sp.mean(abs(cD4[:]))
    Extracted_Features[i,4]=sp.mean(abs(cD5[:]))
    Extracted_Features[i,5]=sp.mean(abs(cD6[:]))
    Extracted_Features[i,6]=sp.mean(abs(cA6[:]))
   
    Extracted_Features[i,7]=sp.std(cD1[:]);
    Extracted_Features[i,8]=sp.std(cD2[:]);
    Extracted_Features[i,9]=sp.std(cD3[:]);
    Extracted_Features[i,10]=sp.std(cD4[:]);
    Extracted_Features[i,11]=sp.std(cD5[:]);
    Extracted_Features[i,12]=sp.std(cD6[:]);
    Extracted_Features[i,13]=sp.std(cA6[:]);
    
    Extracted_Features[i,14]=stats.skew(cD1[:]);
    Extracted_Features[i,15]=stats.skew(cD2[:]);
    Extracted_Features[i,16]=stats.skew(cD3[:]);
    Extracted_Features[i,17]=stats.skew(cD4[:]);
    Extracted_Features[i,18]=stats.skew(cD5[:]);
    Extracted_Features[i,19]=stats.skew(cD6[:]);
    Extracted_Features[i,20]=stats.skew(cA6[:]);
    
    Extracted_Features[i,21]=stats.kurtosis(cD1[:]);
    Extracted_Features[i,22]=stats.kurtosis(cD2[:]);
    Extracted_Features[i,23]=stats.kurtosis(cD3[:]);
    Extracted_Features[i,24]=stats.kurtosis(cD4[:]);
    Extracted_Features[i,25]=stats.kurtosis(cD5[:]);
    Extracted_Features[i,26]=stats.kurtosis(cD6[:]);
    Extracted_Features[i,27]=stats.kurtosis(cA6[:]);
    
    Extracted_Features[i,28]=sp.median(cD1[:]);
    Extracted_Features[i,29]=sp.median(cD2[:]);
    Extracted_Features[i,30]=sp.median(cD3[:]);
    Extracted_Features[i,31]=sp.median(cD4[:]);
    Extracted_Features[i,32]=sp.median(cD5[:]);
    Extracted_Features[i,33]=sp.median(cD6[:]);
    Extracted_Features[i,34]=sp.median(cA6[:]);
    
    Extracted_Features[i,35]=np.sqrt(np.mean(cD1[:]**2));#RMS Value
    Extracted_Features[i,36]=np.sqrt(np.mean(cD2[:]**2));
    Extracted_Features[i,37]=np.sqrt(np.mean(cD3[:]**2));
    Extracted_Features[i,38]=np.sqrt(np.mean(cD4[:]**2));
    Extracted_Features[i,39]=np.sqrt(np.mean(cD5[:]**2));
    Extracted_Features[i,40]=np.sqrt(np.mean(cD6[:]**2));
    Extracted_Features[i,41]=np.sqrt(np.mean(cA6[:]**2));
    
    Extracted_Features[i,42]=sp.mean(abs(cD1[:]))/sp.mean(abs(cD2[:]))
    Extracted_Features[i,43]=sp.mean(abs(cD2[:]))/sp.mean(abs(cD3[:]))
    Extracted_Features[i,44]=sp.mean(abs(cD3[:]))/sp.mean(abs(cD4[:]))
    Extracted_Features[i,45]=sp.mean(abs(cD4[:]))/sp.mean(abs(cD5[:]))
    Extracted_Features[i,46]=sp.mean(abs(cD5[:]))/sp.mean(abs(cD6[:]))
    Extracted_Features[i,47]=sp.mean(abs(cD6[:]))/sp.mean(abs(cA6[:]))
    
    temp_list.append("No_epilepsy")

for i in range(numcols, 2*numcols):
    coeff = pywt.wavedec(Epilesy[:,i-numcols], waveletname, level=6)
    cA6,cD6,cD5,cD4, cD3, cD2, cD1=coeff

    Extracted_Features[i,0]=sp.mean(abs(cD1[:]))
    Extracted_Features[i,1]=sp.mean(abs(cD2[:]))
    Extracted_Features[i,2]=sp.mean(abs(cD3[:]))
    Extracted_Features[i,3]=sp.mean(abs(cD4[:]))
    Extracted_Features[i,4]=sp.mean(abs(cD5[:]))
    Extracted_Features[i,5]=sp.mean(abs(cD6[:]))
    Extracted_Features[i,6]=sp.mean(abs(cA6[:]))
   
    Extracted_Features[i,7]=sp.std(cD1[:]);
    Extracted_Features[i,8]=sp.std(cD2[:]);
    Extracted_Features[i,9]=sp.std(cD3[:]);
    Extracted_Features[i,10]=sp.std(cD4[:]);
    Extracted_Features[i,11]=sp.std(cD5[:]);
    Extracted_Features[i,12]=sp.std(cD6[:]);
    Extracted_Features[i,13]=sp.std(cA6[:]);
    
    Extracted_Features[i,14]=stats.skew(cD1[:]);
    Extracted_Features[i,15]=stats.skew(cD2[:]);
    Extracted_Features[i,16]=stats.skew(cD3[:]);
    Extracted_Features[i,17]=stats.skew(cD4[:]);
    Extracted_Features[i,18]=stats.skew(cD5[:]);
    Extracted_Features[i,19]=stats.skew(cD6[:]);
    Extracted_Features[i,20]=stats.skew(cA6[:]);
    
    Extracted_Features[i,21]=stats.kurtosis(cD1[:]);
    Extracted_Features[i,22]=stats.kurtosis(cD2[:]);
    Extracted_Features[i,23]=stats.kurtosis(cD3[:]);
    Extracted_Features[i,24]=stats.kurtosis(cD4[:]);
    Extracted_Features[i,25]=stats.kurtosis(cD5[:]);
    Extracted_Features[i,26]=stats.kurtosis(cD6[:]);
    Extracted_Features[i,27]=stats.kurtosis(cA6[:]);
    
    Extracted_Features[i,28]=sp.median(cD1[:]);
    Extracted_Features[i,29]=sp.median(cD2[:]);
    Extracted_Features[i,30]=sp.median(cD3[:]);
    Extracted_Features[i,31]=sp.median(cD4[:]);
    Extracted_Features[i,32]=sp.median(cD5[:]);
    Extracted_Features[i,33]=sp.median(cD6[:]);
    Extracted_Features[i,34]=sp.median(cA6[:]);
    
    Extracted_Features[i,35]=np.sqrt(np.mean(cD1[:]**2));#RMS Value
    Extracted_Features[i,36]=np.sqrt(np.mean(cD2[:]**2));
    Extracted_Features[i,37]=np.sqrt(np.mean(cD3[:]**2));
    Extracted_Features[i,38]=np.sqrt(np.mean(cD4[:]**2));
    Extracted_Features[i,39]=np.sqrt(np.mean(cD5[:]**2));
    Extracted_Features[i,40]=np.sqrt(np.mean(cD6[:]**2));
    Extracted_Features[i,41]=np.sqrt(np.mean(cA6[:]**2));
    
    Extracted_Features[i,42]=sp.mean(abs(cD1[:]))/sp.mean(abs(cD2[:]))
    Extracted_Features[i,43]=sp.mean(abs(cD2[:]))/sp.mean(abs(cD3[:]))
    Extracted_Features[i,44]=sp.mean(abs(cD3[:]))/sp.mean(abs(cD4[:]))
    Extracted_Features[i,45]=sp.mean(abs(cD4[:]))/sp.mean(abs(cD5[:]))
    Extracted_Features[i,46]=sp.mean(abs(cD5[:]))/sp.mean(abs(cD6[:]))
    Extracted_Features[i,47]=sp.mean(abs(cD6[:]))/sp.mean(abs(cA6[:]))
    
    temp_list.append("Epilepsy")
    
"""CLASSIFICATION"""

X = Extracted_Features
y = temp_list

from sklearn.model_selection import cross_val_score
from sklearn import svm
"""
The parameters and kernels of SVM classifierr can be changed as follows
C = 10.0  # SVM regularization parameter
svm.SVC(kernel='linear', C=C)
svm.LinearSVC(C=C, max_iter=10000)
svm.SVC(kernel='rbf', gamma=0.7, C=C)
svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
"""
C = 10.0  # SVM regularization parameter
CV=5    #Number of Folds in Cross Validation 
clf =svm.SVC(kernel='linear', C=C)
Acc_scores = cross_val_score(clf, X, y, cv=CV)
print("Accuracy: %0.3f (+/- %0.3f)" % (Acc_scores.mean(), Acc_scores.std() * 2))
f1_scores = cross_val_score(clf, X, y, cv=CV,scoring='f1_macro')
print("F1 score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2))
Precision_scores = cross_val_score(clf, X, y, cv=CV,scoring='precision_macro')
print("Precision score: %0.3f (+/- %0.3f)" % (Precision_scores.mean(), Precision_scores.std() * 2))
Recall_scores = cross_val_score(clf, X, y, cv=CV,scoring='recall_macro')
print("Recall score: %0.3f (+/- %0.3f)" % (Recall_scores.mean(), Recall_scores.std() * 2))

from sklearn.metrics import cohen_kappa_score, make_scorer
kappa_scorer = make_scorer(cohen_kappa_score)
#metrics.cohen_kappa_score(y1, y2, labels=None, weights=None, sample_weight=None)
Kappa_scores = cross_val_score(clf, X, y, cv=CV,scoring=kappa_scorer)
print("Kappa score: %0.3f (+/- %0.3f)" % (Kappa_scores.mean(), Kappa_scores.std() * 2))

