#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 18:29:24 2021
copyright: Quing Zhu, Optical and Ultrasound Imaging Laboratory
Email: zhu.q@wustl.edu

This code is for plotting ROC.
@author: whitaker
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

folder = "whole_images2/AUC_both2"# this is the folder containing all predicted scores
files = sorted(os.listdir(folder))
csvLst = list(filter(lambda f: f.endswith('.csv'), files))
cancer_mean = np.array([])
normal_mean = np.array([])
n_cancer = 0
n_normal = 0
for i in range(len(csvLst)):
    result = pd.read_csv( folder + "/" + csvLst[i])
    print(csvLst[i][:-10],' ')
    mean = np.mean(result.iloc[:,1:])
    print("n_whole = ", len(mean))
    print("score = ", round(np.mean(mean),4),' ')
    print("std = ", round(np.std(mean),4),' ')
    if "Cancer" in csvLst[i]:
        n_cancer += np.sum(result.count())-8
        print("acc = ", round(np.sum(mean < 0.5)/len(result.columns),4),' ')
        cancer_mean = np.append(cancer_mean, mean[:210], axis = 0) # balance the numbers in the 2 classes

    elif "Normal" in csvLst[i]:
        n_normal += np.sum(result.count())-8
        print("acc = ", round(np.sum(mean > 0.5)/len(result.columns),4),' ')
        normal_mean = np.append(normal_mean, mean[:189], axis = 0) # balance the numbers in the 2 classes

x = np.append(cancer_mean, normal_mean)
y = np.append(np.zeros(len(cancer_mean)),np.ones(len(normal_mean)))

fpr, tpr, thresholds = roc_curve(y, x, pos_label=1)
roc_auc = auc(fpr, tpr)

# plot ROC
lw = 2
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()

# find optimal point
ind = np.argmax(tpr-fpr)
print('thresh = ', thresholds[ind])
print('tpr = ', tpr[ind])
print('1-fpr = ', 1-fpr[ind])
