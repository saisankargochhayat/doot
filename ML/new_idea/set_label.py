import sys
import os
path=os.getcwd()
path=path.strip('new_idea')
sys.path.append(path)
import pandas
from helper import svm,knn,lda,sgd,dtree,misc_helper
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
sets = [['a','m','n','s','t','g','q'],['b','e','c','o','x'],['h','k','u','v'],
['d','r','p'],['f'],['l'],['i'],['w'],['y']]


initialData = pandas.read_csv('datasets/main_train.csv')
for j in range(len(sets)):
    initialData['label'][initialData['label'].isin(sets[j])] = str(j)
initialData.to_csv('datasets/set_train.csv',index=False)
print("Train set dataset formed")

initialData = pandas.read_csv('datasets/main_test.csv')
initialData['set_label'] = -1
for j in range(len(sets)):
    initialData['set_label'][initialData['label'].isin(sets[j])] = str(j)
initialData.to_csv('datasets/set_test.csv',index=False)
print("Test set dataset formed")
