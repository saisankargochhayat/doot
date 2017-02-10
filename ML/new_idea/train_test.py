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

initialData = pandas.read_csv('datasets/main.csv')

train,test = train_test_split(initialData,stratify=initialData['label'],test_size=0.2)
train.to_csv('datasets/main_train.csv',index=False)
test.to_csv('datasets/main_test.csv',index=False)
print("Splitted to Train and Test")
