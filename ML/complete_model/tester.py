#Uses master dataset
import pandas
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
import pickle
import sys
import os
path=os.getcwd()

path=path.strip('complete_model')
sys.path.append(path)

from setlist import setlist
from sets_model_generator import sets_model_generator
from train_test_generator import train_test_generator
from set_divide_model_generator import set_divide_model_generator
from featurelists import set_divide_features, set_features

def update_progress(progress,n):
    hashes = '#'*int(progress*40/n)
    blanks = ' '*(40-len(hashes))
    percent = (progress*100)/n
    sys.stdout.write('\r[{0}] {1}%'.format(hashes+blanks, percent))

n = 50
generator = train_test_generator()
model = set_divide_model_generator()
sum_acc = 0
for test in range(n):
    generator.generate()
    acc = model.train()
    sum_acc += acc
    update_progress(test+1,n)
sum_acc = sum_acc/n
print("\n")
print("Accuracy is: "+str(sum_acc))
