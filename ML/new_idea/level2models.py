import sys
import os
path=os.getcwd()
path=path.strip('new_idea')
sys.path.append(path)
import pandas
from helper import svm,knn,lda,sgd,dtree,misc_helper
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
sets = [['a','m','n','s','t','g','q'],['b','e','c','o','x'],['h','k','u','v'],
['d','r','p'],['f'],['l'],['i'],['w'],['y']]
main_train = pandas.read_csv('datasets/main_train.csv')
models = []
scalars = []
set_test = pandas.read_csv('datasets/set_test.csv')
for i in range(len(sets)):
    curr_set = sets[i]
    if len(curr_set) > 1:
        curr_data = main_train[main_train['label'].isin(curr_set)]
        model,scalar = svm.get_model(curr_data)
        models.append(model)
        scalars.append(scalar)

for i in range(len(models)):
    model = models[i]
    scalar = scalars[i]
    model_name = 'models/set_'+str(i)+'.sav'
    scalar_name = 'scalars/set_'+str(i)+'.sav'
    pickle.dump(model,open(model_name,'wb+'))
    pickle.dump(scalar,open(scalar_name,'wb+'))
    actual = pandas.read_csv('datasets/set_'+str(i)+'_actual.csv')
    level1 = pandas.read_csv('datasets/set_'+str(i)+'_test.csv')
    actual = actual['label'].values
    level1 = level1.drop('label',axis=1).values
    level1 = scalar.transform(level1)
    predictions = model.predict(level1)
    print("Set "+str(i))
    print(accuracy_score(actual,predictions))

print("Models and scalars trained")
