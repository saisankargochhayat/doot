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
from helper import misc_helper
from setlist import setlist
from featurelists import set_divide_features, set_features

class sets_model_generator:
    def train(self):
        accuracies = []
        for i in range(len(setlist)):
            if len(setlist[i]) > 1:
                train = pandas.read_csv('datasets/train_set_'+str(i))
                train_target = pandas.read_csv('datasets/train_target_set_'+str(i))
                test = pandas.read_csv('datasets/test_set_'+str(i))
                test_target = pandas.read_csv('datasets/test_target_set_'+str(i))
                train = train[set_features[i]]
                test = test[set_features[i]]
                model = svm.SVC(kernel='linear',probability=True)
                model.fit(train,np.ravel(train_target))
                predictions = model.predict(test)
                result = accuracy_score(test_target,predictions)
                pickle.dump(model,open('models/set_'+str(i)+'_model.sav','wb'))
                accuracies.append(result)
        return accuracies
