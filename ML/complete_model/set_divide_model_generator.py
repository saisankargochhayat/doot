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
from featurelists import set_divide_features, set_features

class set_divide_model_generator:
    def train(self):

        trainFeatures = pandas.read_csv('datasets/train.csv')
        testFeatures = pandas.read_csv('datasets/test.csv')
        trainTargetSets = pandas.read_csv('datasets/train_target_sets.csv')
        testTargetSets = pandas.read_csv('datasets/test_target_sets.csv')

        trainFeatures = trainFeatures[set_divide_features]
        testFeatures = testFeatures[set_divide_features]
        model = svm.SVC(kernel='linear',probability=True)
        model.fit(trainFeatures,np.ravel(trainTargetSets))
        predictions = model.predict(testFeatures)
        result = accuracy_score(testTargetSets,predictions)

        pickle.dump(model,open('models/set_divide_model.sav','wb'))
        return result
