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

class main_model:
    def train(self):
        #Load models
        set_divide_model = pickle.load(open('models/set_divide_model.sav','rb'))
        models = []
        for i in range(len(setlist)):
            if len(setlist[i]) > 1:
                models.append(pickle.load(open('models/set_'+str(i)+'_model.sav','rb')))
        #Load Data
        testFeatures = pandas.read_csv('datasets/test.csv')
        testTarget = pandas.read_csv('datasets/test_target.csv')
        predictions = []
        for index,row in testFeatures.iterrows():
            setIndex = int(set_divide_model.predict([row[set_divide_features]])[0])
            if setIndex >= len(models):
                predictions.append(setlist[setIndex][0])
            else:
                letter = models[setIndex].predict([row[set_features[setIndex]]])[0]
                predictions.append(letter)
        return accuracy_score(testTarget,predictions)
