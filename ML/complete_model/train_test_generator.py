import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from setlist import setlist
import sys
import os
path=os.getcwd()
path=path.strip('complete_model')
sys.path.append(path)
from helper import svm,misc_helper

class train_test_generator:
    def generate(self):
        dataFrame = pandas.read_csv('../../CSV_Data/master_dataset.csv')
        feature_columns = list(dataFrame.columns.values)[0:-1]
        features,target = misc_helper.split_feature_target(dataFrame)
        train,test,train_target,test_target = train_test_split(features,target,test_size = 0.2,stratify=target)
        train,test = misc_helper.get_scaled_data(train,test)

        #Initial Datasets
        train = pandas.DataFrame(train,columns=feature_columns)
        train.to_csv('datasets/train.csv',index=False)
        train_target = pandas.DataFrame(train_target,columns=['label'])
        train_target.to_csv('datasets/train_target.csv',index=False)
        test = pandas.DataFrame(test,columns=feature_columns)
        test.to_csv('datasets/test.csv',index=False)
        test_target = pandas.DataFrame(test_target,columns=['label'])
        test_target.to_csv('datasets/test_target.csv',index=False)

        #
        train_target_sets = train_target.copy(deep=True)
        test_target_sets = test_target.copy(deep=True)
        for i in range(len(setlist)):
            train_target_sets['label'][train_target['label'].isin(setlist[i])] = str(i)
        train_target_sets.to_csv('datasets/train_target_sets.csv',index=False)
        for i in range(len(setlist)):
            test_target_sets['label'][test_target['label'].isin(setlist[i])] = str(i)
        test_target_sets.to_csv('datasets/test_target_sets.csv',index=False)

        #Diving into sets
        train_sets_features = [[] for i in range(len(setlist)) if len(setlist[i]) > 1]
        train_sets_targets = [[] for i in range(len(setlist)) if len(setlist[i]) > 1]
        test_sets_features = [[] for i in range(len(setlist)) if len(setlist[i]) > 1]
        test_sets_targets = [[] for i in range(len(setlist)) if len(setlist[i]) > 1]

        for index,row in train.iterrows():
            setIndex = int(train_target_sets['label'][index])
            if setIndex < len(train_sets_features):
                train_sets_features[setIndex].append(row)
                train_sets_targets[setIndex].append(train_target['label'][index])
        for index,row in test.iterrows():
            setIndex = int(test_target_sets['label'][index])
            if setIndex < len(test_sets_features):
                test_sets_features[setIndex].append(row)
                test_sets_targets[setIndex].append(test_target['label'][index])
        for i in range(len(train_sets_features)):
            df = pandas.DataFrame(train_sets_features[i],columns=feature_columns)
            df.to_csv('datasets/train_set_'+str(i),index=False)
            df = pandas.DataFrame(train_sets_targets[i],columns=['label'])
            df.to_csv('datasets/train_target_set_'+str(i),index=False)

            df = pandas.DataFrame(test_sets_features[i],columns=feature_columns)
            df.to_csv('datasets/test_set_'+str(i),index=False)
            df = pandas.DataFrame(test_sets_targets[i],columns=['label'])
            df.to_csv('datasets/test_target_set_'+str(i),index=False)
