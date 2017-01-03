import pandas
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
# import misc_helper
from . import misc_helper

def find_accuracy(dataFrame, n_neighbors=5, weights='uniform'):
    features,target = misc_helper.split_feature_target(dataFrame)
    train,test,train_target,test_target = train_test_split(features,target,test_size = 0.2,stratify=target)
    train,test = misc_helper.get_scaled_data(train,test)
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(train,train_target)

    predictions = model.predict(test)
    result = accuracy_score(test_target,predictions)
    confusion = confusion_matrix(test_target,predictions)
    return result,confusion

def get_model(dataFrame):
    features,target = misc_helper.split_feature_target(dataFrame)
    features,scaler = misc_helper.get_scaler(features)
    # dataFrame = preprocessing.scale(dataFrame)
    model = KNeighborsClassifier()
    model.fit(features,target)
    return model,scaler

def get_set_model(dataFrame,my_set,feature_list):
    dataFrame = dataFrame[dataFrame['label'].isin(my_set)]
    dataFrame = dataFrame[feature_list]
    features,target = misc_helper.split_feature_target(dataFrame)
    features,scaler = misc_helper.get_scaler(features)
    model = KNeighborsClassifier()
    model.fit(features,target)
    return model,scaler

def get_set_accuracy(dataFrame,my_set,feature_list):
    dataFrame = dataFrame[dataFrame['label'].isin(my_set)]
    dataFrame = dataFrame[feature_list]
    features,target = misc_helper.split_feature_target(dataFrame)
    train,test,train_target,test_target = train_test_split(features,target,test_size = 0.2,stratify=target)
    train,test = misc_helper.get_scaled_data(train,test)
    model = KNeighborsClassifier()
    model.fit(train,train_target)

    predictions = model.predict(test)
    result = accuracy_score(test_target,predictions)
    confusion = confusion_matrix(test_target,predictions)
    return result,confusion
