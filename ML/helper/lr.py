import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import linear_mode
from sklearn.metrics import classification_report
# import misc_helper
from . import misc_helper

def find_accuracy(dataFrame):
    features,target = misc_helper.split_feature_target(dataFrame)
    train,test,train_target,test_target = train_test_split(features,target,test_size = 0.2,stratify=target)
    train,test = misc_helper.get_scaled_data(train,test)
    model = linear_model.LogisticRegression(C=1e5)
    model.fit(train,train_target)

    predictions = model.predict(test)
    result = accuracy_score(test_target,predictions)
    confusion = confusion_matrix(test_target,predictions)
    classification=classification_report(test_target,predictions)
    return result,confusion,classification


def get_model(dataFrame):
    features,target = misc_helper.split_feature_target(dataFrame)
    features,scaler = misc_helper.get_scaler(features)
    # dataFrame = preprocessing.scale(dataFrame)
    model = linear_model.LogisticRegression(C=1e5)
    model.fit(features,target)
    return model,scaler

def get_set_model(dataFrame,my_set,feature_list):
    dataFrame = dataFrame[dataFrame['label'].isin(my_set)]
    dataFrame = dataFrame[feature_list]
    features,target = misc_helper.split_feature_target(dataFrame)
    features,scaler = misc_helper.get_scaler(features)
    model = linear_model.LogisticRegression(C=1e5)
    model.fit(features,target)
    return model,scaler

def get_set_accuracy(dataFrame,my_set,feature_list):
    dataFrame = dataFrame[dataFrame['label'].isin(my_set)]
    dataFrame = dataFrame[feature_list]
    features,target = misc_helper.split_feature_target(dataFrame)
    train,test,train_target,test_target = train_test_split(features,target,test_size = 0.2,stratify=target)
    train,test = misc_helper.get_scaled_data(train,test)
    model = linear_model.LogisticRegression(C=1e5)
    model.fit(train,train_target)

    predictions = model.predict(test)
    result = accuracy_score(test_target,predictions)
    confusion = confusion_matrix(test_target,predictions)
    return result,confusion

def get_precision(dataFrame, n_neighbors=5, weights='uniform'):
    features,target = misc_helper.split_feature_target(dataFrame)
    sum_confusion = np.array([[0 for x in range(24)] for y in range(24)])
    for i in range(50):
        train,test,train_target,test_target = train_test_split(features,target,test_size = 0.2,stratify=target)
        train,test = misc_helper.get_scaled_data(train,test)
        model = linear_model.LogisticRegression(C=1e5)
        # print(model)
        model.fit(train,train_target)

        predictions = model.predict(test)
        confusion = confusion_matrix(test_target,predictions)
        sum_confusion = np.add(sum_confusion,confusion)
    sum_confusion = sum_confusion.transpose()
    precision = [[0.0 for x in range(24)] for y in range(24)]
    for i in range(len(precision)):
        precision[i] = np.divide(sum_confusion[i],np.sum(sum_confusion[i]))
    return precision
