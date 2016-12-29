import pandas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from . import misc_helper

def find_accuracy(dataFrame):
    features,target = misc_helper.split_feature_target(dataFrame)
    train,test,train_target,test_target = train_test_split(features,target,test_size = 0.2,stratify=target)
    train,test = misc_helper.get_scaled_data(train,test)
    model = LinearDiscriminantAnalysis()
    model.fit(train,train_target)

    predictions = model.predict(test)
    result = accuracy_score(test_target,predictions)
    confusion = confusion_matrix(test_target,predictions)
    return result,confusion

def get_model(dataFrame):
    features,target = misc_helper.split_feature_target(dataFrame)
    scaler = misc_helper.get_scaler(dataFrame)
    # dataFrame = preprocessing.scale(dataFrame)
    model = LinearDiscriminantAnalysis()
    model.fit(features,target)
    return model,scaler
