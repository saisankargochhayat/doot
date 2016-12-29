import pandas
import numpy as np
from sklearn import preprocessing

def split_feature_target(dataFrame):
    target = dataFrame['label'].values
    features = dataFrame.drop('label',axis=1).values
    return features,target

def get_scaled_data(train,test):
    scaler = preprocessing.StandardScaler()
    scaler.fit_transform(train)
    scaler.transform(test)
    return train,test

def get_scaler(data):
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    return data
