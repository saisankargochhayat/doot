import pandas
import numpy as np
from sklearn import preprocessing
import string
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
    data = scaler.fit_transform(data)
    return data,scaler

def write_matrix(raw_conf,filename):
    labels = list(string.ascii_lowercase)
    labels.remove('j')
    labels.remove('z')
    labels = np.insert(labels,0,'-')
    text_file = open(filename,'w')
    labels = np.array(labels)
    text_file.write(",".join(labels))
    text_file.write('\n')
    for i in range(len(raw_conf)):
        text_file.write(labels[i+1]+',')
        text_file.write(','.join(map(str,raw_conf[i])))
        text_file.write('\n')
