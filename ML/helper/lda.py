import pandas
from sklearn.lda import LDA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def find_accuracy(dataFrame):
    uniqueLabels = dataFrame['label'].unique()
    target = dataFrame['label'].values
    dataFrame = dataFrame.drop('label',axis=1).values
    train,test,train_target,test_target = train_test_split(dataFrame,target,test_size = 0.2,stratify=target)

    model = LDA()
    model.fit(train,train_target)
    result = {}
    result['train'] = model.score(train,train_target)
    predictions = model.predict(test)
    result['test'] = accuracy_score(test_target,predictions)
    return result
