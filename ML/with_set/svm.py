import pandas
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

allData = pandas.read_csv('../../CSV_Data/dataset_3.csv')
setlist = [['a','s','m','n','t','i'],['c','d','o','q','e','x'],['g','h','p'],['u','v','w','k'],
['b','f','l','y','r']]
for current_set in setlist:
    print(current_set)
    dataFrame = allData[allData['label'].isin(current_set)]
    uniqueLabels = dataFrame['label'].unique()
    target = dataFrame['label'].values
    dataFrame = dataFrame.drop('label',axis=1).values
    train,test,train_target,test_target = train_test_split(dataFrame,target,test_size = 0.2,stratify=target)

    model = svm.SVC(kernel='linear')
    model.fit(train,train_target)
    print("Training Accuracy : "+ str(model.score(train,train_target)))
    predictions = model.predict(test)
    print("Test Accuracy : " + str(accuracy_score(test_target,predictions)))
    print(" Confusion Matrix : ")
    print(confusion_matrix(test_target, predictions,labels =uniqueLabels))
