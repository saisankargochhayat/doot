import pandas
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

allData = pandas.read_csv('../../CSV_Data/dataset_4.csv')
setlist = [['a','m','n','s','t','g','q','o','x'],['b','e','c'],['h','k','u','v'],['d','r','p'],
            ['f','l','i','w','y']]
list_i_want = ['thumb_meta_proxi','thumb_proxi_inter','index_meta_proxi','index_proxi_inter'
,'middle_meta_proxi','middle_proxi_inter','ring_meta_proxi','ring_proxi_inter','pinky_meta_proxi',
'pinky_proxi_inter','thumb_center_distance','index_center_distance','middle_center_distance',
'ring_center_distance','pinky_center_distance','palm_direction','label']
allData = allData[list_i_want]
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
