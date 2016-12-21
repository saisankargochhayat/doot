import pandas
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def get_model(allData):
    my_set = ['b','e','c']

    my_list = ['index_meta_proxi','index_proxi_inter',
    'middle_meta_proxi','middle_proxi_inter','thumb_index','index_middle','index_direction_x',
    'index_direction_y','index_direction_z','middle_direction_x','middle_direction_y','middle_direction_z'
    ,'label']
    # allData = allData[my_list]

    dataFrame = allData[allData['label'].isin(my_set)]


    target = dataFrame['label'].values
    dataFrame = dataFrame.drop('label',axis=1).values
    model = svm.SVC(kernel='linear')
    model.fit(dataFrame,target)
    return model

def get_predictions(model,dataFrame):
    my_list = ['index_meta_proxi','index_proxi_inter',
    'middle_meta_proxi','middle_proxi_inter','thumb_index','index_middle','index_direction_x',
    'index_direction_y','index_direction_z','middle_direction_x','middle_direction_y','middle_direction_z']
    # dataFrame = dataFrame[my_list]
    return model.predict(dataFrame.values)

def get_accuracy(allData):
    my_set = ['b','e','c']

    my_list = ['index_meta_proxi','index_proxi_inter',
    'middle_meta_proxi','middle_proxi_inter','thumb_index','index_middle','index_direction_x',
    'index_direction_y','index_direction_z','middle_direction_x','middle_direction_y','middle_direction_z'
    ,'label']
    # allData = allData[my_list]

    allData = allData[allData['label'].isin(my_set)]


    sum_acc = 0
    sum_confusion = [[0 for x in range(len(my_set))] for y in range(len(my_set))]
    for i in range(100):
        trainData , testData = train_test_split(allData,test_size=0.2,stratify=allData['label'])
        model = svm.get_model(trainData)
        predictions = model.predict(testData.drop('label',axis=1).values)
        sum_acc = sum_acc + accuracy_score(testData['label'].values,predictions)
        sum_confusion = np.add(sum_confusion,confusion_matrix(testData['label'].values,predictions))
    print(my_set)
    print(sum_acc/100)
    print(sum_confusion)
