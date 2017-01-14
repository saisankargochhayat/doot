import pandas
from helper import svm,knn,lda,sgd,dtree,misc_helper
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
initialData = pandas.read_csv('../CSV_Data/dataset_8.csv')
allData = initialData
all_features = allData.columns.values
sets = [['a','m','n','s','t','q','o','g','x'],['b','e','c'],['h','k','u','v'],['d','r','p']]
coord = ['x','y','z']
finger_map = ['thumb','index','middle','ring','pinky']
feature_list_1 = all_features.tolist()
for axis in coord:
    feature_list_1.remove('hand_direction_'+ axis)
feature_list_2 = ['label']
for axis in coord:
    feature_list_2.append('hand_direction_'+ axis)
for finger in finger_map:
    for axis in coord:
        feature_list_2.append(finger + '_direction_' + axis)
feature_lists = [feature_list_1,feature_list_2,feature_list_2,feature_list_2]

sum_accuracy = 0
for i in range(100):
    main_train,main_test = train_test_split(allData,test_size = 0.2,stratify=allData['label'])
    modelList = []
    scalerList = []

    model,scaler = svm.get_set_model(main_train,sets[0],feature_lists[0])
    modelList.append(model)
    scalerList.append(scaler)

    model,scaler = svm.get_set_model(main_train,sets[1],feature_lists[1])
    modelList.append(model)
    scalerList.append(scaler)

    model,scaler = svm.get_set_model(main_train,sets[2],feature_lists[2])
    modelList.append(model)
    scalerList.append(scaler)

    model,scaler = svm.get_set_model(main_train,sets[3],feature_lists[3])
    modelList.append(model)
    scalerList.append(scaler)

    for i in range(0,4):
        main_train['label'][main_train['label'].isin(sets[i])] = str(i)
    # features,target = misc_helper.split_feature_target(main_train)
    # features,set_scaler = misc_helper.get_scaler(features)
    # # dataFrame = preprocessing.scale(dataFrame)
    # set_model = main_svm.SVC(kernel='linear')
    # set_model.fit(features,target)
    set_model , set_scaler = svm.get_set_model(main_train,main_train['label'].unique(),feature_list_1)
    main_test_features = main_test[feature_list_1].drop('label',axis=1).values
    set_predictions = set_model.predict(set_scaler.transform(main_test_features))
    main_test['actual'] = main_test['label']
    main_test['set_label'] = set_predictions
    main_test['label'] = main_test['set_label']

    for i in range(0,4):
        current_data = main_test[main_test['set_label'] == str(i)]
        # print(current_data)
        current_data = current_data[feature_lists[i]]
        feature_data = current_data.drop('label',axis=1).values
        # print(feature_data)
        feature_data = scalerList[i].transform(feature_data)
        main_test['label'][main_test['set_label']==str(i)] = modelList[i].predict(feature_data)
    sum_accuracy = sum_accuracy + accuracy_score(main_test['actual'].values,main_test['label'].values)

print(sum_accuracy/100)
