import sys
import os
path=os.getcwd()
path=path.strip('new_idea')
sys.path.append(path)
import pandas
from helper import svm,knn,lda,sgd,dtree,misc_helper
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
initialData = pandas.read_csv('datasets/main.csv')
print(initialData)
allData = initialData
all_features = allData.columns.values
sets = [['a','m','n','s','t','g','q'],['b','e','c','o','x'],['h','k','u','v'],
['d','r','p'],['f'],['l'],['i'],['w'],['y']]
finger_map = ['thumb','index','middle','ring','pinky']

feature_lists = [all_features,all_features,all_features,all_features,all_features]

sum_accuracy = 0
sum_set_divide = 0
sum_set= [0.0 for x in range(len(sets))]
for i in range(100):
    main_train,main_test = train_test_split(allData,test_size = 0.2,stratify=allData['label'])
    modelList = []
    scalerList = []

    for set_index in range(len(sets)):
        model,scaler = svm.get_set_model(main_train,sets[set_index],feature_lists[set_index])
        modelList.append(model)
        scalerList.append(scaler)

    for i in range(len(sets)):
        main_train['label'][main_train['label'].isin(sets[i])] = str(i)
    # features,target = misc_helper.split_feature_target(main_train)
    # features,set_scaler = misc_helper.get_scaler(features)
    # # dataFrame = preprocessing.scale(dataFrame)
    # set_model = main_svm.SVC(kernel='linear')
    # set_model.fit(features,target)
    set_model , set_scaler = svm.get_set_model(main_train,main_train['label'].unique(),all_features)
    main_test_features = main_test[all_features].drop('label',axis=1).values
    set_predictions = set_model.predict(set_scaler.transform(main_test_features))
    main_test['actual'] = main_test['label']
    main_test['set_label'] = set_predictions
    main_test['label'] = main_test['set_label']
    main_test['actual_set'] = main_test['actual']
    for i in range(len(sets)):
        main_test['actual_set'][main_test['actual_set'].isin(sets[i])] = str(i)
    for i in range(len(sets)):
        current_data = main_test[main_test['set_label'] == str(i)]
        # print(current_data)
        current_data = current_data[feature_lists[i]]
        feature_data = current_data.drop('label',axis=1).values
        # print(feature_data)
        feature_data = scalerList[i].transform(feature_data)
        main_test['label'][main_test['set_label']==str(i)] = modelList[i].predict(feature_data)

    sum_accuracy = sum_accuracy + accuracy_score(main_test['actual'].values,main_test['label'].values)
    for i in range(len(sum_set)):
        actuals = main_test['actual'][main_test['actual'].isin(sets[i])].values
        indexes = main_test['actual'][main_test['actual'].isin(sets[i])].index.tolist()
        predicts = main_test['label'][indexes].values
        sum_set[i] = sum_set[i] + accuracy_score(actuals,predicts)
        main_test['label'][main_test['actual'].isin(sets[i])]
    sum_set_divide = sum_set_divide + accuracy_score(main_test['set_label'].values,main_test['actual_set'])
print("Overall accuracy")
print(sum_accuracy/100)
sum_set = np.divide(sum_set,100)
print("Accuracy of different sets")
print(sum_set)
print("Set divide accuracy")
print(sum_set_divide/100)
