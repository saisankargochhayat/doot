import pandas

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from helper import svm

initialData = pandas.read_csv('../CSV_Data/dataset_4.csv')
allData = initialData
setlist = [['a','m','n','s','t','g','q','o','x'],['b','e','c'],['h','k','u','v'],['d','r','p'],
            ['f','l','i','w','y']]
sum_accuracy = 0
for i in range(20):
    main_train,main_test = train_test_split(allData,test_size = 0.2,stratify=allData['label'])
    modelList = []
    for current_set in setlist:
        curr_train = main_train[main_train['label'].isin(current_set)]
        modelList.append(svm.get_model(curr_train))

    for i in range(len(setlist)):
        main_train['label'][main_train['label'].isin(setlist[i])] = str(i)

    set_model = svm.get_model(main_train)
    set_predictions = set_model.predict(main_test.drop('label',axis=1).values)
    main_test['actual'] = main_test['label']
    main_test['set_label'] = set_predictions
    for i in range(len(setlist)):
        current_data = main_test[main_test['set_label'] == str(i)]
        feature_data = current_data.drop('label',axis=1)
        feature_data = feature_data.drop('set_label',axis=1)
        feature_data = feature_data.drop('actual',axis=1)
        main_test['label'][main_test['set_label']==str(i)] = modelList[i].predict(feature_data.values)
    sum_accuracy = sum_accuracy + accuracy_score(main_test['actual'].values,main_test['label'].values)

print(sum_accuracy/20)
