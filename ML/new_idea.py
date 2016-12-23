import pandas
from helper import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from helper import set_1,set_2,set_3,set_4
from sklearn import preprocessing

initialData = pandas.read_csv('../CSV_Data/dataset_6.csv')
allData = initialData

matrix = allData.drop('label',axis=1).values
matrix = preprocessing.scale(matrix)
allData = pandas.DataFrame(matrix,columns = initialData.columns.drop('label'))
allData['label'] = initialData['label']
# column_names = allData.columns.drop('label')
# print(column_names)
# allData = pandas.DataFrame(preprocessing.scale(allData), columns = column_names)
setlist = [['a','m','n','s','t','g','q','x','o'],['b','e','c'],['h','k','u','v'],['d','r','p']]
sum_accuracy = 0
for i in range(100):
    main_train,main_test = train_test_split(allData,test_size = 0.2,stratify=allData['label'])
    modelList = []
    modelList.append(set_1.get_model(main_train))
    modelList.append(set_2.get_model(main_train))
    modelList.append(set_3.get_model(main_train))
    modelList.append(set_4.get_model(main_train))
    for i in range(4):
        main_train['label'][main_train['label'].isin(setlist[i])] = str(i)
    set_model = svm.get_model(main_train)
    set_predictions = set_model.predict(main_test.drop('label',axis=1).values)
    main_test['actual'] = main_test['label']
    main_test['set_label'] = set_predictions
    main_test['label'] = main_test['set_label']
# 1
    current_data = main_test[main_test['set_label'] == str(0)]
    feature_data = current_data.drop('label',axis=1)
    feature_data = feature_data.drop('set_label',axis=1)
    feature_data = feature_data.drop('actual',axis=1)

    main_test['label'][main_test['set_label']==str(0)] = set_1.get_predictions(modelList[0],feature_data)

# 2
    current_data = main_test[main_test['set_label'] == str(1)]
    feature_data = current_data.drop('label',axis=1)
    feature_data = feature_data.drop('set_label',axis=1)
    feature_data = feature_data.drop('actual',axis=1)
    main_test['label'][main_test['set_label']==str(1)] = set_2.get_predictions(modelList[1],feature_data)

# 3
    current_data = main_test[main_test['set_label'] == str(2)]
    feature_data = current_data.drop('label',axis=1)
    feature_data = feature_data.drop('set_label',axis=1)
    feature_data = feature_data.drop('actual',axis=1)
    main_test['label'][main_test['set_label']==str(2)] = set_3.get_predictions(modelList[2],feature_data)

# 4
    current_data = main_test[main_test['set_label'] == str(3)]
    feature_data = current_data.drop('label',axis=1)
    feature_data = feature_data.drop('set_label',axis=1)
    feature_data = feature_data.drop('actual',axis=1)
    main_test['label'][main_test['set_label']==str(3)] = set_4.get_predictions(modelList[3],feature_data)


    sum_accuracy = sum_accuracy + accuracy_score(main_test['actual'].values,main_test['label'].values)

print(sum_accuracy/100)
