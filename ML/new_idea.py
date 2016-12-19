import pandas
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

initialData = pandas.read_csv('../CSV_Data/dataset_3.csv')
allData = initialData
setlist = [['b','f','l','y','r'],['a','s','m','n','t','i'],['c','d','o','q','e','x'],['h','p'],
['u','v','w','k']]
modelList = []
dataFrame4 = allData[allData['label'].isin(['u','v','w','k'])]
target4 = dataFrame4['label'].values
dataFrame4 = dataFrame4.drop('label',axis=1).values
model4 = svm.SVC(kernel='linear')
model4.fit(dataFrame4,target4)
for current_set in setlist:
    dataFrame = allData[allData['label'].isin(current_set)]
    target = dataFrame['label'].values
    dataFrame = dataFrame.drop('label',axis=1).values
    model = svm.SVC(kernel='linear')
    model.fit(dataFrame,target)
    modelList.append(model)


target = allData['label'].values
allData = allData.drop('label',axis=1).values
train,test,train_target,test_target = train_test_split(allData,target,test_size = 0.2,stratify=target)

train_target = pandas.DataFrame(train_target,columns=['label'])
for i in range(5):
    train_target['label'][train_target['label'].isin(setlist[i])] = str(i)
train_target = train_target['label'].values
setmodel = svm.SVC(kernel='linear')

model.fit(train,train_target)
setPredictions = model.predict(test)


test_df = pandas.DataFrame(test,columns= list(initialData.columns.values).remove('label'))
test_df['label'] = setPredictions
print(len(setPredictions))
for i in range(4):
    current_predict = modelList[i].predict(test_df[test_df['label'] == str(i)].drop('label',axis=1).values)
    test_df['label'][test_df['label'] == str(i)] = current_predict
current_predict = model4.predict(test_df[test_df['label'] == str(4)].drop('label',axis=1).values)
test_df['label'][test_df['label'] == str(4)] = current_predict
predictions = test_df['label'].values

print(accuracy_score(test_target,predictions))
print(confusion_matrix(test_target,predictions,labels= initialData['label'].unique()))
