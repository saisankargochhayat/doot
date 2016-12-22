import pandas
from helper import svm
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

setlist = [['a','m','n','s','t','g','q','x','o'],['b','e','c'],['h','k','u','v'],
['d','r','p'],['f'],['l'],['i'],['w'],['y']]

allData = pandas.read_csv('../CSV_Data/dataset_6.csv')
sum_acc = 0
for i in range(100):
    master_set = []
    for curr_set in setlist:
        master_set.append(curr_set[randint(0,len(curr_set)-1)])
    currentData = allData[allData['label'].isin(master_set)]
    train,test = train_test_split(currentData,test_size = 0.2,stratify=currentData['label'])

    model = svm.get_model(train)
    predictions = model.predict(test.drop('label',axis=1).values)
    sum_acc = sum_acc + accuracy_score(predictions,test['label'])
print(sum_acc/100)
