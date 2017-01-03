import pandas
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from helper import svm,sgd,knn,dtree,lda
import numpy as np
setlist = [['a','m','n','s','t','g','q','x','o'],['b','e','c'],['h','k','u','v'],
['d','r','p'],['f'],['l'],['i'],['w'],['y']]

initialData = pandas.read_csv('../CSV_Data/dataset_6.csv')
allData = initialData
matrix = allData.drop('label',axis=1).values
matrix = preprocessing.scale(matrix)
allData = pandas.DataFrame(matrix,columns = initialData.columns.drop('label'))
allData['label'] = initialData['label']
sum_acc = 0
sum_confusion = [[0 for x in range(9)] for y in range(9)]
for i in range(100):
    master_set = []
    for curr_set in setlist:
        master_set.append(curr_set[randint(0,len(curr_set)-1)])
    currentData = allData[allData['label'].isin(master_set)]
    acc,con = sgd.find_accuracy(currentData)
    sum_acc = sum_acc + acc
    sum_confusion = np.add(sum_confusion,con)

print(sum_confusion)
print(sum_acc/100)
