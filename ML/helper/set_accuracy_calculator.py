import pandas
import set_4
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

initialData = pandas.read_csv('../../CSV_Data/dataset_6.csv')
allData = initialData

matrix = allData.drop('label',axis=1).values
matrix = preprocessing.scale(matrix)
allData = pandas.DataFrame(matrix,columns = initialData.columns.drop('label'))
allData['label'] = initialData['label']

set_4.get_accuracy(allData)
