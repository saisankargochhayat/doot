import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import sys
import os
path=os.getcwd()

path=path.strip('with_set')
path=path.strip('set_divide')
sys.path.append(path)
from helper import svm,sgd,knn,dtree,lda
import glob
path = os.path.split(path)[0]
path =path.strip('ML')
# allfiles = glob.glob(path+"/*")

setlist = [['a','m','n','s','t','g','q','o','x'],['b','e','c'],['h','k','u','v'],
['d','r','p'],['f'],['l'],['i'],['w'],['y']]
set_name_list = ['set_'+str(x) for x in range(len(setlist))]
# for filename in allfiles:
#
all_data = pandas.read_csv(path+'/CSV_Data/dataset_8.csv')
for curr_set,i in zip(setlist,set_name_list):
    all_data['label'][all_data['label'].isin(curr_set)] = i

sum_acc=0;
sum_conf_matrix = [[0 for x in range(len(set_name_list))] for y in range(len(set_name_list))]
for i in range(100):
    acc , conf = svm.find_accuracy(all_data)
    sum_acc = sum_acc + acc
    sum_conf_matrix = np.add(sum_conf_matrix,conf)
print(sum_acc/100)
