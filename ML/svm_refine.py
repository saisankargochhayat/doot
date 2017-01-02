import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from helper import svm
dataFrame = pandas.read_csv('../CSV_Data/dataset_6.csv')
max_acc = 0
best_kernel = 'linear'
best_degree = 1
degrees = [x for x in range(1,6)]
kernels = ['poly']
for kernel in kernels:
    for degree in degrees:
        sum_acc = 0
        for i in range(50):
            acc,confusion = svm.find_accuracy(dataFrame,kernel,degree)
            sum_acc = sum_acc+ acc
        acc = sum_acc/50
        print(str(kernel) + " " + str(degree) + " " +str(acc))
        if(acc > max_acc):
            max_acc = acc
            best_kernel = kernel
            best_degree = degree
print("Best Kernel : " + str(best_kernel))
print("Best Degree : " + str(best_degree))
