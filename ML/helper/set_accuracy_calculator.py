import pandas
import set_1
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

master_data = pandas.read_csv('../../CSV_Data/dataset_6.csv')

set_1.get_accuracy(master_data)
