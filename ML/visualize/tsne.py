import pandas
import numpy as np
import os
path=os.getcwd()

path=path.strip('visualize')
import sys
sys.path.append(path)
from helper import misc_helper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
dataFrame = pandas.read_csv('../../CSV_Data/dataset_8.csv')
features,target = misc_helper.split_feature_target(dataFrame)
scaler = preprocessing.StandardScaler()
scaler.fit_transform(features)
tsne = TSNE(n_components = 2,random_state = 0)
x_test_2d = tsne.fit_transform(features)
plt.figure()
for idx,cl in enumerate(np.unique(features)):
    plt.scatter(x=features[cl,0],y = features[cl,1])
plt.show()
