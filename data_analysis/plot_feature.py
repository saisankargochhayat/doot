import pandas
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from collections import Counter
initialData = pandas.read_csv('../CSV_Data/dataset_5.csv')
dataFrame = initialData

matrix = dataFrame.drop('label',axis=1).values
# matrix = preprocessing.scale(matrix)
dataFrame = pandas.DataFrame(matrix,columns = initialData.columns.drop('label'))
dataFrame['label'] = initialData['label']

import statistics as s
ydata = []
y1data = []
labeldata = []
alphabets = [chr(x) for x in range(97,97+25)]
alphabets.remove('j')
for label in alphabets:
    curr_data = dataFrame['grabStrength'][dataFrame['label']==label].values
    ydata.append(s.pvariance(curr_data))
    y1data.append(curr_data.mean())
    labeldata.append(ord(label))

plt.xticks(labeldata,alphabets)
plt.plot(labeldata,ydata,color='red')
plt.plot(labeldata,y1data,color='blue')
plt.show()
