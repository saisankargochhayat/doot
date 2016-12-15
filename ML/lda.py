import pandas
from sklearn.lda import LDA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataFrame = pandas.read_csv('../CSV_Data/dataset_3.csv')
train,test = train_test_split(dataFrame,test_size = 0.2)
train_target = train['label'].values
train = train.drop('label',axis=1).values
actual = test['label'].values
test = test.drop('label',axis=1).values
model = LDA()
model.fit(train,train_target)
print(model.score(train,train_target))
predictions = model.predict(test)
print(accuracy_score(actual,predictions))
