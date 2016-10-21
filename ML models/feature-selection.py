from sklearn import tree
import numpy as np

data = []
labels = []

with open("../Collected data/Rishav/a_f_rishav.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])

#Recursive feature elimination ,Provides ranking of features
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(data, labels)
ranking = rfe.ranking_.reshape(31,)
print(ranking)

# L1-based feature selection, rejects a couple of weak/irrelavant/reductant features
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data, labels)
model = SelectFromModel(lsvc, prefit=True)
data_new = model.transform(data)
print(data_new.shape)

#Tree-based feature selection ,rejects a couple of weak/irrelavant/reductant weak features
#Better than L1 based feature_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier()
clf = clf.fit(data,labels)
model = SelectFromModel(clf, prefit=True)
data_neww = model.transform(data)
print(data_neww.shape)
