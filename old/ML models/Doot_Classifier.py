from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from collections import Counter

def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

data = []
labels = []
train_data = []
train_labels = []
test_data = []
test_labels = []
predictions=[]
#Training the classifier using training dataset

with open("../Server/master_data/Sai/a_f.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        train_data.append(map(float, attr[0:31]))
        train_labels.append(attr[31][0:-1])


with open("../Server/master_data/Alice/a_f.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        test_data.append(map(float, attr[0:31]))
        test_labels.append(attr[31][0:-1])



clfknn = KNeighborsClassifier()
clfknn.fit(train_data,train_labels)
predictionsknn = clfknn.predict(test_data)


clfsvm = svm.SVC()
clfsvm.fit(train_data,train_labels)
predictionssvm = clfsvm.predict(test_data)

clfdt = tree.DecisionTreeClassifier()
clfdt.fit(train_data,train_labels)
predictionsdt = clfdt.predict(test_data)

clfsdg = SGDClassifier(loss="hinge", penalty="l2")
clfsdg.fit(train_data,train_labels)
predictionssdg = clfsdg.predict(test_data)

for i in range(len(test_data)):
  lst=[predictionssvm[i],predictionsknn[i],predictionsdt[i],predictionssdg[i]]
  predictions.append(Most_Common(lst))

from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels, predictions))
