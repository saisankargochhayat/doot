
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

data = []
labels = []

knn_model= KNeighborsClassifier()
svm_model = svm.SVC()
sgd_model = SGDClassifier(loss="hinge", penalty="l2")
dtree_model = tree.DecisionTreeClassifier()


# ----------------------------Alice----------------------------
with open("master_data/Alice/a_f.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Alice/g_m.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Alice/n_s.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Alice/t_y.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
# ----------------------------Anisha----------------------------
with open("master_data/Anisha/a_f.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Anisha/g_m.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Anisha/n_s.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Anisha/t_y.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])

# ----------------------------Asutosh----------------------------
with open("master_data/Asutosh/a_f.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Asutosh/g_m.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Asutosh/n_s.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Asutosh/t_y.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])

# ----------------------------Rishav----------------------------
with open("master_data/Rishav/a_f.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Rishav/g_m.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Rishav/n_s.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Rishav/t_y.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])

# ----------------------------Sai----------------------------
with open("master_data/Sai/a_f.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Sai/g_m.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Sai/n_s.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Sai/t_y.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])

# ----------------------------Sandy----------------------------
with open("master_data/Sandy/a_f.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Sandy/g_m.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Sandy/n_s.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Sandy/t_y.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])

# ----------------------------Sohini----------------------------
with open("master_data/Sohini/a_f.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Sohini/g_m.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Sohini/n_s.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])
with open("master_data/Sohini/t_y.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:31])))
        labels.append(attr[31][0:-1])

from sklearn.cross_validation import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = .5)


knn_model.fit(train_data,train_labels)
svm_model.fit(train_data,train_labels)
sgd_model.fit(train_data,train_labels)
dtree_model.fit(train_data,train_labels)


predictions = knn_model.predict(test_data)
print(accuracy_score(test_labels, predictions))

predictions = svm_model.predict(test_data)
print(accuracy_score(test_labels, predictions))

predictions = sgd_model.predict(test_data)
print(accuracy_score(test_labels, predictions))

predictions = dtree_model.predict(test_data)
print(accuracy_score(test_labels, predictions))
