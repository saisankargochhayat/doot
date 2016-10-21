from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold
# Build a classification task using 3 informative features
data = []
labels =[]
with open("all_data/rishav_all_a_f_data.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        # if(len(attr) > 1):
        #     attr[435] = attr[435][0:len(attr[435])-1]
        data.append(attr)
with open("all_data/rishav_all_a_f_label.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        # if(attr[0]!='\n'):
        labels.append(attr[0][0:len(attr[0])-1])
headers = []
with open("all_data/header.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        # if(attr[0]!='\n'):
        headers.append(attr)
headers = headers[0]
with open("all_data/lalu_all_a_f_data.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        # if(len(attr) > 1):
        #     attr[435] = attr[435][0:len(attr[435])-1]
        data.append(attr)
with open("all_data/lalu_all_a_f_label.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        # if(attr[0]!='\n'):
        labels.append(attr[0][0:len(attr[0])-1])
# print(len(labels))
# print(len(data[0]))
#
#
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# print(len(sel.fit_transform(data)[0]))
from sklearn.cross_validation import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = .1)
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
print(str(len(test_data)) + " " + str(len(train_data)))
selector = RFE(svc,200,step=100)
selector = selector.fit(train_data,train_labels)

rankings = selector.ranking_
for i in range(len(rankings)):
    print(headers[i] + " : " + str(rankings[i]))
