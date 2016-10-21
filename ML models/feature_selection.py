from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
data = []
labels =[]
with open("../Collected Data/Rishav/a_f_rishav.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:27])))
        labels.append(attr[31][0:-1])

with open("../Collected Data/Rishav/g_m_rishav.csv", 'r') as ppd:
    for line in ppd:
        attr = line.split(',')
        data.append(list(map(float, attr[0:27])))
        labels.append(attr[31][0:-1])

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(data, labels)

print("Optimal number of features : %d" % rfecv.n_features_)
print(rfecv.support_)
