import subprocess
from bottle import run, post, request, response, get, route
from bottle import static_file
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn import svm
# import pprint

data = []
labels = []

knn_model= KNeighborsClassifier()
svm_model = svm.SVC()
sgd_model = SGDClassifier(loss="hinge", penalty="l2")
dtree_model = tree.DecisionTreeClassifier()
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
knn_model.fit(data,labels)
svm_model.fit(data,labels)
sgd_model.fit(data,labels)
dtree_model.fit(data,labels)
print("Trained")

@route('/',method='GET')
def index():
    return static_file('index.html',root='static/')

@route('/predictor',method='GET')
def predictor():
    return static_file('predictor.html',root='static/')

@route('/visualizer',method='GET')
def predictor():
    return static_file('visualizer.html',root='static/')

@route('/recorder',method='GET')
def predictor():
    return static_file('recorder.html',root='static/')

@route('/static/<filename>',method='GET')
def serve_static(filename):
    return static_file(filename,root='static/')
@route('/train',method = 'GET')
def train():
    return "Trained"

@route('/predict',method = 'POST')
def predict():
    test_data = request.json['ar'];
    # print(test_data)
    predictions = {};
    predictions['knn'] = str(knn_model.predict(test_data)[0])
    predictions['svm'] = str(svm_model.predict(test_data)[0])
    predictions['sgd'] = str(sgd_model.predict(test_data)[0])
    predictions['dtree'] = str(dtree_model.predict(test_data)[0])
    # print(predictions)
    return predictions
run(host='localhost', port=8080, debug=True)
