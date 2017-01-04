from tornado import websocket, web, ioloop
import sys
sys.path.insert(0,'/home/rishi/doot/ML')
import tornado.escape
from tornado import gen
import tornado.httpserver
import tornado.options
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import collections
# from data_loader import data_loader
import json
import pprint
import pandas
from sklearn import svm
import numpy as np
from  tornado.escape import json_decode
from  tornado.escape import json_encode
from feature_extracter_live import *
from sklearn import preprocessing
from helper import svm,knn,dtree,sgd,lda
from textblob import TextBlob
# define("port", default=8080, help="run on the given port", type=int)

data = []
labels = []
dataFrame = pandas.read_csv('../CSV_Data/dataset_0.csv')
svm_model , svm_scaler = svm.get_model(dataFrame)
knn_model , knn_scaler = knn.get_model(dataFrame)
sgd_model , sgd_scaler = sgd.get_model(dataFrame)
dtree_model , dtree_scaler = dtree.get_model(dataFrame)
lda_model , lda_scaler = lda.get_model(dataFrame)
print("Trained")
sentence = ""
class HomeHandler(web.RequestHandler):
    def get(self):
        self.render("static/index.html")
class Words(web.RequestHandler):
    def get(self):
        self.render("static/words.html")
class Predictor(web.RequestHandler):
    def get(self):
        self.render("static/predictor.html")

class Visualizer(web.RequestHandler):
    def get(self):
        self.render("static/visualizer.html")

class Predict(websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        print("WebSocket opened")

    def on_message(self, message):
        global sentence
        msg = json.loads(message)
        test=extract_array(msg)
        predictions = {}
        vote = {}
        predictions['svm'] = str(svm_model.predict(svm_scaler.transform(test))[0])
        if predictions['svm'] in vote:
            vote[predictions['svm']] = vote[predictions['svm']]+1
        else:
            vote[predictions['svm']] = 1

        predictions['knn'] = str(knn_model.predict(knn_scaler.transform(test))[0])
        if predictions['knn'] in vote:
            vote[predictions['knn']] = vote[predictions['knn']]+1
        else:
            vote[predictions['knn']] = 1

        predictions['lda'] = str(lda_model.predict(lda_scaler.transform(test))[0])
        if predictions['lda'] in vote:
            vote[predictions['lda']] = vote[predictions['lda']]+1
        else:
            vote[predictions['lda']] = 1

        predictions['sgd'] = str(sgd_model.predict(sgd_scaler.transform(test))[0])
        if predictions['sgd'] in vote:
            vote[predictions['sgd']] = vote[predictions['sgd']]+1
        else:
            vote[predictions['sgd']] = 1

        predictions['dtree'] = str(dtree_model.predict(dtree_scaler.transform(test))[0])
        if predictions['dtree'] in vote:
            vote[predictions['dtree']] = vote[predictions['dtree']]+1
        else:
            vote[predictions['dtree']] = 1
        count = collections.Counter(vote)
        predictions['max_vote'] = count.most_common(1)[0][0]
        letter = predictions['max_vote']
        if(letter=='space' or letter=='back'):
            if(letter=='space'):
                a = sentence.split(" ")
                word = a[len(a)-1]
                blob = TextBlob(word)
                predictions['word'] = str(blob.correct())
                a[len(a)-1] = str(blob.correct())
                sentence = " ".join(a)
                sentence = sentence+" "
            else:
                sentence = sentence[:-1]
        else:
            sentence = sentence + letter
        self.write_message(predictions)

    def on_close(self):
        print("WebSocket closed")

app = web.Application([
    (r'/static/(.*)', web.StaticFileHandler, {'path': 'static/'}),
    (r"/",HomeHandler),
    (r"/predictor",Predictor),
    (r"/visualizer",Visualizer),
    (r"/words",Words),
    (r"/ws",Predict),
    ])

if __name__ == '__main__':
    app.listen(8080)
    print("Listening at 127.0.0.1:8080")
    ioloop.IOLoop.instance().start()
