from tornado import websocket, web, ioloop
import tornado.escape
from tornado import gen
import tornado.httpserver
import tornado.options
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from data_loader import data_loader
# define("port", default=8080, help="run on the given port", type=int)

data = []
labels = []

knn_model= KNeighborsClassifier()
svm_model = svm.SVC()
sgd_model = SGDClassifier(loss="hinge", penalty="l2")
dtree_model = tree.DecisionTreeClassifier()

# data_loader is a function that loads data from the given csv files
data,labels = data_loader()

knn_model.fit(data,labels)
svm_model.fit(data,labels)
sgd_model.fit(data,labels)
dtree_model.fit(data,labels)
print("Trained")

class HomeHandler(web.RequestHandler):
    def get(self):
        self.render("static/index.html")

class Predictor(web.RequestHandler):
    def get(self):
        self.render("static/predictor.html")

class Visualizer(web.RequestHandler):
    def get(self):
        self.render("static/visualizer.html")

class Predict(web.RequestHandler):
    """docstring for ."""
    def post(self):
            test_data = request.json['ar'];
            # print(test_data)
            predictions = {};
            predictions['knn'] = str(knn_model.predict(test_data)[0])
            predictions['svm'] = str(svm_model.predict(test_data)[0])
            predictions['sgd'] = str(sgd_model.predict(test_data)[0])
            predictions['dtree'] = str(dtree_model.predict(test_data)[0])
            self.write(predictions)

app = web.Application([
    (r'/static/(.*)', web.StaticFileHandler, {'path': 'static/'}),
    (r"/",HomeHandler),
    (r"/predictor",Predictor),
    (r"/visualizer",Visualizer),
    (r"/predict",Predict),
    ])

if __name__ == '__main__':
    app.listen(8080)
    print("Listening at 127.0.0.1:8080")
    ioloop.IOLoop.instance().start()
