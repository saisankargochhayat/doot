import subprocess

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

class Application(tornado.web.Application):
    def __init__(self):
        handlers=[
        (r"/",HomeHandler),
        (r"/predictor",Predictor),
        (r"/visualizer",Visualizer),
        (r"/train",Train),
        (r"/predict",Predict),
        ]
        settings = dict(
            blog_title=u"Doot",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            ui_modules={"Entry": EntryModule},
            xsrf_cookies=True,
            cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            debug=True,
        )
        super(Application, self).__init__(handlers, **settings)
        class HomeHandler(BaseHandler):
            def get(self):
                self.render("index.html")

        class Predictor(BaseHandler):
            def get(self):
                self.render("predictor.html")

        class Visualizer(BaseHandler):
            def get(self):
                self.render("visualizer.html")

        class Train(BaseHandler):
            def get(self):
                self.write("Trained")

        class Predict(BaseHandler):
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

    if __name__ == "__main__":
        Application.listen(8080)
        tornado.ioloop.IOLoop.instance().start()
