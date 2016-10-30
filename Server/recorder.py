from bottle import run, post, request, response, get, route,Bottle
from bottle import static_file
from pymongo import MongoClient
import json
import csv
client = MongoClient()
db = client.doot

@route('/',method='GET')
def index():
    return static_file('index.html',root='static/')

@route('/recorder',method='GET')
def recorder():
    return static_file('recorder.html',root='static/')

@route('/static/<filename>',method='GET')
def serve_static(filename):
    return static_file(filename,root='static/')

@route('/savedata',method='POST')
def savedata():
    data = request.forms['frame']
    user = request.forms['user']
    collection = db[user]
    collection.insert_one(json.loads(data))
    return "Done"

run(host='localhost', port=8080, debug=True)
