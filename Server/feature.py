import subprocess
from bottle import run, post, request, response, get, route
from bottle import static_file
import json
import csv
# import pprint
data_dict = {}
data_list = []
done=False
def store(obj,keyname):
    if(obj):
        for key in obj:
            value = obj[key]
            if(type(value) == type(data_dict)):
                store(value,keyname+"_"+key)
            else:
                if(type(value) == type(data_list)):
                    for i in range(len(value)):
                        store(obj,keyname+"_"+str(i))
                else:
                    data_dict[keyname] = value


def store(data,keyname):
    if(type(data) == type(data_dict)):
        for key in data:
            store(data[key],keyname+"_"+str(key))
    else:
        if(type(data) == type(data_list)):
            for i in range(len(data)):
                store(data[i],keyname+"_"+str(i))
        else:
            data_dict[keyname] = data

@route('/',method='GET')
def index():
    return static_file('feature_recorder.html',root='static/')

@route('/static/<filename>',method='GET')
def serve_static(filename):
    return static_file(filename,root='static/')

sample_str = "sample_str"
sample_bool = False
@route('/feature',method='POST')
def feature():
    global data_dict,done
    data = request.json
    store(data,"")
    print(len(data_dict))
    label = data_dict['_label']
    del data_dict['_label']
    deleting_key = []
    for key in data_dict:
        value = data_dict[key]
        if(type(value) == type(sample_str)):
            deleting_key.append(key)
        else:
            if(type(value) == type(sample_bool)):
                if(value):
                    print(key+" : "+str(value))
                    data_dict[key]=1
                else:
                    print(key+" : "+str(value))
                    data_dict[key] = 0
    for key in deleting_key:
        del data_dict[key]
    if(len(data_dict) == 436):
        with open('all_data/lalu_all_a_f_data.csv','a') as f:
            w = csv.DictWriter(f,delimiter=',',lineterminator='\n',fieldnames = data_dict.keys())
            if(not done):
                w.writeheader()
                done = True
    data_dict = {}
    return "Done"
run(host='localhost', port=8080, debug=True)
