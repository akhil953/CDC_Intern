import os, os.path
import random
import string
import simplejson
import sys
import webbrowser

import cherrypy


class App:
    @cherrypy.expose
    def index(self):
        return open('home.html')

    @cherrypy.expose()
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def sendData(self):
        json_obj = cherrypy.request.json
        name = json_obj['name']
        age = json_obj['age']
        hr = json_obj['hr']
        bt = json_obj['bt']
        bp = json_obj['bp']
        bl = json_obj['bl']
        
        #do something with the data
	if(hr>=91 and hr<=100):
            print("excelent")
        elif(hr>=81 and hr<=90):
            print("good")
        elif(hr>=71 and hr<=80):
            print("normal")
        elif(hr>=61 and hr<=70):
            print("may be")
        elif(hr>=51 and hr<=60):
            print("less")
        elif(hr>=41 and hr<=50):
            print("poor")
        elif(hr>=0 and hr<=40):
            print(".....")
        else:
            print("Strange ..!!")

        #rslt= 20
        #return {"exists": rslt}

            
if __name__ == '__main__':
    conf = {
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './static'
        }
    }
    
    cherrypy.quickstart(App(), '/', conf)
