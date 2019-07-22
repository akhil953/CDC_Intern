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
        #do something with the data
            json_obj = cherrypy.request.json
            name = json_obj['name']
            age = json_obj['age']
            hr = json_obj['hr']
            bt = json_obj['bt']
            bp = json_obj['bp']
            bl = json_obj['bl']
            return {"exists": name}

            
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