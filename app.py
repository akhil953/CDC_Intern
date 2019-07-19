import random
import string
import cherrypy
from os.path import abspath
class StringGenerator(object):
    @cherrypy.expose
    def index(self):
        return """<!DOCTYPE html>
<html>
    <head>
        <title>HAPS</title>
        <meta charset="utf-8"/>
   
    </head>
    
    <body>

        <section id="inputData">
            <header>
                <ul>
                    <li><a href="https://github.com/akhil953/CDC_Intern.git" target="_blank">GitHub</a></li>
                    <li><a href="#" >About</a></li>
                </ul>
            </header>
        
            <form id="detais" action="#" method="POST">
                <p id="ctext">Check you heart status</p>
                <p><input type="text" placeholder="Name" ></p>
                <p><input type="number" placeholder="Age" required/></p>
                <p><input type="number" placeholder="Heart Rate" required/></p>
                <p><input type="number" placeholder="Body Temp" required/></p>
                <p><input type="number" placeholder="Blood Pressure" required/></p>
                <p><input type="number" placeholder="Blood Level" required/></p>
                <p><input type="submit" value="calculate"/></p>
            </form>
        </section>
        <h1>HAPS</h1>

        <a id="site" href="https://www.cdcllp.com/" target="_blank">www.cdcllp.com</a>
    </body>
    <style>
        body{
    width: 100%;
    height: 100vh;
    margin: 0;
    padding: 0;
    background-image: url("../IMG/bgblack.png");
    background-position: top;
    background-repeat: no-repeat;
    background-size: 100vw 100vh;
    font-family: sans-serif;
    display: grid;
    grid-template-columns: 1fr 1fr;
    overflow: hidden;
}
h1{
    color: white;
    font-size: 45px;
    grid-column: 2/3;
    text-align: right;
    padding-right: 40px;
}
#inputData{    
    grid-column: 1/2;
    display: grid;
    grid-template-rows: 10% 90%;
}
#ctext{
    color: rgb(168, 168, 168);
}
form{
    width: 100%;
    height: 100%;
    text-align: center;
    margin: auto;
    padding-top: 10%;
    grid-row: 2/3;
}
input[type="text"],input[type="number"]{
    background: transparent;
    border-radius: 10px;
    color:white;
    text-align: center;
    height: 30px;
    width: 280px;
    box-shadow: none;

    -webkit-transition: all 0.30s ease-in-out;
    -moz-transition: all 0.30s ease-in-out;
    -ms-transition: all 0.30s ease-in-out;
    -o-transition: all 0.30s ease-in-out;
    outline: none;
    border: 1px solid #3B3B3B;
}
input::placeholder {
    text-align: center;
    
    margin-top: 30px;
  }
  input[type=number]::-webkit-inner-spin-button, 
input[type=number]::-webkit-outer-spin-button { 
  -webkit-appearance: none; 
  margin: 0; 
}
input[type="submit"]{
    background-color: transparent;
    color: rgb(168, 168, 168);
    text-decoration: none;
    border-radius: 10px;
    border-style: solid;
    height: 30px;
    width: auto;

    -webkit-transition: all 0.10s ease-in-out;
    -moz-transition: all 0.10s ease-in-out;
    -ms-transition: all 0.10s ease-in-out;
    -o-transition: all 0.10s ease-in-out;
    outline: none;
    border: 1px solid rgb(168, 168, 168);
}
input:focus{
    box-shadow: 0 0 5px white;
    width: 300px;
    border: 1px solid white;
  }
input[type="submit"]:focus{
    box-shadow: 0 0 2px white;
    width: auto;
    border: 1px solid white;
  }
ul{
    grid-row: 1/2;
    padding-top: 20px;
}
li{
    display: inline;
    padding-right: 30px;
}
a{
    text-decoration: none;
    text-align: bottom ;
    color: white;
}
#site{
    
    position: absolute;
    bottom: 25px;
    right: 40px;
    grid-column: 2/3;
    
    margin: 0;
    padding: 0;
}
    </style>
        </html>"""

    @cherrypy.expose
    def generate(self, length=8):
        return ''.join(random.sample(string.hexdigits, int(length)))


if __name__ == '__main__':
    cherrypy.quickstart(StringGenerator())

