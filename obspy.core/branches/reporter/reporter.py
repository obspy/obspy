from StringIO import StringIO
from xml.etree import ElementTree as etree
import BaseHTTPServer
import sqlite3
import time
import datetime
import cgi


HOST_NAME = 'localhost'
PORT_NUMBER = 8080
DB_NAME = "reporter.db"

# create db connection
conn = sqlite3.connect(DB_NAME)

# create tables
try:
    conn.execute('''
        CREATE TABLE report (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            errors INTEGER,
            failures INTEGER,
            system TEXT,
            architecture TEXT,
            version TEXT,
            xml TEXT)
    ''')
except:
    pass



INSERT_SQL = """
    INSERT INTO report (timestamp, errors, failures, system, architecture, 
        version, xml) 
    VALUES (?, ?, ?, ?, ?, ?, ?)
"""

SELECT_ALL_SQL = """
    SELECT * FROM report ORDER BY timestamp DESC LIMIT 20
"""

SELECT_SQL = """
    SELECT * FROM report WHERE id=?
"""

class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):

    def do_GET(self):
        """
        Respond to a GET request.
        """
        if self.path.startswith('/?xml_id='):
            try:
                id = self.path[9:]
                result = conn.execute(SELECT_SQL, (id))
                item = result.fetchone()
            except:
                self.send_response(200)
                return
            self.send_response(200)
            self.send_header("Content-type", "application/xhtml+xml")
            self.end_headers()
            self.wfile.write(item[7])
        elif self.path.startswith('/?id='):
            try:
                id = self.path[5:]
                result = conn.execute(SELECT_SQL, (id))
                item = result.fetchone()
                root = etree.parse(StringIO(item[7])).getroot()
            except:
                self.send_response(200)
                return
            self.send_response(200)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write("<html>")
            self.wfile.write("<head>")
            self.wfile.write("<title>Report #%s</title>" % id)
            self.wfile.write("<style type='text/css'>")
            self.wfile.write("pre {background-color:#EEEEEE;}")
            self.wfile.write("</style>")
            self.wfile.write("</head>")
            self.wfile.write("<body><h1>Report #%s</h1>" % id)
            self.wfile.write("<p><a href='/'>Return to overview</a></p>")
            self.wfile.write("<h2>Failures/Errors</h2>")
            for item in root.find('errors')._children:
                self.wfile.write("<pre>%s</pre" % item.text)
            for item in root.find('failures')._children:
                self.wfile.write("<pre>%s</pre" % item.text)
            self.wfile.write("<h2>Platform</h2>")
            self.wfile.write("<ul>")
            for item in root.find('platform')._children:
                self.wfile.write("<li>%s : %s</li>" % (item.tag, item.text))
            self.wfile.write("</ul>")
            self.wfile.write("<h2>ObsPy</h2>")
            self.wfile.write("<ul>")
            for item in root.find('obspy')._children:
                self.wfile.write("<li>%s : %s</li>" % (item.tag, item.text))
            self.wfile.write("</ul>")
            self.wfile.write("<h2>Dependencies</h2>")
            self.wfile.write("<ul>")
            for item in root.find('dependencies')._children:
                self.wfile.write("<li>%s : %s</li>" % (item.tag, item.text))
            self.wfile.write("</ul>")
            self.wfile.write("</body></html>")
        else:
            results = conn.execute(SELECT_ALL_SQL)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write("<html>")
            self.wfile.write("<head><title>Last 20 reports</title></head>")
            self.wfile.write("<body><h1>Reports</h1>")
            self.wfile.write('<table width="100%" border="1">')
            self.wfile.write("<tr>")
            self.wfile.write("<th>Timestamp</th>")
            self.wfile.write("<th>Failures/Errors</th>")
            self.wfile.write("<th>System</th>")
            self.wfile.write("<th>Architecture</th>")
            self.wfile.write("<th>Python Version</th>")
            self.wfile.write("<th>XML</th>")
            self.wfile.write("</tr>")
            for item in results:
                self.wfile.write("<tr>")
                dt = datetime.datetime.fromtimestamp(item[1])
                self.wfile.write("<td>%s</td>" % dt)
                if int(item[3]) > 0:
                    color = "#FF0000"
                elif int(item[2]) > 0:
                    color = "#FF7700"
                else:
                    color = "#00FF00"
                self.wfile.write("<td style='background-color:%s'>" % color)
                self.wfile.write("%s/%s</td>" % (item[2], item[3]))
                self.wfile.write("<td>%s</td>" % item[4])
                self.wfile.write("<td>%s</td>" % item[5])
                self.wfile.write("<td>%s</td>" % item[6])
                self.wfile.write("<td>")
                self.wfile.write("<a href='?id=%s'>" % item[0])
                self.wfile.write("[Error report]</a>")
                self.wfile.write(" | ")
                self.wfile.write("<a href='?xml_id=%s'>" % item[0])
                self.wfile.write("[XML]</a>")
                self.wfile.write("</td>")
                self.wfile.write("</tr>")
            self.wfile.write("</table>")
            self.wfile.write("</body></html>")

    def do_POST(self):
        """
        Respond to a POST request.
        """
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                'REQUEST_METHOD': 'POST',
                'CONTENT_TYPE': self.headers['Content-Type'],
            })
        try:
            ts = int(form['timestamp'].value)
            xml_doc = form['xml'].value
            errors = int(form['errors'].value)
            failures = int(form['failures'].value)
            system = form['system'].value
            python_version = form['python_version'].value
            architecture = form['architecture'].value
            conn.execute(INSERT_SQL, (ts, failures, errors, system,
                                      architecture, python_version, xml_doc))
            conn.commit()
        except Exception, e:
            self.send_response(500, str(e))
        else:
            self.send_response(200)


if __name__ == '__main__':
    server_class = BaseHTTPServer.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    print time.asctime(), "Server Starts - %s:%s" % (HOST_NAME, PORT_NUMBER)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print time.asctime(), "Server Stops - %s:%s" % (HOST_NAME, PORT_NUMBER)
