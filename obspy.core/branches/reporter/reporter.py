from StringIO import StringIO
from daemon import Daemon
from xml.etree import ElementTree as etree
import BaseHTTPServer
import cgi
import datetime
import sqlite3
import sys
import time


HOST_NAME = 'localhost'
PORT_NUMBER = 8080
DB_NAME = "/home/barsch/opt/reporter/reporter.db"

# create db connection
conn = sqlite3.connect(DB_NAME)

# create tables
try:
    conn.execute('''
        CREATE TABLE report (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            tests INTEGER,
            errors INTEGER,
            modules INTEGER,
            system TEXT,
            architecture TEXT,
            version TEXT,
            xml TEXT)
    ''')
except:
    pass



INSERT_SQL = """
    INSERT INTO report (timestamp, tests, errors, modules, system, architecture, 
        version, xml) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

SELECT_ALL_SQL = """
    SELECT * FROM report ORDER BY timestamp DESC LIMIT 20
"""

SELECT_SQL = """
    SELECT * FROM report WHERE id=?
"""

class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):

    def _stylesheet(self):
        self.wfile.write("""
	<style type='text/css'>
          body {
	    font-family: Verdana; 
	    margin: 20px;
	  }
          pre {
	    background-color:#EEEEEE;
	  }
	  th {
	    background-color:#EEEEEE;
	    margin: 0;
	    padding: 5px;
	    text-align: center;
            border: 1px solid gray;
	  }
	  table {
	    border-collapse: collapse;
	    margin: 0;
	    padding: 0;
	  }
          td {
	    text-align: center;
            border: 1px solid gray;
	    margin: 0;
	    padding: 5px;
	  }
	  .error {
	    background-color: #FF0000;
	  }
	  .ok {
	    background-color: #00FF00;
	  }
        </style>
	""")

    def do_GET(self):
        """
        Respond to a GET request.
        """
        if self.path.startswith('/?xml_id='):
            try:
                id = self.path[9:]
                result = conn.execute(SELECT_SQL, (id,))
                item = result.fetchone()
            except:
                self.send_response(200)
                return
            self.send_response(200)
            self.send_header("Content-type", "text/xml")
            self.end_headers()
            self.wfile.write(item[8])
        elif self.path.startswith('/?id='):
            try:
                id = self.path[5:]
                result = conn.execute(SELECT_SQL, (id,))
                item = result.fetchone()
                root = etree.parse(StringIO(item[8])).getroot()
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
	    self._stylesheet()
            self.wfile.write("</head>")
            self.wfile.write("<body><h1>Report #%s</h1>" % id)
            self.wfile.write("<p><a href='?'>Return to overview</a></p>")
            self.wfile.write("<h2>Platform</h2>")
            self.wfile.write("<ul>")
            for item in root.find('platform')._children:
                self.wfile.write("<li><b>%s</b> : %s</li>" % (item.tag, item.text))
            self.wfile.write("</ul>")
            self.wfile.write("<h2>Dependencies</h2>\n")
            self.wfile.write("<ul>")
            for item in root.find('dependencies')._children:
                self.wfile.write("<li><b>%s</b> : %s</li>" % (item.tag, item.text))
            self.wfile.write("</ul>")
            self.wfile.write("<h2>ObsPy</h2>\n")
            self.wfile.write("<table>\n")
            self.wfile.write("  <tr>\n")
            self.wfile.write("    <th width='200'>Module</th>\n")
            self.wfile.write("    <th width='200'>Version</th>\n")
            self.wfile.write("    <th>Errors/Failures</th>\n")
            self.wfile.write("    <th>Tracebacks</th>\n")
            self.wfile.write("  </tr>\n")
	    errlog = ""
	    errid = 0
            for item in root.find('obspy')._children:
	        errcases = ""
                self.wfile.write("  <tr>\n")
	        version = item.find('installed').text
                self.wfile.write("    <td>obspy.%s</td>" % (item.tag))
   	        self.wfile.write("    <td>%s</td>" % (version))
		if item.find('tested')!=None:
 		    tests = int(item.find('tests').text)
		    errors = 0
                    for sitem in item.find('errors')._children:
                        errlog += "<a name='%d'><h5>#%d</h5></a>" % (errid, errid)
                        errlog += "<pre>%s</pre>" % sitem.text
			errcases += "<a href='#%d'>#%d</a> " % (errid, errid)
			errid += 1
			errors += 1
                    for sitem in item.find('failures')._children:
                        errlog += "<a name='%d'><h5>#%d</h5></a>" % (errid, errid)
                        errlog += "<pre>%s</pre>" % sitem.text
			errcases += "<a href='#%d'>#%d</a> " % (errid, errid)
			errors += 1
			errid += 1
                    if errors > 0:
                        color = "error"
                    else:
                        color = "ok"
                    self.wfile.write("    <td class='%s'>%d of %d</td>" % (color, errors, tests))
                else:
                    self.wfile.write("    <td>Not tested</td>\n")		    
                self.wfile.write("<td>%s</td>" % (errcases))
                self.wfile.write("  </tr>\n")
            self.wfile.write("</table>\n")
            self.wfile.write(errlog)
            try:
	        log = root.find('install_log').text
		self.wfile.write("<h2>Install Log</h2>")
                self.wfile.write("<pre>%s</pre" % log)
            except:
	        pass
            self.wfile.write("</body></html>")
        else:
            results = conn.execute(SELECT_ALL_SQL)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write("<html>")
            self.wfile.write("<head>")
            self.wfile.write("<title>Last 20 reports</title>")
	    self._stylesheet()
            self.wfile.write("</head>")
            self.wfile.write("<body><h1>Reports</h1>")
            self.wfile.write('<table>')
            self.wfile.write("<tr>")
            self.wfile.write("<th>Timestamp</th>")
            self.wfile.write("<th>Failures/Errors [# Modules]</th>")
            self.wfile.write("<th>System</th>")
            self.wfile.write("<th>Python Version</th>")
            self.wfile.write("<th>XML</th>")
            self.wfile.write("</tr>")
            for item in results:
                self.wfile.write("<tr>")
                dt = datetime.datetime.fromtimestamp(item[1])
                self.wfile.write("<td>%s</td>" % dt)
		errors = int(item[3])
		tests = int(item[2])
                if errors > 0:
                    color = "#FF0000"
                else:
                    color = "#00FF00"
                self.wfile.write("<td style='background-color:%s'>" % color)
                self.wfile.write('%s of %s &nbsp; [%s]' % (item[3], item[2], item[4]))
                self.wfile.write("</td>")
                self.wfile.write("<td>%s (%s)</td>" % (item[5], item[6]))
                self.wfile.write("<td>%s</td>" % item[7])
                self.wfile.write("<td>")
                self.wfile.write("[<a href='?id=%s'>" % item[0])
                self.wfile.write("Error report</a>]")
                self.wfile.write(" | ")
                self.wfile.write("[<a href='?xml_id=%s'>" % item[0])
                self.wfile.write("XML</a>]")
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
            modules = int(form['modules'].value)
            tests = int(form['tests'].value)
            system = form['system'].value
            python_version = form['python_version'].value
            architecture = form['architecture'].value
            conn.execute(INSERT_SQL, (ts, tests, errors, modules, system,
                                      architecture, python_version, xml_doc))
            conn.commit()
        except Exception, e:
            self.send_response(500, str(e))
        else:
            self.send_response(200)


def run(self):
    server_class = BaseHTTPServer.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    print time.asctime(), "Server Starts - %s:%s" % (HOST_NAME,
                                                     PORT_NUMBER)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print time.asctime(), "Server Stops - %s:%s" % (HOST_NAME,
                                                    PORT_NUMBER)

class MyDaemon(Daemon):
    run = run


if __name__ == "__main__":
    daemon = MyDaemon('/home/barsch/opt/reporter/reporter.pid')
    if len(sys.argv) == 2:
        if 'start' == sys.argv[1]:
            daemon.start()
        elif 'stop' == sys.argv[1]:
            daemon.stop()
        elif 'restart' == sys.argv[1]:
            daemon.restart()
        else:
            print "Unknown command"
            sys.exit(2)
        sys.exit(0)
    else:
        print "usage: %s start|stop|restart" % sys.argv[0]
        sys.exit(2)
