from StringIO import StringIO
from xml.etree import ElementTree as etree
import BaseHTTPServer
import sqlite3
import time
import datetime


HOST_NAME = 'localhost'
PORT_NUMBER = 8080
DB_NAME = "reporter.db"

# create db connection
conn = sqlite3.connect(DB_NAME)

# create tables
try:
    conn.execute('''
        CREATE TABLE report (timestamp INTEGER, 
                             xml TEXT)
    ''')
except:
    pass


class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):

    def do_GET(request):
        """
        Respond to a GET request.
        """
        results = conn.execute("SELECT * FROM report LIMIT 20")
        request.send_response(200)
        request.send_header("Content-type", "text/html")
        request.end_headers()
        request.wfile.write("<html>")
        request.wfile.write("<head><title>Test reports</title></head>")
        request.wfile.write("<body><h1>Reports</h1>")
        request.wfile.write('<table width="100%" border="1">')
        request.wfile.write("<tr>")
        request.wfile.write("<th>Timestamp</th>")
        request.wfile.write("<th>Platform</th>")
        request.wfile.write("<th>ObsPy</th>")
        request.wfile.write("<th>Dependencies</th>")
        request.wfile.write("</tr>")
        for item in results:
            # parse xml
            xml = etree.parse(StringIO(item[1])).getroot()
            request.wfile.write("<tr>")
            dt = datetime.datetime.fromtimestamp(item[0])
            request.wfile.write("<td>%s</td>" % dt)
            # platform
            request.wfile.write("<td><ul>")
            for item in xml.find("platform")._children:
                name = item.tag
                version = item.text
                request.wfile.write("<li>%s : %s</li>" % (name, version))
            request.wfile.write("</ul></td>")
            # obspy
            request.wfile.write("<td><ul>")
            for item in xml.find("obspy")._children:
                name = item.tag
                version = item.text
                request.wfile.write("<li>%s : %s</li>" % (name, version))
            request.wfile.write("</ul></td>")
            # dependencies
            request.wfile.write("<td><ul>")
            for item in xml.find("dependencies")._children:
                name = item.tag
                version = item.text
                request.wfile.write("<li>%s : %s</li>" % (name, version))
            request.wfile.write("</ul></td>")
            request.wfile.write("</tr>")
        request.wfile.write("</table>")
        request.wfile.write("</body></html>")

    def do_PUT(request):
        """
        Respond to a PUT request.
        """
        import pdb;pdb.set_trace()
        #dt = result['timestamp']
        #conn.execute("INSERT INTO report VALUES (?, ?)", (dt, xml_doc))
        #conn.commit()
        request.send_response(200)


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
