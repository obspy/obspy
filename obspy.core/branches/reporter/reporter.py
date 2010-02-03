from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
from lxml import etree
import pymysql

# Create server
server = SimpleXMLRPCServer(("localhost", 8000))
server.register_introspection_functions()

# create mysql connection
conn = pymysql.connect(host='127.0.0.1', port=3306,
                       user='', passwd="", db='')
import pdb;pdb.set_trace()

class MyFuncs:
    def report(self, ok, result):
        root = etree.Element("report")
        for key, value in result.iteritems():
            if isinstance(value, dict):
                child = etree.SubElement(root, key)
                for subkey, subvalue in value.iteritems():
                    subkey = subkey.split('(')[0].strip()
                    if subvalue:
                        etree.SubElement(child, subkey).text = str(subvalue)
                    else:
                        etree.SubElement(child, subkey)
            elif value:
                etree.SubElement(root, key).text = str(value)
            else:
                etree.SubElement(root, key)
        cur = conn.cursor()
        cur.execute("SELECT Host,User FROM user")
        cur.close()
        print etree.tostring(root, pretty_print=True, encoding="UTF-8")
        return ""


server.register_instance(MyFuncs())


# Run the server's main loop
try:
    server.serve_forever()
except KeyboardInterrupt:
    conn.close()
