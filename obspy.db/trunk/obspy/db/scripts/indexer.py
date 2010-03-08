#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

from obspy.db import __version__
from obspy.db.db import Base
from obspy.db.indexer import worker, WaveformFileCrawler
from optparse import OptionParser
from sqlalchemy import create_engine
from sqlalchemy.orm.session import sessionmaker
import BaseHTTPServer
import logging
import multiprocessing
import os
import select
import sys
import time


HOST_NAME = 'localhost'
PORT_NUMBER = 8080


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):

    def do_GET(self):
        """
        Respond to a GET request.
        """
        out = "<html><body>"
        out += "<pre>%s</pre>" % self.server.input_queue
        out += "<pre>%s</pre>" % self.server.work_queue
        out += "<pre>%s</pre>" % self.server.output_queue
        out += "</body></html>"
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(out)


class WaveformIndexer(BaseHTTPServer.HTTPServer, WaveformFileCrawler):
    """
    A waveform indexer server.
    """

    def serve_forever(self, poll_interval=0.5):
        self.running = True
        while self.running:
            r, _w, _e = select.select([self], [], [], poll_interval)
            if r:
                self._handle_request_noblock()
            self.iterate()


def _runIndexer(options):
    logging.info("Starting Indexer ...")
    # paths
    if ';' in options.data:
        items = options.data.split(';')
    else:
        items = [options.data]
    paths = {}
    for item in items:
        if '=' in item:
            path, wc = item.split('=', 1)
            wcs = wc.split(',')
        else:
            path = item
            wcs = ['*.*']
        # check path
        if not os.path.isdir(path):
            logging.warn("Skipping inaccessible path %s ..." % path)
            continue
        paths[path] = wcs
    if not paths:
        return
    # create file queue and worker processes
    manager = multiprocessing.Manager()
    in_queue = manager.dict()
    work_queue = manager.list()
    out_queue = manager.list()
    log_queue = manager.list()
    for i in range(options.number_of_cpus):
        args = (i, in_queue, work_queue, out_queue, log_queue,
                options.preview_path)
        p = multiprocessing.Process(target=worker, args=args)
        p.daemon = True
        p.start()
    # connect to database
    engine = create_engine(options.db_uri, encoding='utf-8',
                           convert_unicode=True)
    metadata = Base.metadata
    metadata.create_all(engine, checkfirst=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    # initialize crawler
    server_class = WaveformIndexer
    service = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    # init db + options
    service.session = session
    service.skip_dots = options.skip_dots
    service.cleanup = options.cleanup
    service.log = logging
    # set queues
    service.input_queue = in_queue
    service.work_queue = work_queue
    service.output_queue = out_queue
    service.log_queue = log_queue
    service.paths = paths
    service._resetWalker()
    service._stepWalker()
    print time.asctime(), "Server Starts - %s:%s" % (HOST_NAME,
                                                     PORT_NUMBER)
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    service.server_close()
    print time.asctime(), "Server Stops - %s:%s" % (HOST_NAME,
                                                    PORT_NUMBER)


def main():
    usage = "USAGE: %prog [options]\n\n" + \
            "\n".join(__doc__.split("\n")[3:])
    parser = OptionParser(usage.strip(), version="%prog " + __version__)
    parser.add_option("-d", default='data=*.*', type="string", dest="data",
                      help="Path and search pattern of waveform files. The "
                           "indexer will crawl recursive through all "
                           "sub-directories within each given path. Multiple "
                           "paths have to be separated with a semicolon, e.g. "
                           "'/first/path=*.*;/second/path;/third/path=*.gse'. "
                           "File patterns are given as comma-separated list of "
                           "wildcards after a equal sign. e.g. "
                           "'/path=*.gse2,*.mseed'. "
                           "Default path is 'data=*.*, '")
    parser.add_option("-p", default='preview', type="string",
                      dest="preview_path",
                      help="Path where all preview files are generated. "
                           "Default path is './preview'")
    parser.add_option("-n", type="int", dest="number_of_cpus",
                      help="Number of CPUs used for the indexer.",
                      default=multiprocessing.cpu_count())
    parser.add_option("-u", default='sqlite:///indexer.sqlite', type="string",
                      dest="db_uri",
                      help="Database connection URI, such as "
                           "postgresql://scott:tiger@localhost/mydatabase."
                           " Default is a SQLite database './indexer.sqlite'.")
    parser.add_option("-i", type="float", default=0.1, dest="poll_interval",
                      help="Poll interval for file crawler in seconds "
                           "(default is 0.1 seconds).")
    parser.add_option("--cleanup", action="store_true", dest="cleanup",
                      default=False,
                      help="Clean database from non-existing files or paths "
                           "if activated, but will skip all paths marked as "
                           "archived in the database.")
    parser.add_option("--all_files",
                      action="store_false", dest="skip_dots", default=True,
                      help="The indexer will automatically skip paths or "
                           "files starting with a dot. This option forces to "
                           "parse all paths and files.")

    (options, _) = parser.parse_args()
    _runIndexer(options)


if __name__ == "__main__":
    main()
