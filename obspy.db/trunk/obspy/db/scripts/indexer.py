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
import select
import sys


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
    logging.info("Starting indexer %s:%s ..." % (options.host, options.port))
    # initialize crawler
    service = WaveformIndexer((options.host, options.port), MyHandler)
    try:
        # paths
        if ',' in options.data:
            paths = options.data.split(',')
        else:
            paths = [options.data]
        paths = service._preparePaths(paths)
        if not paths:
            return
        # create file queue and worker processes
        manager = multiprocessing.Manager()
        in_queue = manager.dict()
        work_queue = manager.list()
        out_queue = manager.list()
        log_queue = manager.list()
        cpu_list = []
        for i in range(options.number_of_cpus):
            args = (i, in_queue, work_queue, out_queue, log_queue)
            p = multiprocessing.Process(target=worker, args=args)
            p.daemon = True
            p.start()
            cpu_list.append(p)
        # connect to database
        engine = create_engine(options.db_uri, encoding='utf-8',
                               convert_unicode=True)
        metadata = Base.metadata
        metadata.create_all(engine, checkfirst=True)
        # init db + options
        service.session = sessionmaker(bind=engine)
        service.skip_dots = options.skip_dots
        service.cleanup = options.cleanup
        service.log = logging
        # set queues
        service.input_queue = in_queue
        service.work_queue = work_queue
        service.output_queue = out_queue
        service.log_queue = log_queue
        service.number_of_cpus = options.number_of_cpus
        service.paths = paths
        service._resetWalker()
        service._stepWalker()
        service.serve_forever()
    except KeyboardInterrupt:
        quit()
    logging.info("Indexer stopped.")


def main():
    usage = "USAGE: %prog [options]\n\n" + \
            "\n".join(__doc__.split("\n")[3:])
    parser = OptionParser(usage.strip(), version="%prog " + __version__)
    parser.add_option("-d", default='data=*.*', type="string", dest="data",
        help="""Path, search patterns and feature plug-ins of waveform files.
The indexer will crawl recursive through all sub-directories within each given
path. Multiple paths have to be separated with a comma, e.g.
'/first/path=*.*,/second/path,/third/path=*.gse'.
File patterns are given as space-separated list of wildcards after a equal
sign, e.g.
'/path=*.gse2 *.mseed;/second/path=*.*'.
Feature plug-ins are given as space-separated list of plug-in names after a
semicolon, e.g.
'/path=*.mseed;feature1 feature2,/second/path;feature1'.
Be aware that features must be provided behind file patterns (if any)! There is
no default feature enabled.
Default path option is 'data=*.*'.""")
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
    parser.add_option("-v", action="store_true", dest="verbose",
                      default=False,
                      help="Verbose output.")
    parser.add_option("-l", type="string", dest="log",
                      help="Log file name. If no log file is given, stdout" + \
                      "is used.", default="")
    parser.add_option("-s", type="string", dest="host",
                      help="Server host name. Default is 'localhost'.",
                      default="localhost")
    parser.add_option("-p", type="int", dest="port",
                      help="Port number. Default is '8081'.",
                      default=8081)
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
    # set level of verbosity
    if options.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    if options.log == "":
        logging.basicConfig(stream=sys.stdout, level=level)
    else:
        logging.basicConfig(filename=options.log, level=level)
    _runIndexer(options)


if __name__ == "__main__":
    main()
