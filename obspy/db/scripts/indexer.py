#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A command-line program that indexes seismogram files into a database.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)

.. rubric:: Usage Examples

(1) Run indexer as daemon continuously crawling the given paths but index only
    the last 24 hours (-r24) of a waveform archive::

       #!/bin/bash
       DB=postgresql://username:password@localhost:5432/database
       DATA=/path/to/archive/2010,/path/to/archive/2011,/path/to/arclink
       LOG=/path/to/indexer.log
       ./obspy-indexer -v -i0.0 -n1 -u$DB -d$DATA -r24 -l$LOG &

(2) Run only once and remove duplicates::

       ./obspy-indexer -v -i0.0 --run_once --check_duplicates -n1 -u$DB -d$DATA
"""

from obspy import __version__
from obspy.db.db import Base
from obspy.db.indexer import worker, WaveformFileCrawler
from obspy.db.util import parseMappingData
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
        out = """<html>
  <head>
    <title>obspy-indexer status</title>
    <meta http-equiv="refresh" content="10" />
    <style type="text/css">
      th { text-align: left; font-family:monospace; width: 150px;
           vertical-align: top; padding: 3px; }
      td { font-family:monospace; padding: 3px;}
      pre { margin: 0; }
    </style>
  </head>
  <body>
    <h1>obspy-indexer</h1>
    <h2>Options</h2>
"""
        out += '<table>'
        for key, value in sorted(self.server.options.__dict__.items()):
            out += "<tr><th>%s</th><td>%s</td></tr>" % (key, value)
        if self.server.mappings:
            out += "<tr><th>mapping rules</th><td>%s</td></tr>" % \
                   (self.server.mappings)
        out += '</table>'
        out += '<h2>Status</h2>'
        out += '<table>'
        out += "<tr><th>current path</th><td>%s</td></tr>" % \
            (self.server._current_path)
        out += "<tr><th>patterns</th><td><pre>%s</pre></td></tr>" % \
            ('\n'.join(self.server.patterns))
        out += "<tr><th>features</th><td><pre>%s</pre></td></tr>" % \
            ('\n'.join(self.server.features))
        out += "<tr><th>file queue</th><td><pre>%s</pre></td></tr>" % \
            ('\n'.join(self.server._current_files))
        out += '</table>'
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
    service.log = logging
    try:
        # prepare paths
        if ',' in options.data:
            paths = options.data.split(',')
        else:
            paths = [options.data]
        paths = service._preparePaths(paths)
        if not paths:
            return
        # prepare map file
        if options.map_file:
            data = open(options.map_file, 'r').readlines()
            mappings = parseMappingData(data)
            logging.info("Parsed %d lines from mapping file %s" %
                         (len(data), options.map_file))
        else:
            mappings = {}
        # create file queue and worker processes
        manager = multiprocessing.Manager()
        in_queue = manager.dict()
        work_queue = manager.list()
        out_queue = manager.list()
        log_queue = manager.list()
        # spawn processes
        for i in range(options.number_of_cpus):
            args = (i, in_queue, work_queue, out_queue, log_queue, mappings)
            p = multiprocessing.Process(target=worker, args=args)
            p.daemon = True
            p.start()
        # connect to database
        engine = create_engine(options.db_uri, encoding='utf-8',
                               convert_unicode=True)
        metadata = Base.metadata
        # recreate database
        if options.drop_database:
            metadata.drop_all(engine, checkfirst=True)
        metadata.create_all(engine, checkfirst=True)
        # initialize database + options
        Session = sessionmaker(bind=engine)
        service.session = Session
        service.options = options
        service.mappings = mappings
        # set queues
        service.input_queue = in_queue
        service.work_queue = work_queue
        service.output_queue = out_queue
        service.log_queue = log_queue
        service.paths = paths
        service._resetWalker()
        service._stepWalker()
        service.serve_forever(options.poll_interval)
    except KeyboardInterrupt:
        quit()
    logging.info("Indexer stopped.")


def main():
    usage = "USAGE: %prog [options]\n\n" + \
            "\n".join(__doc__.split("\n")[3:])
    parser = OptionParser(usage.strip(), version="%prog " + __version__)
    parser.add_option(
        "-d", default='data=*.*', type="string", dest="data",
        help="""Path, search patterns and feature plug-ins of waveform files.
The indexer will crawl recursive through all sub-directories within each given
path. Multiple paths have to be separated with a comma, e.g.
'/first/path=*.*,/second/path,/third/path=*.gse'.
File patterns are given as space-separated list of wildcards after a equal
sign, e.g.
'/path=*.gse2 *.mseed,/second/path=*.*'.
Feature plug-ins may be added using the hash (#) character, e.g.
'/path=*.mseed#feature1#feature2,/second/path#feature2'.
Be aware that features must be provided behind file patterns (if any)! There is
no default feature enabled.
Default path option is 'data=*.*'.""")
    parser.add_option(
        "-u", default='sqlite:///indexer.sqlite', type="string",
        dest="db_uri",
        help="Database connection URI, such as "
             "postgresql://scott:tiger@localhost/mydatabase."
             "Default is a SQLite database './indexer.sqlite'.")
    parser.add_option(
        "-n", type="int", dest="number_of_cpus",
        help="Number of CPUs used for the indexer.",
        default=multiprocessing.cpu_count())
    parser.add_option(
        "-i", type="float", default=0.1, dest="poll_interval",
        help="Poll interval for file crawler in seconds (default is 0.1).")
    parser.add_option(
        "-r", type="int", dest="recent", default=0,
        help="Index only recent files modified within the given" +
             "number of hours. This option is deactivated by default.")
    parser.add_option(
        "-v", action="store_true", dest="verbose", default=False,
        help="Verbose output.")
    parser.add_option(
        "-l", type="string", dest="log", default="",
        help="Log file name. If no log file is given, stdout will be used.")
    parser.add_option(
        "-m", "--mapping_file", type="string", dest="map_file",
        help="Correct network, station, location and channel codes using a" +
             " custom mapping file.", default=None)
    parser.add_option(
        "--all_files", action="store_false", dest="skip_dots",
        default=True,
        help="The indexer will automatically skip paths or "
             "files starting with a dot. This option forces to "
             "parse all paths and files.")
    parser.add_option(
        "-1", "--run_once", action="store_true",
        dest="run_once", default=False,
        help="The indexer will parse cycle through all given directories only "
             "once and quit afterwards.")
    parser.add_option(
        "--check_duplicates", action="store_true",
        dest="check_duplicates", default=False,
        help="Checks for duplicate entries within database." +
             "This feature will slow down the indexer progress.")
    parser.add_option(
        "--cleanup", action="store_true", dest="cleanup",
        default=False,
        help="Clean database from non-existing files or paths " +
             "if activated, but will skip all paths marked as " +
             "archived in the database.")
    parser.add_option(
        "--force-reindex", action="store_true",
        dest="force_reindex", default=False,
        help="Reindex existing index entry for every crawled file.")
    parser.add_option(
        "--drop_database", action="store_true",
        dest="drop_database", default=False,
        help="Deletes and recreates the complete database at start up.")
    parser.add_option(
        "--host", type="string", dest="host",
        help="Server host name. Default is 'localhost'.", default="localhost")
    parser.add_option(
        "--port", type="int", dest="port", default=0,
        help="Port number. If not given a free port will be picked.")

    (options, _) = parser.parse_args()
    # set level of verbosity
    if options.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    if options.log == "":
        logging.basicConfig(stream=sys.stdout, level=level,
                            format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        logging.basicConfig(filename=options.log, level=level,
                            format="%(asctime)s [%(levelname)s] %(message)s")
    _runIndexer(options)


if __name__ == "__main__":
    main()
