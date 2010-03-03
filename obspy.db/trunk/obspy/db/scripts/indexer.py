#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

from SimpleXMLRPCServer import SimpleXMLRPCServer
from obspy.db import __version__
from obspy.db.db import Base
from obspy.db.indexer import worker, WaveformFileCrawler
from optparse import OptionParser
from sqlalchemy import create_engine
from sqlalchemy.orm.session import sessionmaker
import logging
import multiprocessing
import os
import select
import sys


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class WaveformIndexer(SimpleXMLRPCServer, WaveformFileCrawler):
    """
    A waveform indexer server.
    """
    def __init__(self, session, in_queue, work_queue, out_queue, paths,
                 options):
        SimpleXMLRPCServer.__init__(self, ("localhost", 8000))
        # init db + options
        self.session = session
        self.options = options
        # set queues
        self.input_queue = in_queue
        self.work_queue = work_queue
        self.output_queue = out_queue
        self.paths = paths
        self._resetWalker()
        self._stepWalker()

    def serve_forever(self, poll_interval=0.5):
        self.running = True
        #self.__is_shut_down.clear()
        while self.running:
            r, _w, _e = select.select([self], [], [], poll_interval)
            if r:
                self._handle_request_noblock()
            self.iterate()
        #self.__is_shut_down.set()


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
    for i in range(options.number_of_cpus):
        args = (i, in_queue, work_queue, out_queue, options.preview_path)
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
    service = WaveformIndexer(session, in_queue, work_queue, out_queue, paths,
                              options)
    try:
        service.serve_forever(poll_interval=options.poll_interval)
    except KeyboardInterrupt:
        pass


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
