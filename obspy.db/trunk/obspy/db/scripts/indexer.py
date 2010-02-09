#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

from obspy.db import __version__
from obspy.db.indexer import runIndexer
from optparse import OptionParser
import multiprocessing



def main():
    usage = "USAGE: %prog [options]\n\n" + \
            "\n".join(__doc__.split("\n")[3:])
    parser = OptionParser(usage.strip(), version="%prog " + __version__)
    parser.add_option("-d", default='data', type="string", dest="data",
                      help="Path where waveform data files are situated. The "
                           "indexer will crawl recursive through all "
                           "sub-directories within this root path. Default "
                           "path is './data=*.*:'")
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
    runIndexer(options)


if __name__ == "__main__":
    main()
