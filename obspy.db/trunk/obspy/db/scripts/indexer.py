#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

from obspy.db.indexer import runIndexer
from optparse import OptionParser
import multiprocessing


def main():
    usage = "USAGE: %prog [options] modules\n\n" + \
            "\n".join(__doc__.split("\n")[3:])
    parser = OptionParser(usage.strip())
    parser.add_option("-n", "--number-of-processors", type="int",
                      dest="number_of_processors",
                      help="Number of CPUs used for the indexer.",
                      default=multiprocessing.cpu_count())
    parser.add_option("-u", "--uri", default='sqlite:///indexer.sqlite',
                      type="string", dest="db_uri",
                      help="Database connection URI, such as " + \
                           "postgresql://scott:tiger@localhost/mydatabase")
    parser.add_option("-o", "--output-path", default='preview',
                      type="string", dest="output_path",
                      help="Path where preview files are generated.")
    parser.add_option("-c", "--cleanup", action="store_true", dest="cleanup",
                      default=False,
                      help="Cleans database from non-existing files or paths.")
    parser.add_option("-s", "--skip-dots",
                      action="store_false", dest="skip_dots", default=True,
                      help="Skips files or paths starting with a dot.")
    parser.add_option("-i", "--poll-interval", type="float", default=0.1,
                      dest="poll_interval",
                      help="Poll interval.")

    (options, _) = parser.parse_args()
    options.paths = parser.largs
    runIndexer(options)


if __name__ == "__main__":
    main()
