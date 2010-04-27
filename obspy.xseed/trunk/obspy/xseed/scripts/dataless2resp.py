#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A command-line program that converts Dataless SEED into RESP files.
"""

from glob import glob
from obspy.xseed.parser import Parser
from optparse import OptionParser
import os
import sys


def dataless2resp(filename, options):
    if isinstance(filename, list):
        files = []
        for item in filename:
            files.extend(glob(item))
    else:
        files = glob(filename)
    if options.verbose:
        msg = 'Found %s files.' % len(files) + os.linesep
        sys.stdout.write(msg)
    for file in files:
        if not os.path.isfile(file):
            continue
        f = open(file, 'rb')
        if f.read(7)[6] != 'V':
            if options.verbose:
                msg = 'Skipping file %s' % file
                msg += '\t-- not a Dataless SEED file' + os.linesep
                sys.stdout.write(msg)
            f.close()
            continue
        f.close()
        if options.verbose:
            msg = 'Parsing file %s' % file + os.linesep
            sys.stdout.write(msg)
        try:
            parser = Parser(file, debug=options.debug)
            if options.zipped:
                folder = os.path.join(os.path.curdir, os.path.basename(file))
                parser.writeRESP(folder=folder, zipped=True)
            else:
                parser.writeRESP(folder=os.path.curdir, zipped=False)
        except Exception, e:
            if options.debug:
                raise
            msg = '\tError parsing file %s' % file + os.linesep
            msg += '\t' + str(e) + os.linesep
            sys.stderr.write(msg)


def main():
    usage = "USAGE: %prog [options] filename"
    parser = OptionParser(usage)
    parser.add_option("-d", "--debug", default=False,
                      action="store_true", dest="debug",
                      help="show debugging information")
    parser.add_option("-q", "--quiet", default=True,
                      action="store_false", dest="verbose",
                      help="non verbose mode")
    parser.add_option("-z", "--zipped", default=False,
                      action="store_true", dest="zipped",
                      help="Pack files of one station into a ZIP archive.")
    (options, args) = parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        return
    filenames = args
    if len(filenames) == 1:
        filenames = filenames[0]
    dataless2resp(filenames, options)


if __name__ == "__main__":
    main()
