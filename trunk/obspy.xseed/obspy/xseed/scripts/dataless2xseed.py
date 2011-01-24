#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A command-line program that converts Dataless SEED into XML-SEED files.
"""

from glob import glob
from obspy.xseed.parser import Parser
from optparse import OptionParser
import os
import sys


def dataless2xseed(filename, options):
    if isinstance(filename, list):
        files = []
        for item in filename:
            files.extend(glob(item))
    else:
        files = glob(filename)
    outdir = False
    outfile = False
    if options.output:
        if os.path.isdir(options.output):
            outdir = options.output
        elif len(files) > 1:
            msg = 'More than one filename is given.' + os.linesep
            msg += '\t--output argument will not be used.\n'
            sys.stdout.write(msg)
        else:
            outfile = options.output
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
        if outdir:
            output = os.path.join(outdir,
                                  os.path.basename(file) + os.extsep + 'xml')
        elif outfile:
            output = outfile
        else:
            output = os.path.basename(file) + os.extsep + 'xml'
        if options.verbose:
            msg = 'Parsing file %s' % file + os.linesep
            sys.stdout.write(msg)
        try:
            parser = Parser(file, debug=options.debug)
            parser.writeXSEED(output, version=str(options.version),
                              split_stations=options.split_stations)
        except Exception, e:
            if options.debug:
                raise
            msg = '\tError parsing file %s' % file + os.linesep
            msg += '\t' + str(e) + os.linesep
            sys.stderr.write(msg)


def main():
    usage = "USAGE: %prog [options] filename"
    parser = OptionParser(usage)
    parser.add_option("-s", "--split-stations", default=False,
                      action="store_true", dest="split_stations",
                      help="split multiple stations within one dataless file "
                           "into multiple XML-SEED files, labeled with the "
                           "start time of the volume.")
    parser.add_option("-d", "--debug", default=False,
                      action="store_true", dest="debug",
                      help="show debugging information")
    parser.add_option("-q", "--quiet", default=True,
                      action="store_false", dest="verbose",
                      help="non verbose mode")
    parser.add_option("-o", "--output", dest="output", default=None,
                      help="output filename or directory")
    parser.add_option("-v", "--version", dest="version", default=1.1,
                      help="XML-SEED version, 1.0 or 1.1", type="float")
    (options, args) = parser.parse_args()
    if len(args) == 0:
        parser.print_help()
        return
    filenames = args
    if len(filenames) == 1:
        filenames = filenames[0]
    dataless2xseed(filenames, options)


if __name__ == "__main__":
    main()
