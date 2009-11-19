#!/usr/bin/python
# -*- coding: utf-8 -*-
from glob import glob
from obspy.xseed.parser import Parser
from optparse import OptionParser
import os
import sys


def xseed2dataless(filename, outfile, verbose, debug):
    if isinstance(filename, list):
        files = []
        for item in filename:
            files.extend(glob(item))
    else:
        files = glob(filename)
    if len(files) > 1 and outfile:
        msg = 'More than one filename is given.' + os.linesep
        msg += '\t--output argument will not be used.\n'
        sys.stdout.write(msg)
        outfile = None
    if verbose:
        msg = 'Found %s files.' % len(files) + os.linesep
        sys.stdout.write(msg)
    for file in files:
        f = open(file, 'r')
        if f.read(1) != '<':
            if verbose:
                msg = 'Skipping file %s' % file
                msg += '\t-- not a XML-SEED file' + os.linesep
                sys.stdout.write(msg)
            f.close()
            continue
        f.close()
        if not outfile:
            output = os.path.basename(file) + os.extsep + 'dataless'
        else:
            output = outfile
        if verbose:
            msg = 'Parsing file %s' % file + os.linesep
            sys.stdout.write(msg)
        try:
            parser = Parser(file, debug=debug)
            parser.writeSEED(output)
        except Exception, e:
            if debug:
                raise
            msg = '\tError parsing file %s' % file + os.linesep
            msg += '\t' + str(e) + os.linesep
            sys.stderr.write(msg)


def main():
    usage = "usage: %prog [options] filename"
    parser = OptionParser(usage)
    parser.add_option("-d", "--debug", default=False,
                      action="store_true", dest="debug",
                      help="show debugging information")
    parser.add_option("-q", "--quiet", default=True,
                      action="store_false", dest="verbose",
                      help="non verbose mode")
    parser.add_option("-o", "--output", dest="output", default=None,
                      help="output filename. Only valid when parsing one file")
    (options, args) = parser.parse_args()
    if len(args) == 0:
        parser.error("incorrect number of arguments")
    filenames = args
    if len(filenames) == 1:
        filenames = filenames[0]
    xseed2dataless(filenames, options.output, options.verbose, options.debug)


if __name__ == "__main__":
    main()
