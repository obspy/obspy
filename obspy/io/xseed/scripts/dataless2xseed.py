#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A command-line program that converts Dataless SEED into XML-SEED files.
"""
import os
import sys
from argparse import ArgumentParser
from glob import glob

from obspy import __version__
from obspy.io.xseed.parser import Parser


def dataless2xseed(filename, options):
    files = []
    for item in filename:
        files.extend(glob(item))
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
        if f.read(7)[6:] != b'V':
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
            parser.write_xseed(output, version=str(options.version),
                               split_stations=options.split_stations)
        except Exception as e:
            if options.debug:
                raise
            msg = '\tError parsing file %s' % file + os.linesep
            msg += '\t' + str(e) + os.linesep
            sys.stderr.write(msg)


def main(argv=None):
    parser = ArgumentParser(prog='obspy-dataless2xseed',
                            description=__doc__.strip())
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + __version__)
    parser.add_argument('-s', '--split-stations', action='store_true',
                        help='split multiple stations within one dataless '
                             'file into multiple XML-SEED files, labeled with '
                             'the start time of the volume.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='show debugging information')
    parser.add_argument('-q', '--quiet', action='store_false', dest='verbose',
                        help='non verbose mode')
    parser.add_argument('-o', '--output', default=None,
                        help='output filename or directory')
    parser.add_argument('-x', '--xml-version', dest='version', default=1.1,
                        help='XML-SEED version, 1.0 or 1.1', type=float)
    parser.add_argument('files', nargs='+', help='files to process')

    args = parser.parse_args(argv)

    dataless2xseed(args.files, args)


if __name__ == "__main__":
    main()
