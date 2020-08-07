#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A command-line program that converts XML-SEED into Dataless SEED files.
"""
import os
import sys
from argparse import ArgumentParser
from glob import glob

from obspy import __version__
from obspy.io.xseed.parser import Parser


def xseed2dataless(filename, options):
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
        if f.read(1) != b'<':
            if options.verbose:
                msg = 'Skipping file %s' % file
                msg += '\t-- not a XML-SEED file' + os.linesep
                sys.stdout.write(msg)
            f.close()
            continue
        f.close()
        if outdir:
            output = os.path.join(outdir,
                                  os.path.basename(file) + os.extsep +
                                  'dataless')
        elif outfile:
            output = outfile
        else:
            output = os.path.basename(file) + os.extsep + 'dataless'
        if options.verbose:
            msg = 'Parsing file %s' % file + os.linesep
            sys.stdout.write(msg)
        try:
            parser = Parser(file, debug=options.debug)
            parser.write_seed(output)
        except Exception as e:
            if options.debug:
                raise
            msg = '\tError parsing file %s' % file + os.linesep
            msg += '\t' + str(e) + os.linesep
            sys.stderr.write(msg)


def main(argv=None):
    parser = ArgumentParser(prog='obspy-xseed2dataless',
                            description=__doc__.strip())
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + __version__)
    parser.add_argument('-d', '--debug', action='store_true',
                        help='show debugging information')
    parser.add_argument('-q', '--quiet', action='store_false', dest='verbose',
                        help='non verbose mode')
    parser.add_argument('-o', '--output', default=None,
                        help='output filename or directory')
    parser.add_argument('files', nargs='+', help='files to convert')
    args = parser.parse_args(argv)

    xseed2dataless(args.files, args)


if __name__ == "__main__":
    main()
