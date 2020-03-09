#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A command-line program that converts Dataless SEED into RESP files.
"""
import os
import sys
from argparse import ArgumentParser
from glob import glob

from obspy import __version__
from obspy.io.xseed.parser import Parser


def dataless2resp(filename, options):
    files = []
    for item in filename:
        files.extend(glob(item))
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
        if options.verbose:
            msg = 'Parsing file %s' % file + os.linesep
            sys.stdout.write(msg)
        try:
            parser = Parser(file, debug=options.debug)
            if options.zipped:
                folder = os.path.join(os.path.curdir, os.path.basename(file))
                parser.write_resp(folder=folder, zipped=True)
            else:
                parser.write_resp(folder=os.path.curdir, zipped=False)
        except Exception as e:
            if options.debug:
                raise
            msg = '\tError parsing file %s' % file + os.linesep
            msg += '\t' + str(e) + os.linesep
            sys.stderr.write(msg)


def main(argv=None):
    parser = ArgumentParser(prog='obspy-dataless2resp',
                            description=__doc__.strip())
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + __version__)
    parser.add_argument('-d', '--debug', action='store_true',
                        help='show debugging information')
    parser.add_argument('-q', '--quiet', action='store_false', dest='verbose',
                        help='non verbose mode')
    parser.add_argument('-z', '--zipped', action='store_true',
                        help='Pack files of one station into a ZIP archive.')
    parser.add_argument('files', nargs='+', help='Files to convert.')
    args = parser.parse_args(argv)

    dataless2resp(args.files, args)


if __name__ == "__main__":
    main()
