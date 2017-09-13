#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Print stream information for waveform data in local files.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from argparse import ArgumentParser

from obspy import Stream, __version__, read
from obspy.core.util.base import ENTRY_POINTS


def main(argv=None):
    parser = ArgumentParser(prog='obspy-print', description=__doc__.strip())
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + __version__)
    parser.add_argument('-f', '--format', choices=ENTRY_POINTS['waveform'],
                        help='Waveform format (slightly faster if specified).')
    parser.add_argument('-n', '--no-merge', action='store_false',
                        dest='merge', help='Switch off cleanup merge.')
    parser.add_argument('--no-sorting', action='store_false',
                        dest='sort', help='Switch off sorting of traces.')
    parser.add_argument('-g', '--print-gaps', action='store_true',
                        help='Switch on printing of gap information.')
    parser.add_argument('files', nargs='+',
                        help='Files to process.')

    args = parser.parse_args(argv)

    st = Stream()
    for f in args.files:
        st += read(f, format=args.format)
    if args.merge:
        st.merge(-1)
    if args.sort:
        st.sort()
    print(st.__str__(extended=True))
    if args.print_gaps:
        print()
        st.print_gaps()


if __name__ == "__main__":
    main()
