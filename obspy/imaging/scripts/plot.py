#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wiggle plot of the data in files
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from argparse import ArgumentParser

from obspy import Stream, __version__, read
from obspy.core.util.base import ENTRY_POINTS
from obspy.core.util.misc import MatplotlibBackend


def main(argv=None):
    parser = ArgumentParser(prog='obspy-plot', description=__doc__.strip())
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + __version__)
    parser.add_argument('-f', '--format', choices=ENTRY_POINTS['waveform'],
                        help='Waveform format.')
    parser.add_argument('-o', '--outfile',
                        help='Output filename.')
    parser.add_argument('-n', '--no-automerge', dest='automerge',
                        action='store_false',
                        help='Disable automatic merging of matching channels.')
    parser.add_argument('files', nargs='+',
                        help='Files to plot.')
    args = parser.parse_args(argv)

    if args.outfile is not None:
        MatplotlibBackend.switch_backend("AGG", sloppy=False)

    st = Stream()
    for f in args.files:
        st += read(f, format=args.format)
    st.plot(outfile=args.outfile, automerge=args.automerge)


if __name__ == "__main__":
    main()
