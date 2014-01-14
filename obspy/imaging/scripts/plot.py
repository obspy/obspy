#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
USAGE: obspy-plot [ -f format ] file1 file2 ...

Wiggle plot of the data in files
"""
from __future__ import unicode_literals
from obspy import read, Stream
from obspy import __version__
from optparse import OptionParser


def main():
    parser = OptionParser(__doc__.strip(), version="%prog " + __version__)
    parser.add_option("-f", default=None, type="string",
                      dest="format", help="Waveform format.")
    parser.add_option("-o", "--outfile", default=None, type="string",
                      dest="outfile", help="Output filename.")
    parser.add_option("-n", "--no-automerge", default=True, dest="automerge",
                      action="store_false",
                      help="Disable automatic merging of matching channels.")

    (options, args) = parser.parse_args()

    # Print help and exit if no arguments are given
    if len(args) == 0:
        parser.print_help()
        raise SystemExit()

    st = Stream()
    for arg in args:
        st += read(arg, format=options.format)
    st.plot(outfile=options.outfile, automerge=options.automerge)


if __name__ == "__main__":
    main()
