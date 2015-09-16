# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


def gen_sc3_id(dt, numenc=6, sym="abcdefghijklmnopqrstuvwxyz"):
    """
    Generate an event ID following the SeisComP3 convention. By default it
    divides a year into 26^6 intervals assigning each a unique combination of
    characters.

    >>> print _get_sc3_id(UTCDateTime(2015, 8, 18, 10, 55, 51, 367580))
    2015qffasl
    """
    numsym = len(sym)
    x = (((((dt.julday - 1) * 24) + dt.hour) * 60 + dt.minute) * 60 + dt.second) * 1000 \
    + dt.microsecond / 1000
    dx = (((370 * 24) * 60) * 60) * 1000
    rng = 1
    tmp = rng
    for i in xrange(numenc):
        rng *= numsym
    w = int(dx / rng)
    if w == 0:
        w = 1

    if  dx >= rng:
        x = int(x / w)
    else:
        x = x * int(rng / dx)
    enc = ''
    for i in xrange(numenc):
        d = int(x / numsym)
        r = x % numsym
        enc += sym[r]
        x = d
    return '%d%s' % (dt.year, enc[::-1])
