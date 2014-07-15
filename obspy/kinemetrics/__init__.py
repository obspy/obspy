#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
obspy.kinemetrics - EVT format support for ObsPy
================================================

Evt read support for ObsPy.

This module provides read support for the EVT Kinemetrics data format.
It is based on the Kinemetrics description of the format and the provided
C code (Kw2asc.c).

:copyright:
    The ObsPy Development Team (devs@obspy.org), Henri Martin, Thomas Lecocq,
    Kinemetrics(c)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
