#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
ObsPy - a Python framework for seismological observatories.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see Beyreuther et al. 2010, Megies et al. 2011).
The goal of the ObsPy project is to facilitate rapid application development
for seismology.

For more information visit http://www.obspy.org.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import os
import sys
import glob


def setupAll():
    basedir = os.path.abspath(os.path.dirname(__file__))
    paths = glob.glob('obspy.*')
    paths = [os.path.abspath(p) for p in paths]
    for path in paths:
        if not os.path.isdir(path):
            continue
        script_name = os.path.join(path, 'setup.py')
        if not os.path.exists(script_name):
            continue
        os.chdir(path)
        sys_call = 'python setup.py ' + ' '.join(sys.argv[1:])
        os.system(sys_call)
        os.chdir(basedir)


if __name__ == '__main__':
    setupAll()
