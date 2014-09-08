#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for the waveform solver input file generator.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import os
from setuptools import setup, find_packages

setup_config = dict(
    name="taupy",
    version="0.0.1a",
    description="Python port of TauP",
    packages=find_packages(),
    license="GNU General Public License, version 3 (GPLv3)",
    platforms="OS Independent",
)


if __name__ == "__main__":
    setup(**setup_config, requires=['numpy'])
