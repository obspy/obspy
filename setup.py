#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for TauPy.

:copyright:
    Nicolas Rothenhäusler (n.rothenhaeusler@campus.lmu.de)
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from setuptools import setup, find_packages

setup_config = dict(
    name="taupy",
    version="0.0.1a",
    description="Python port of TauP",
    packages=find_packages(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Library or ' +
        'Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics'],
    url="https://github.com/krischer/TauPy",
    license="GNU General Public License, version 3 (GPLv3)",
    platforms="OS Independent",
    requires=["numpy"],
    include_package_data=True,
    # this is needed for "easy_install taupy==dev"
    download_url=("https://github.com/krischer/TauPy/zipball/master"
                  "#egg=obspy=dev"),
)


if __name__ == "__main__":
    setup(**setup_config)
