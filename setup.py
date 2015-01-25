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

import glob
import inspect
import os
import sys


ROOT = os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe())))


def __get_model_filename(model_name, model_dir):
    model_name = os.path.splitext(os.path.basename(model_name))[0]
    filename = os.path.join(
        model_dir, model_name +
        ("__py%i%i__tvel" % sys.version_info[:2]) + os.path.extsep + "pickle")
    return filename


def build_models():
    """
    Builds the models during install time. This is needed as the models are
    pickled Python classes which are not compatible across Python versions.
    """
    taupy_path = os.path.join(ROOT, "taupy")
    model_input = os.path.join(taupy_path, "data")
    model_dir = os.path.join(model_input, "taup_models")

    sys.path.insert(0, ROOT)
    from taupy.TauP_Create import TauP_Create

    for model in glob.glob(os.path.join(model_input, "*.tvel")):
        print("Building model '%s'..." % model)
        sys.stdout.flush()
        output_filename = __get_model_filename(model, model_dir)
        mod_create = TauP_Create(input_filename=model,
                                 output_filename=output_filename)
        mod_create.loadVMod()
        mod_create.run()


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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics'],
    url="https://github.com/obspy/TauPy",
    license="GNU General Public License, version 3 (GPLv3)",
    platforms="OS Independent",
    requires=["numpy", "future"],
    include_package_data=True,
    # this is needed for "easy_install taupy==dev"
    download_url=("https://github.com/obspy/TauPy/zipball/master"
                  "#egg=obspy=dev"),
)


if __name__ == "__main__":
    build_models()
    setup(**setup_config)
