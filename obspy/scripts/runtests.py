#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A command-line program that runs all ObsPy tests.

All tests in ObsPy are located in the tests directory of each specific
module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
.. rubric:: Examples

(1) Run all local tests (ignoring tests requiring a network connection) on
    command line::
        $ obspy-runtests
(2) Run all tests on command line (including network tests)::
        $ obspy-runtests --all
(3) Run tests of module :mod:`obspy.io.mseed`::
        $ obspy-runtests io/mseed
(4) Run tests of multiple modules, e.g. :mod:`obspy.io.wav` and
    :mod:`obspy.io.sac`::
        $ obspy-runtests io/wav obspy/io/sac
(5) Run a specific test case::
        $ obspy-runtests core/tests/test_stats.py::TestStats::test_init
(6) Create a self-contained html-report of test results saved as
   'obspy_report.html' in ObsPy's directory:
        $ obspy-runtests --report
(7) Run tests and print a coverage report to screen and save coverage.xml:
        $ obspy-runtests --coverage
(8) Save the image outputs of the testsuite, called 'obspy_image_tests':
        $ obspy-runtests --keep-images
"""
import os
import sys
from pathlib import Path

import obspy


def main():
    """
    Entry point for setup.py.
    Wrapper for a profiler if requested otherwise just call run() directly.
    If profiling is enabled we disable interactivity as it would wait for user
    input and influence the statistics. However the -r option still works.
    """
    try:
        import pytest
    except ImportError:
        msg = (
            "In order to run the ObsPy test suite, you must install the test "
            "requirements. This can be done with pip via pip install "
            "obspy[test] or pip install -e .[test] if installing in editable "
            "mode for development."
        )
        raise ImportError(msg)
    here = Path().cwd()
    base_obspy_path = Path(obspy.__file__).parent
    os.chdir(base_obspy_path)
    try:
        pytest.main()
    finally:
        os.chdir(here)


def run_tests(network=False,
              all=False,
              coverage=False,
              report=False,
              keep_images=False,):
    """
    Run ObsPy's test suite.

    :type all: bool
    :param all: Run all tests
    :type network: bool
    :param network: run only network tests.
    :type coverage: bool
    :param coverage: Calculate code coverage. Report to screen and recreate
        coverage report (coverage.xml) in current directory.
    :type report: bool
    :param report: Create a self-contained html report of test results
        in current directory called "obspy_report.html". This can be useful
        for sharing test results.
    :type keep_images: bool
    :param keep_images: If True, keep all images generated during testing in
        current directory called "obspy_test_images".
    """

    # append used functions to argv and run tests
    params = ['network', 'all', 'coverage', 'report', 'keep_images']
    locs = locals()
    for param in params:
        if getattr(locs, param):
            sys.argv.append(f'--{param}')
    main()


if __name__ == "__main__":
    main()
