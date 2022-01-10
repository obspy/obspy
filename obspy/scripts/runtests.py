#!/usr/bin/env python
# -*- coding: utf-8 -*-
#:copyright:
#    The ObsPy Development Team (devs@obspy.org)
#:license:
#    GNU Lesser General Public License, Version 3
#    (https://www.gnu.org/copyleft/lesser.html)
"""
A command-line program that runs all ObsPy tests.

All tests in ObsPy are located in the tests directory of each specific
module.

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
(6) Create a self-contained html-report of test results with pytest-html
    plugin::
        $ obspy-runtests --html path/report.html
(7) Run tests and print a coverage report to screen and save coverage.xml
    with pytest-cov plugin:
        $ obspy-runtests --coverage
(8) Save the image outputs of the testsuite, called 'obspy_image_tests':
        $ obspy-runtests --keep-images
(9) Run the test suite, drop into a pdb debugging session for each failure:
        $ obspy-runtests --pdb
"""
import os
import sys
from pathlib import Path

from pytest_jsonreport.plugin import JSONReport
import pytest
import requests
import obspy

# URL to upload json report
REPORT_URL = "tests.obspy.org"


def main():
    """
    Entry point for setup.py.
    Wrapper for a profiler if requested otherwise just call run() directly.
    If profiling is enabled we disable interactivity as it would wait for user
    input and influence the statistics. However the -r option still works.
    """
    if '-h' in sys.argv or '--help' in sys.argv:
        print(__doc__)
    here = Path().cwd()
    base_obspy_path = Path(obspy.__file__).parent
    plugin = JSONReport()
    os.chdir(base_obspy_path)
    if all(['--json-report-file' not in arg for arg in sys.argv]):
        sys.argv.append('--json-report-file=none')
    try:
        pytest.main(plugins=[plugin])
    finally:
        os.chdir(here)
    report = (True if '--report' in sys.argv else
              False if '--no-report' in sys.argv else None)
    upload_json_report(report=report, data=plugin.report)


def upload_json_report(report=None, data=None):
    """Upload the json report to ObsPy test server."""
    if report is None:
        msg = f"Do you want to report this to {REPORT_URL} ? [n]: "
        answer = input(msg).lower()
        report = 'y' in answer
    if report:
        response = requests.post(f"https://{REPORT_URL}/post/v2/", json=data)
        # get the response
        if response.status_code == 200:
            report_url = response.json().get('url', REPORT_URL)
            print('Your test results have been reported and are available at: '
                  '{}\nThank you!'.format(report_url))
        # handle errors
        else:
            print(f"Error: Could not sent a test report to {REPORT_URL}.")
            print(response.reason)


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
