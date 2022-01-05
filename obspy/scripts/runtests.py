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

"""
import os
from pathlib import Path

import obspy


def run_tests():
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
    pytest.main()
    # try:
    #     pytest.main()
    # finally:
    os.chdir(here)


if __name__ == "__main__":
    run_tests()
