# -*- coding: utf-8 -*-

import unittest
from matplotlib.tests.test_coding_standards import assert_pep8_conformance, \
    HAS_PEP8
import obspy
from obspy.core.util.base import getMatplotlibVersion
from obspy.core.util.decorator import skipIf

MATPLOTLIB_VERSION = getMatplotlibVersion()

EXCLUDE_FILES = []
EXPECTED_BAD_FILES = [
    "*/obspy/lib/__init__.py",
    ]
PEP8_ADDITONAL_IGNORE = []

if HAS_PEP8:
    from matplotlib.tests.test_coding_standards import \
        StandardReportWithExclusions
    StandardReportWithExclusions.expected_bad_files = EXPECTED_BAD_FILES


class Pep8TestCase(unittest.TestCase):
    """
    Test codebase for Pep8 compliance.
    """
    @skipIf(MATPLOTLIB_VERSION < [1, 4, 0], "matplotlib >= 1.4 is required")
    def test_pep8(self):
        """
        Test codebase for Pep8 compliance.
        """
        assert_pep8_conformance(module=obspy, exclude_files=EXCLUDE_FILES,
                                extra_exclude_file=None,
                                pep8_additional_ignore=PEP8_ADDITONAL_IGNORE)


def suite():
    return unittest.makeSuite(Pep8TestCase, 'test')


if __name__ == '__main__':
    if not HAS_PEP8:
        raise unittest.SkipTest('pep8 is required for this test suite')
    unittest.main(defaultTest='suite')
