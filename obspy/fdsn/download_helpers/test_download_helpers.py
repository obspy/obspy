#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.fdsn.client test suite.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import unicode_literals
from future import standard_library  # NOQA
import unittest


class DownloadHelpersUtilTestCase(unittest.TestCase):
    """
    Test cases for utility functionality for the download helpers.
    """
    def test_something(self):
        """
        Does not yet do anything.
        """
        self.assertTrue(True)


def suite():
    return unittest.makeSuite(DownloadHelpersUtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
