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


from obspy.fdsn.download_helpers.download_helpers import \
    _filter_channel_priority


class DownloadHelpersUtilTestCase(unittest.TestCase):
    """
    Test cases for utility functionality for the download helpers.
    """
    def test_something(self):
        """
        Does not yet do anything.
        """
        self.assertTrue(True)

    def test_channel_priority_filtering(self):
        """
        Tests the channel priority filtering.
        """
        channels = ["BHE", "SHE", "BHZ", "HHE"]

        filtered_channels = _filter_channel_priority(channels, priorities=[
            "HH[Z,N,E]", "BH[Z,N,E]", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, ["HHE"])

        filtered_channels = _filter_channel_priority(channels, priorities=[
            "BH[Z,N,E]", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, ["BHE", "BHZ"])

        filtered_channels = _filter_channel_priority(channels, priorities=[
            "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, [])

        filtered_channels = _filter_channel_priority(channels, priorities=[
            "*"])
        self.assertEqual(filtered_channels, channels)

        filtered_channels = _filter_channel_priority(channels, priorities=[
            "BH*", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, ["BHE", "BHZ"])

        filtered_channels = _filter_channel_priority(channels, priorities=[
            "BH[N,Z]", "MH[Z,N,E]", "EH[Z,N,E]", "LH[Z,N,E]"])
        self.assertEqual(filtered_channels, ["BHZ"])

        filtered_channels = _filter_channel_priority(channels, priorities=[
            "S*", "BH*"])
        self.assertEqual(filtered_channels, ["SHE"])


def suite():
    return unittest.makeSuite(DownloadHelpersUtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
