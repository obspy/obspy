#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.fdsn.download_helpers test suite.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import unicode_literals
from future import standard_library  # NOQA
import unittest

from obspy.fdsn.download_helpers.utils import filter_availability
from obspy.core.compatibility import mock

from obspy.fdsn.download_helpers.download_helpers import \
    _filter_channel_priority

from obspy.fdsn.download_helpers import domain


class DomainTestCase(unittest.TestCase):
    """
    Test case for the domain definitions.
    """
    def test_rectangular_domain(self):
        """
        Test the rectangular domain.
        """
        dom = domain.RectangularDomain(-10, 10, -20, 20)
        query_params = dom.get_query_parameters()
        self.assertEqual(query_params, {
            "minlatitude": -10,
            "maxlatitude": 10,
            "minlongitude": -20,
            "maxlongitude": 20})

        self.assertRaises(NotImplementedError, dom.is_in_domain, 0, 0)

    def test_circular_domain(self):
        """
        Test the circular domain.
        """
        dom = domain.CircularDomain(10, 20, 30, 40)
        query_params = dom.get_query_parameters()
        self.assertEqual(query_params, {
            "latitude": 10,
            "longitude": 20,
            "minradius": 30,
            "maxradius": 40})

        self.assertRaises(NotImplementedError, dom.is_in_domain, 0, 0)

    def test_global_domain(self):
        """
        Test the global domain.
        """
        dom = domain.GlobalDomain()
        query_params = dom.get_query_parameters()
        self.assertEqual(query_params, {})

        self.assertRaises(NotImplementedError, dom.is_in_domain, 0, 0)

    def test_subclassing_without_abstract_method(self):
        """
        Subclassing without implementing the get_query_parameters method
        results in a TypeError upon instantiation time.
        """
        class NewDom(domain.Domain):
            pass

        self.assertRaises(TypeError, NewDom)

    def test_instantiating_root_domain_object_fails(self):
        """
        Trying to create a root domain object should fail.
        """
        self.assertRaises(TypeError, domain.Domain)


class DownloadHelpersUtilTestCase(unittest.TestCase):
    """
    Test cases for utility functionality for the download helpers.
    """
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

        # Different ways to not filter.
        filtered_channels = _filter_channel_priority(channels, priorities=[
            "*"])
        self.assertEqual(filtered_channels, ["BHE", "SHE", "BHZ", "HHE"])

        filtered_channels = _filter_channel_priority(channels,
                                                     priorities=None)
        self.assertEqual(filtered_channels, ["BHE", "SHE", "BHZ", "HHE"])



def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(DomainTestCase, 'test'))
    suite.addTest(unittest.makeSuite(DownloadHelpersUtilTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
