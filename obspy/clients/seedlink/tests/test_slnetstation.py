# -*- coding: utf-8 -*-
"""
The obspy.clients.seedlink.client.slnetstation test suite.
"""
import unittest

import pytest

from obspy.clients.seedlink.client.slnetstation import SLNetStation


pytestmark = pytest.mark.network


class SLNetStationTestCase(unittest.TestCase):

    def test_issue769(self):
        """
        Assure that different station objects don't share selector lists.
        """
        station1 = SLNetStation('', '', None, -1, None)
        station2 = SLNetStation('', '', None, -1, None)

        station1.append_selectors('FOO')

        self.assertNotEqual(id(station1.selectors), id(station2.selectors))
        self.assertEqual(station1.get_selectors(), ['FOO'])
        self.assertEqual(station2.get_selectors(), [])
