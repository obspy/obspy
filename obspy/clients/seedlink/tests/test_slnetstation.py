# -*- coding: utf-8 -*-
"""
The obspy.clients.seedlink.client.slnetstation test suite.
"""

import pytest

from obspy.clients.seedlink.client.slnetstation import SLNetStation


pytestmark = pytest.mark.network


class TestSLNetStation():

    def test_issue769(self):
        """
        Assure that different station objects don't share selector lists.
        """
        station1 = SLNetStation('', '', None, -1, None)
        station2 = SLNetStation('', '', None, -1, None)

        station1.append_selectors('FOO')

        assert id(station1.selectors) != id(station2.selectors)
        assert station1.get_selectors() == ['FOO']
        assert station2.get_selectors() == []
