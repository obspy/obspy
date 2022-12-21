# -*- coding: utf-8 -*-
"""
The obspy.clients.seedlink.client.slstate test suite.
"""

import pytest

from obspy.clients.seedlink.client.slstate import SLState


pytestmark = pytest.mark.network


class TestSLState():

    def test_issue561(self):
        """
        Assure that different state objects don't share data buffers.
        """
        slstate1 = SLState()
        slstate2 = SLState()

        assert id(slstate1.databuf) != id(slstate2.databuf)
        assert id(slstate1.packed_buf) != id(slstate2.packed_buf)
