#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the response handling.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import inspect
import obspy
from obspy.signal.invsim import evalresp
from obspy.station import read_inventory
from obspy.xseed import Parser
import os
import unittest


import numpy as np


class ResponseTest(unittest.TestCase):
    """
    Tests the for :class:`~obspy.station.inventory.Inventory` class.
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_evalresp_with_output_from_seed(self):
        """
        The StationXML file has been converted to SEED with the help of a tool
        provided by IRIS:

        https://seiscode.iris.washington.edu/projects/stationxml-converter
        """
        t_samp = 0.05
        nfft = 16384
        date = obspy.UTCDateTime(2013, 1, 1)
        network = "IU"
        station = "ANMO"
        locid = "10"
        channel = "BHZ"
        units = "VEL"

        seed_file = os.path.join(self.data_dir,
                                 "IRIS_single_channel_with_response.seed")
        p = Parser(seed_file)

        filename = p.getRESP()[0][-1]
        filename.seek(0, 0)

        seed_response, seed_freq = evalresp(t_samp, nfft, filename, date=date,
                                            station=station, channel=channel,
                                            network=network, locid=locid,
                                            units=units, freq=True)

        inv = read_inventory(os.path.join(
            self.data_dir, "IRIS_single_channel_with_response.xml"))
        xml_response, xml_freq = \
            inv[0][0][0].response.get_evalresp_response(t_samp, nfft)

        self.assertTrue(np.allclose(seed_freq, xml_freq, rtol=1E-5))
        self.assertTrue(np.allclose(seed_response, xml_response,
                                    rtol=1E-5))


def suite():
    return unittest.makeSuite(ResponseTest, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
