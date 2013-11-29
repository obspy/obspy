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
from obspy.station import read_inventory
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
        from obspy.xseed import Parser
        from obspy.signal.invsim import evalresp
        import matplotlib.pylab as plt

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

        plt.subplot(211)
        plt.semilogx(seed_freq, np.abs(seed_response), label="SEED")
        plt.subplot(212)
        phase = np.unwrap(np.arctan2(-seed_response.imag, seed_response.real))
        plt.semilogx(seed_freq, phase)

        inv = read_inventory(os.path.join(
            self.data_dir, "IRIS_single_channel_with_response.xml"))
        xml_response, xml_freq = \
            inv[0][0][0].response.get_evalresp_response(t_samp, nfft)

        plt.subplot(211)
        plt.semilogx(xml_freq, np.abs(xml_response), label="XML")

        plt.legend(loc="lower right")
        plt.subplot(212)
        phase = np.unwrap(np.arctan2(-xml_response.imag, xml_response.real))
        plt.semilogx(xml_freq, phase)
        plt.show()

    def test_some_output(self):
        """
        Some simple sanity tests.
        """
        inv = read_inventory(os.path.join(
            self.data_dir, "IRIS_single_channel_with_response.xml"))
        output, freqs = inv[0][0][0].response.get_evalresp_response(
            0.005, 16384)

       #is_output = output[::len(output)/20]

        #should_output = np.array([
            #0.00000000+0.j,  0.99940241-0.19369858j,
            #0.93868406-0.38609437j,  0.83377693-0.56438594j,
            #0.68342237-0.71523076j,  0.49396289-0.82029753j,
            #0.28367566-0.86312959j,  0.08032223-0.83887866j,
            #-0.08939465-0.75933843j, -0.21078977-0.64764319j,
            #-0.28382352-0.5273433j, -0.31782256-0.41482287j,
            #-0.32471096-0.31805747j, -0.31488869-0.23898002j,
            #-0.29592758-0.17631394j, -0.27277897-0.12755105j,
            #-0.24844576-0.09001069j, -0.22463134-0.06129346j,
            #-0.20222597-0.03941633j, -0.18163282-0.02280549j,
            #-0.16297413-0.01023814j])

        #np.testing.assert_array_almost_equal(is_output, should_output)

        if False:
            import matplotlib.pylab as plt

            plt.subplot(211)
            plt.semilogx(freqs, np.abs(output))
            plt.subplot(212)
            phase = np.unwrap(np.arctan2(-output.imag, output.real))
            plt.semilogx(freqs, phase)
            plt.show()


def suite():
    return unittest.makeSuite(ResponseTest, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
