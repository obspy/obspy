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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import numpy as np
from math import pi
from matplotlib import rcParams
from obspy import UTCDateTime
from obspy.signal.invsim import evalresp
from obspy.station import read_inventory
from obspy.xseed import Parser
from obspy.station.response import _pitick2latex
import os
import unittest
from obspy.core.util.testing import ImageComparison, getMatplotlibVersion
from obspy.core.util.misc import CatchOutput
import warnings


MATPLOTLIB_VERSION = getMatplotlibVersion()


class ResponseTestCase(unittest.TestCase):
    """
    Tests the for :class:`~obspy.station.response.Response` class.
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")
        self.image_dir = os.path.join(os.path.dirname(__file__), 'images')
        self.nperr = np.geterr()
        np.seterr(all='ignore')

    def tearDown(self):
        np.seterr(**self.nperr)

    def test_evalresp_with_output_from_seed(self):
        """
        The StationXML file has been converted to SEED with the help of a tool
        provided by IRIS:

        https://seiscode.iris.washington.edu/projects/stationxml-converter
        """
        t_samp = 0.05
        nfft = 16384

        # Test for different output units.
        units = ["DISP", "VEL", "ACC"]
        filenames = ["IRIS_single_channel_with_response", "XM.05", "AU.MEEK"]

        for filename in filenames:
            xml_filename = os.path.join(self.data_dir,
                                        filename + os.path.extsep + "xml")
            seed_filename = os.path.join(self.data_dir,
                                         filename + os.path.extsep + "seed")

            p = Parser(seed_filename)

            # older systems don't like an end date in the year 2599
            t_ = UTCDateTime(2030, 1, 1)
            if p.blockettes[50][0].end_effective_date > t_:
                p.blockettes[50][0].end_effective_date = None
            if p.blockettes[52][0].end_date > t_:
                p.blockettes[52][0].end_date = None

            resp_filename = p.getRESP()[0][-1]

            inv = read_inventory(xml_filename)

            network = inv[0].code
            station = inv[0][0].code
            location = inv[0][0][0].location_code
            channel = inv[0][0][0].code
            date = inv[0][0][0].start_date

            for unit in units:
                resp_filename.seek(0, 0)

                seed_response, seed_freq = evalresp(
                    t_samp, nfft, resp_filename, date=date, station=station,
                    channel=channel, network=network, locid=location,
                    units=unit, freq=True)

                xml_response, xml_freq = \
                    inv[0][0][0].response.get_evalresp_response(t_samp, nfft,
                                                                output=unit)

                self.assertTrue(np.allclose(seed_freq, xml_freq, rtol=1E-5))
                self.assertTrue(np.allclose(seed_response, xml_response,
                                            rtol=1E-5))

    def test_pitick2latex(self):
        self.assertEqual(_pitick2latex(3 * pi / 2), r'$\frac{3\pi}{2}$')
        self.assertEqual(_pitick2latex(2 * pi / 2), r'$\pi$')
        self.assertEqual(_pitick2latex(1 * pi / 2), r'$\frac{\pi}{2}$')
        self.assertEqual(_pitick2latex(0 * pi / 2), r'$0$')
        self.assertEqual(_pitick2latex(-1 * pi / 2), r'$-\frac{\pi}{2}$')
        self.assertEqual(_pitick2latex(-2 * pi / 2), r'$-\pi$')
        self.assertEqual(_pitick2latex(0.5), r'0.500')
        self.assertEqual(_pitick2latex(3 * pi + 0.01), r'9.43')
        self.assertEqual(_pitick2latex(30 * pi + 0.01), r'94.3')
        self.assertEqual(_pitick2latex(300 * pi + 0.01), r'942.')
        self.assertEqual(_pitick2latex(3000 * pi + 0.01), r'9.42e+03')

    def test_response_plot(self):
        """
        Tests the response plot.
        """
        # Bug in matplotlib 1.4.0 - 1.4.2:
        # See https://github.com/matplotlib/matplotlib/issues/4012
        reltol = 1.0
        if [1, 4, 0] <= MATPLOTLIB_VERSION <= [1, 4, 2]:
            reltol = 2.0

        resp = read_inventory()[0][0][0].response
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            with ImageComparison(self.image_dir, "response_response.png",
                                 reltol=reltol) as ic:
                rcParams['savefig.dpi'] = 72
                resp.plot(0.001, output="VEL", start_stage=1, end_stage=3,
                          outfile=ic.name)

    def test_segfault_after_error_handling(self):
        """
        Many functions in evalresp call `error_return()` which uses longjmp()
        to jump to some previously set state.

        ObsPy calls some evalresp functions directly so evalresp cannot call
        setjmp(). In that case longjmp() jumps to an undefined location, most
        likely resulting in a segfault.

        This test tests a workaround for this issue.

        As long as it does not segfault the test is doing alright.
        """
        filename = os.path.join(self.data_dir,
                                "TM.SKLT..BHZ_faulty_response.xml")
        inv = read_inventory(filename)

        t_samp = 0.05
        nfft = 256

        with CatchOutput():
            self.assertRaises(ValueError,
                              inv[0][0][0].response.get_evalresp_response,
                              t_samp, nfft, output="DISP")


def suite():
    return unittest.makeSuite(ResponseTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
