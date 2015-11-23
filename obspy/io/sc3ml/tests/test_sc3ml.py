#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the sc3ml reader.

Modified after obspy.io.stationXML 
    > obspy.obspy.io.stationxml.core.py

:author: 
    Mathijs Koymans (koymans@knmi.nl), 11.2015 - [Jollyfant@GitHub]

:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import fnmatch
import inspect
import io
import os
import re
import unittest

import obspy
from obspy.core.inventory import read_inventory, Inventory, Network

class sc3mlTestCase(unittest.TestCase):

    def setUp(self):
        self.stationxml_inventory = read_inventory("./data/NL_response_stationXML", format="STATIONXML")
        self.sc3ml_inventory = read_inventory("./data/NL_response_sc3ml", format="SC3ML")

    def test_compareXML(self):
        sc3ml_bytes = io.BytesIO()
        self.sc3ml_inventory.write(sc3ml_bytes, "STATIONXML")
        sc3ml_bytes.seek(0, 0)
        sc3ml_lines = sc3ml_bytes.read().decode().splitlines()
        sc3ml_array = [_i.strip() for _i in sc3ml_lines if _i.strip()]

        stationxml_bytes = io.BytesIO()
        self.stationxml_inventory.write(stationxml_bytes, "STATIONXML")
        stationxml_bytes.seek(0, 0)
        stationxml_lines = stationxml_bytes.read().decode().splitlines()
        stationxml_array = [_i.strip() for _i in stationxml_lines if _i.strip()]

        """ 
        The following tags can be different between sc3ml/stationXML

        <Source>SeisComP3</Source> | <Source>sc3ml import</Source>
        <Sender>ODC</Sender> | <Sender>ObsPy Inventory</Sender>
        <Created>2015-11-23T11:52:37+00:00</Created> | <Created>2015-11-23T12:04:25+00:00</Created>
        <Coefficients> | <Coefficients name="EBR.2002.091.H" resourceId="Datalogger#20121207153142.199696.15381">
        <Coefficients> | <Coefficients name="EBR.2002.091.H" resourceId="Datalogger#20121207153142.199696.15381">
        <Coefficients> | <Coefficients name="EBR.2002.091.H" resourceId="Datalogger#20121207153142.199696.15381">

        We disregard these differences because they are unimportant
        """ 

        excluded_tags = ["Source", "Sender", "Created", "Coefficients"]

        """
        Compare the two stationXMLs line by line
        If one XML has an entry that the other one does not, this procedure breaks
        e.g. an extra <type> tag will misalign lines to be compared
        Often the stationXML has a double sensor <type>/<model> tag that sc3ml lacks
        """
        for sc3ml, stationxml in zip(sc3ml_array, stationxml_array):
            if(sc3ml != stationxml):
                tag = str(stationxml).split(">")[0][1:]
                assert(tag in excluded_tags)

    def test_compareUpperLevel(self):
        """ Assert the top-level contents of the two dictionaries """
        stationxml_content = self.stationxml_inventory.get_contents()
        sc3ml_content = self.sc3ml_inventory.get_contents()
        for sc3ml, stationxml in zip(stationxml_content, sc3ml_content):
             self.assertEqual(sc3ml, stationxml)

    def test_compareResponse(self):
        """
        More self.assertEqualion checks
        """
        for sc3ml_net, stationxml_net in zip(self.sc3ml_inventory, self.stationxml_inventory):
            for sc3ml_sta, stationxml_sta in zip(sc3ml_net, stationxml_net):
                for sc3ml_cha, stationxml_cha in zip(sc3ml_sta, stationxml_sta):

                    self.assertEqual(sc3ml_cha.sample_rate, stationxml_cha.sample_rate)
                    self.assertEqual(sc3ml_cha.clock_drift_in_seconds_per_sample, stationxml_cha.clock_drift_in_seconds_per_sample)

                    self.assertEqual(sc3ml_cha.response.instrument_sensitivity.value, stationxml_cha.response.instrument_sensitivity.value)  
                    self.assertEqual(sc3ml_cha.response.instrument_sensitivity.frequency, stationxml_cha.response.instrument_sensitivity.frequency) 
                    self.assertEqual(sc3ml_cha.response.instrument_sensitivity.input_units, stationxml_cha.response.instrument_sensitivity.input_units) 

                    self.assertEqual(len(sc3ml_cha.response.response_stages), len(stationxml_cha.response.response_stages))
                    for sc3ml_stage, stationxml_stage in zip(sc3ml_cha.response.response_stages, stationxml_cha.response.response_stages):
                        self.assertEqual(sc3ml_stage.stage_gain, stationxml_stage.stage_gain)
            
                    """ Check poles / zeros """
                    sc3ml_paz = sc3ml_cha.response.get_paz()
                    stationxml_paz = stationxml_cha.response.get_paz()

                    self.assertEqual(sc3ml_paz.normalization_frequency, stationxml_paz.normalization_frequency)
                    self.assertEqual(sc3ml_paz.normalization_factor, stationxml_paz.normalization_factor)
                    self.assertEqual(sc3ml_paz.pz_transfer_function_type, stationxml_paz.pz_transfer_function_type)
                    for sc3ml_poles, stationxml_poles in zip(sc3ml_paz.poles, stationxml_paz.poles):
                        self.assertEqual(sc3ml_poles, stationxml_poles)
                    for sc3ml_zeros, stationxml_zeros, in zip(sc3ml_paz.zeros, stationxml_paz.zeros):
                        self.assertEqual(sc3ml_zeros, stationxml_zeros)

def suite():
    return unittest.makeSuite(sc3mlTestCase, "test")

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
