#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the scml reader inventory.

Modified after obspy.io.stationXML
    > obspy.obspy.io.stationxml.core.py

:author:
    Mathijs Koymans (koymans@knmi.nl), 11.2015 - [Jollyfant@GitHub]

:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import io
import re
import warnings

import pytest
from obspy.core.inventory import read_inventory
from obspy.core.inventory.response import (CoefficientsTypeResponseStage,
                                           FIRResponseStage)
from obspy.io.seiscomp.inventory import (
    _count_complex, _parse_list_of_complex_string)


class TestSCML():

    @pytest.fixture(autouse=True, scope="function")
    def setup(self, testdata):
        self.stationxml_path = testdata["EB_response_stationXML"]
        self.scml_path = testdata["EB_response_sc3ml"]
        self.stationxml_inventory = read_inventory(self.stationxml_path,
                                                   format="STATIONXML")
        self.scml_inventory = read_inventory(self.scml_path, format="SCML")
        self.USE_NAMESPACE = "http://geofon.gfz-potsdam.de/ns/seiscomp"

    def test_scml_versions(self, testdata):
        """
        Test multiple schema versions
        """
        for version in ['0.5', '0.6', '0.99']:
            filename = testdata['version%s' % version]

            msg = "Schema version not supported."
            with pytest.raises(ValueError, match=msg):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    read_inventory(filename)

    @pytest.mark.filterwarnings('ignore:.*rate of 0')
    def test_channel_level(self, testdata):
        """
        Test inventory without repsonse information up to
        channel level
        """
        inv = read_inventory(testdata["channel_level.sc3ml"])
        assert inv[0].code == "NL"
        assert inv[0][0].code == "HGN"
        for cha in inv[0][0].channels:
            assert cha.code in ["BHE", "BHN", "BHZ"]

    def test_compare_xml(self):
        """
        Easiest way to compare is to write both Inventories back
        to stationXML format and compare line by line
        """
        scml_bytes = io.BytesIO()
        self.scml_inventory.write(scml_bytes, "STATIONXML")
        scml_bytes.seek(0, 0)
        scml_lines = scml_bytes.read().decode().splitlines()
        scml_arr = [_i.strip() for _i in scml_lines if _i.strip()]

        stationxml_bytes = io.BytesIO()
        self.stationxml_inventory.write(stationxml_bytes, "STATIONXML")
        stationxml_bytes.seek(0, 0)
        stationxml_lines = stationxml_bytes.read().decode().splitlines()
        stationxml_arr = [_i.strip() for _i in stationxml_lines if _i.strip()]

        # The following tags can be different between scml/stationXML

        # <Source>SeisComP3</Source> | <Source>scml import</Source>
        # <Sender>ODC</Sender> | <Sender>ObsPy Inventory</Sender>
        # <Created>2015-11-23T11:52:37+00:00</Created>
        # <Coefficients> | <Coefficients name="EBR.2002.091.H" ~

        # We disregard these differences because they are unimportant
        excluded_tags = ["Source", "Sender", "Created", "Name",
                         "Coefficients"]

        # also ignore StorageFormat which doesnt exist anymore in
        # StationXML 1.1 and is saved into extra / a foreign tag
        pattern_format_line = (
            r'<([^:]*):format xmlns:\1='
            r'"http://geofon.gfz-potsdam.de/ns/seiscomp3?-schema/[\.0-9]*">')
        scml_arr = [line for line in scml_arr
                    if not re.search(pattern_format_line, line)]

        # Compare the two stationXMLs line by line
        # If one XML has an entry that the other one does not, this procedure
        # breaks e.g. an extra <type> tag will misalign lines to be compared
        # Often the stationXML has a double sensor <type>/<model> tag that
        # scml lacks
        for scml, stationxml in zip(scml_arr, stationxml_arr):
            if scml != stationxml:
                tag = str(stationxml).split(">")[0][1:]
                assert tag in excluded_tags

    def test_empty_depth(self, testdata):
        """
        Assert depth, latitude, longitude, elevation set to 0.0 if left empty
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("default")
            read_inventory(testdata["sc3ml_empty_depth_and_id.sc3ml"])
            assert len(w) == 4
            assert str(w[0].message) == \
                "Sensor is missing longitude information, using 0.0"
            assert str(w[1].message) == \
                "Sensor is missing latitude information, using 0.0"
            assert str(w[2].message) == \
                "Sensor is missing elevation information, using 0.0"
            assert str(w[3].message) == \
                "Channel is missing depth information, using 0.0"

    def test_compare_upper_level(self):
        """
        Assert the top-level contents of the two dictionaries
        Networks, channels, stations, locations
        """
        stationxml_content = self.stationxml_inventory.get_contents()
        scml_content = self.scml_inventory.get_contents()
        for scml, stationxml in zip(stationxml_content, scml_content):
            assert scml == stationxml

    def test_compare_response(self):
        """
        More assertions in the actual response info
        """
        for scml_net, stationxml_net in zip(self.scml_inventory,
                                            self.stationxml_inventory):

            assert scml_net.code == stationxml_net.code
            assert scml_net.description == stationxml_net.description
            assert scml_net.start_date == stationxml_net.start_date
            assert scml_net.end_date == stationxml_net.end_date
            assert scml_net.restricted_status == \
                stationxml_net.restricted_status

            for scml_sta, stationxml_sta in zip(scml_net, stationxml_net):

                assert scml_sta.latitude == stationxml_sta.latitude
                assert scml_sta.longitude == stationxml_sta.longitude
                assert scml_sta.elevation == stationxml_sta.elevation
                assert scml_sta.creation_date == stationxml_sta.creation_date
                assert scml_sta.termination_date == \
                    stationxml_sta.termination_date

                staxml_items = stationxml_sta.site.__dict__.items()
                scml_items = scml_sta.site.__dict__.items()
                for scml_site, stationxml_site in zip(staxml_items,
                                                      scml_items):
                    assert scml_site == stationxml_site

                for scml_cha, stationxml_cha in zip(scml_sta,
                                                    stationxml_sta):

                    assert scml_cha.code == stationxml_cha.code
                    assert scml_cha.latitude == stationxml_cha.latitude
                    assert scml_cha.longitude == stationxml_cha.longitude
                    assert scml_cha.elevation == stationxml_cha.elevation
                    assert scml_cha.azimuth == stationxml_cha.azimuth
                    assert scml_cha.dip == stationxml_cha.dip
                    # reading stationxml will ignore old StationXML 1.0 defined
                    # StorageFormat, Arclink Inventory XML and SCML get it
                    # stored in extra now
                    with pytest.warns(UserWarning, match='.*storage_format.*'):
                        assert scml_cha.storage_format is None
                        assert stationxml_cha.storage_format is None
                    assert scml_cha.extra['format']['value'] == 'Steim2'
                    namespace = scml_cha.extra['format'].get('namespace')
                    assert namespace.startswith(self.USE_NAMESPACE)

                    cdisps = "clock_drift_in_seconds_per_sample"
                    assert getattr(scml_cha, cdisps) == \
                        getattr(stationxml_cha, cdisps)

                    for scml, stationxml in zip(stationxml_cha.data_logger.
                                                __dict__.items(),
                                                scml_cha.data_logger.
                                                __dict__.items()):
                        assert scml == stationxml
                    for scml, stationxml in zip(stationxml_cha.sensor.
                                                __dict__.items(),
                                                scml_cha.sensor.
                                                __dict__.items()):
                        assert scml == stationxml

                    assert scml_cha.sample_rate == stationxml_cha.sample_rate

                    scml_ins = scml_cha.response.instrument_sensitivity
                    stationxml_ins = scml_cha.response.instrument_sensitivity

                    assert scml_ins.value == stationxml_ins.value
                    assert scml_ins.frequency == stationxml_ins.frequency
                    assert scml_ins.input_units == stationxml_ins.input_units
                    assert len(scml_cha.response.response_stages) == \
                        len(stationxml_cha.response.response_stages)

                    for scml, stationxml in zip(scml_cha.response.
                                                response_stages,
                                                stationxml_cha.response.
                                                response_stages):
                        assert scml.stage_gain == stationxml.stage_gain
                        assert scml.stage_sequence_number == \
                            stationxml.stage_sequence_number

                        # We skip checking this stage, because the input
                        # sample rates may not match
                        # StationXML gives a sample rate of 10e-310 (0) for
                        # some channels while this should be the sample rate
                        # after stage 1 (never 0)
                        if isinstance(scml, CoefficientsTypeResponseStage):
                            continue

                        if isinstance(scml, FIRResponseStage):
                            assert scml.__dict__ == \
                                             stationxml.__dict__

                    """ Check poles / zeros """
                    scml_paz = scml_cha.response.get_paz()
                    stationxml_paz = stationxml_cha.response.get_paz()

                    assert scml_paz.normalization_frequency == \
                        stationxml_paz.normalization_frequency
                    assert scml_paz.normalization_factor == \
                        stationxml_paz.normalization_factor
                    assert scml_paz.pz_transfer_function_type == \
                        stationxml_paz.pz_transfer_function_type
                    for scml, stationxml in zip(scml_paz.poles,
                                                stationxml_paz.poles):
                        assert scml == stationxml
                    for scml, stationxml in zip(scml_paz.zeros,
                                                stationxml_paz.zeros):
                        assert scml == stationxml

    def test_parse_complex_list(self):
        """
        Tests parsing list of complex numbers from seiscomp3 xml.
        """
        complex_string = ("  (   -0.037 ,     0.037 )  (-0.037,-0.037)"
                          "(-6909,     9208)( -6909  ,-9208)  ")
        assert _count_complex(complex_string) == 4
        parsed = _parse_list_of_complex_string(complex_string)
        assert parsed == [('-0.037', '0.037'), ('-0.037', '-0.037'),
                          ('-6909', '9208'), ('-6909', '-9208')]
        # test some bad string
        complex_string = "  (   -0.037 ,     0.037 )  (-0.037,-0.037"
        with pytest.raises(ValueError):
            _count_complex(complex_string)
        with pytest.raises(ValueError):
            _parse_list_of_complex_string(complex_string)

    def test_stage_empty_poles_and_zeros(self, testdata):
        """
        Tests for a case where the poles and zeros are empty see #2633
        """
        scml_mbar_path = testdata["zero_poles_and_zeros.sc3ml"]
        scml_inv = read_inventory(scml_mbar_path)
        response = scml_inv[0][0][0].response
        zeros = response.response_stages[1].zeros
        poles = response.response_stages[1].poles
        assert zeros == []
        assert poles == []
