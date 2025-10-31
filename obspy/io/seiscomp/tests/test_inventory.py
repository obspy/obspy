#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the sc3ml reader inventory.

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
    _count_complex, _parse_list_of_complex_string, SCHEMA_NAMESPACE_BASE)


class TestSC3ML():

    @pytest.fixture(autouse=True, scope="function")
    def setup(self, testdata):
        self.stationxml_path = testdata["EB_response_stationXML"]
        self.sc3ml_path = testdata["EB_response_sc3ml"]
        self.stationxml_inventory = read_inventory(self.stationxml_path,
                                                   format="STATIONXML")
        self.sc3ml_inventory = read_inventory(self.sc3ml_path, format="SC3ML")

    def test_sc3ml_versions(self, testdata):
        """
        Test multiple schema versions
        """
        for version in ['0.5', '0.99']:
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
        sc3ml_bytes = io.BytesIO()
        self.sc3ml_inventory.write(sc3ml_bytes, "STATIONXML")
        sc3ml_bytes.seek(0, 0)
        sc3ml_lines = sc3ml_bytes.read().decode().splitlines()
        sc3ml_arr = [_i.strip() for _i in sc3ml_lines if _i.strip()]

        stationxml_bytes = io.BytesIO()
        self.stationxml_inventory.write(stationxml_bytes, "STATIONXML")
        stationxml_bytes.seek(0, 0)
        stationxml_lines = stationxml_bytes.read().decode().splitlines()
        stationxml_arr = [_i.strip() for _i in stationxml_lines if _i.strip()]

        # The following tags can be different between sc3ml/stationXML

        # <Source>SeisComP3</Source> | <Source>sc3ml import</Source>
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
            r'"http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/[\.0-9]*">')
        sc3ml_arr = [line for line in sc3ml_arr
                     if not re.search(pattern_format_line, line)]

        # Compare the two stationXMLs line by line
        # If one XML has an entry that the other one does not, this procedure
        # breaks e.g. an extra <type> tag will misalign lines to be compared
        # Often the stationXML has a double sensor <type>/<model> tag that
        # sc3ml lacks
        for sc3ml, stationxml in zip(sc3ml_arr, stationxml_arr):
            if sc3ml != stationxml:
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
        sc3ml_content = self.sc3ml_inventory.get_contents()
        for sc3ml, stationxml in zip(stationxml_content, sc3ml_content):
            assert sc3ml == stationxml

    def test_compare_response(self):
        """
        More assertions in the actual response info
        """
        for sc3ml_net, stationxml_net in zip(self.sc3ml_inventory,
                                             self.stationxml_inventory):

            assert sc3ml_net.code == stationxml_net.code
            assert sc3ml_net.description == stationxml_net.description
            assert sc3ml_net.start_date == stationxml_net.start_date
            assert sc3ml_net.end_date == stationxml_net.end_date
            assert sc3ml_net.restricted_status == \
                stationxml_net.restricted_status

            for sc3ml_sta, stationxml_sta in zip(sc3ml_net, stationxml_net):

                assert sc3ml_sta.latitude == stationxml_sta.latitude
                assert sc3ml_sta.longitude == stationxml_sta.longitude
                assert sc3ml_sta.elevation == stationxml_sta.elevation
                assert sc3ml_sta.creation_date == stationxml_sta.creation_date
                assert sc3ml_sta.termination_date == \
                    stationxml_sta.termination_date

                staxml_items = stationxml_sta.site.__dict__.items()
                sc3ml_items = sc3ml_sta.site.__dict__.items()
                for sc3ml_site, stationxml_site in zip(staxml_items,
                                                       sc3ml_items):
                    assert sc3ml_site == stationxml_site

                for sc3ml_cha, stationxml_cha in zip(sc3ml_sta,
                                                     stationxml_sta):

                    assert sc3ml_cha.code == stationxml_cha.code
                    assert sc3ml_cha.latitude == stationxml_cha.latitude
                    assert sc3ml_cha.longitude == stationxml_cha.longitude
                    assert sc3ml_cha.elevation == stationxml_cha.elevation
                    assert sc3ml_cha.azimuth == stationxml_cha.azimuth
                    assert sc3ml_cha.dip == stationxml_cha.dip
                    # reading stationxml will ignore old StationXML 1.0 defined
                    # StorageFormat, Arclink Inventory XML and SC3ML get it
                    # stored in extra now
                    with pytest.warns(UserWarning, match='.*storage_format.*'):
                        assert sc3ml_cha.storage_format is None
                        assert stationxml_cha.storage_format is None
                    assert sc3ml_cha.extra['format']['value'] == 'Steim2'
                    namespace = sc3ml_cha.extra['format'].get('namespace')
                    assert namespace.startswith(SCHEMA_NAMESPACE_BASE)

                    cdisps = "clock_drift_in_seconds_per_sample"
                    assert getattr(sc3ml_cha, cdisps) == \
                        getattr(stationxml_cha, cdisps)

                    for sc3ml, stationxml in zip(stationxml_cha.data_logger.
                                                 __dict__.items(),
                                                 sc3ml_cha.data_logger.
                                                 __dict__.items()):
                        assert sc3ml == stationxml
                    for sc3ml, stationxml in zip(stationxml_cha.sensor.
                                                 __dict__.items(),
                                                 sc3ml_cha.sensor.
                                                 __dict__.items()):
                        assert sc3ml == stationxml

                    assert sc3ml_cha.sample_rate == stationxml_cha.sample_rate

                    sc3ml_ins = sc3ml_cha.response.instrument_sensitivity
                    stationxml_ins = sc3ml_cha.response.instrument_sensitivity

                    assert sc3ml_ins.value == stationxml_ins.value
                    assert sc3ml_ins.frequency == stationxml_ins.frequency
                    assert sc3ml_ins.input_units == stationxml_ins.input_units
                    assert len(sc3ml_cha.response.response_stages) == \
                        len(stationxml_cha.response.response_stages)

                    for sc3ml, stationxml in zip(sc3ml_cha.response.
                                                 response_stages,
                                                 stationxml_cha.response.
                                                 response_stages):
                        assert sc3ml.stage_gain == stationxml.stage_gain
                        assert sc3ml.stage_sequence_number == \
                            stationxml.stage_sequence_number

                        # We skip checking this stage, because the input
                        # sample rates may not match
                        # StationXML gives a sample rate of 10e-310 (0) for
                        # some channels while this should be the sample rate
                        # after stage 1 (never 0)
                        if isinstance(sc3ml, CoefficientsTypeResponseStage):
                            continue

                        if isinstance(sc3ml, FIRResponseStage):
                            assert sc3ml.__dict__ == \
                                             stationxml.__dict__

                    """ Check poles / zeros """
                    sc3ml_paz = sc3ml_cha.response.get_paz()
                    stationxml_paz = stationxml_cha.response.get_paz()

                    assert sc3ml_paz.normalization_frequency == \
                        stationxml_paz.normalization_frequency
                    assert sc3ml_paz.normalization_factor == \
                        stationxml_paz.normalization_factor
                    assert sc3ml_paz.pz_transfer_function_type == \
                        stationxml_paz.pz_transfer_function_type
                    for sc3ml, stationxml in zip(sc3ml_paz.poles,
                                                 stationxml_paz.poles):
                        assert sc3ml == stationxml
                    for sc3ml, stationxml in zip(sc3ml_paz.zeros,
                                                 stationxml_paz.zeros):
                        assert sc3ml == stationxml

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
        sc3ml_mbar_path = testdata["zero_poles_and_zeros.sc3ml"]
        sc3ml_inv = read_inventory(sc3ml_mbar_path)
        response = sc3ml_inv[0][0][0].response
        zeros = response.response_stages[1].zeros
        poles = response.response_stages[1].poles
        assert zeros == []
        assert poles == []
