# -*- coding: utf-8 -*-
import numpy as np
import pytest

from obspy.core.inventory import (Response, PolesZerosResponseStage,
                                  ResponseStage, CoefficientsTypeResponseStage)
from obspy.clients.nrl.client import NRL, LocalNRL, RemoteNRL


@pytest.mark.network
class TestNRLRemote():
    """
    Minimal NRL test suite connecting to online NRL

    """
    def test_nrl_type(self):
        nrl_online = NRL(root='http://ds.iris.edu/NRL')
        assert isinstance(nrl_online, RemoteNRL)


class TestNRLLocal():
    """
    NRL test suite for test cases common to v1 and v2 of NRL without network
    usage.
    """
    def test_error_handling_invalid_path(self):
        msg = \
            "Provided path '/some/really/random/path' seems to be a local " \
            "file path but the directory does not exist."
        with pytest.raises(ValueError, match=msg):
            NRL("/some/really/random/path")


class TestNRLLocalV1():
    """
    NRL test suite using stripped down local NRL without network usage.

    """
    @pytest.fixture(autouse=True, scope="function")
    def setup(self, datapath):
        # Longer diffs in the test assertions.
        self.maxDiff = None
        # Small subset of NRL included in tests/data
        self.local_dl_key = ['REF TEK', 'RT 130 & 130-SMA', '1', '1']
        self.local_sensor_key = ['Guralp', 'CMG-3T', '120s - 50Hz', '1500']
        self.local_nrl_root = str(datapath / 'IRIS')
        self.nrl_local = NRL(root=self.local_nrl_root)

    def test_nrl_type(self):
        assert isinstance(self.nrl_local, LocalNRL)
        assert self.nrl_local._nrl_version == 1

    def test_get_response(self):
        # Get only the sensor response.
        sensor_resp = self.nrl_local.get_sensor_response(self.local_sensor_key)

        # Get only the datalogger response.
        dl_resp = self.nrl_local.get_datalogger_response(self.local_dl_key)

        # Get full response.
        resp = self.nrl_local.get_response(
            datalogger_keys=self.local_dl_key,
            sensor_keys=self.local_sensor_key)

        # Make sure that NRL.get_response() has overall instrument sensitivity
        # correctly recalculated after combining sensor and datalogger
        # information, see #3099.
        # Before fixing this bug the result was 945089653.7285056 which is a
        # relative deviation of 0.00104
        assert resp.instrument_sensitivity.value == pytest.approx(
            944098418.0614196, abs=0, rel=1e-4)

        # All of them should be Response objects.
        assert isinstance(resp, Response)
        assert isinstance(dl_resp, Response)
        assert isinstance(sensor_resp, Response)

        # The full response is the first stage from the sensor and all
        # following from the datalogger.
        assert resp.response_stages[0] == sensor_resp.response_stages[0]
        assert resp.response_stages[1:] == dl_resp.response_stages[1:]

        # Test the actual responses. Testing the parsing of the exact values
        # and what not is done in obspy.io.xseed.
        paz = sensor_resp.response_stages[0]
        assert isinstance(paz, PolesZerosResponseStage)
        np.testing.assert_allclose(
            paz.poles, [(-0.037008 + 0.037008j), (-0.037008 - 0.037008j),
                        (-502.65 + 0j), (-1005 + 0j), (-1131 + 0j)])
        np.testing.assert_allclose(paz.zeros, [0j, 0j])

        assert len(dl_resp.response_stages) == 15
        assert len(resp.response_stages) == 15

        assert isinstance(resp.response_stages[1], ResponseStage)
        for _i in range(2, 15):
            assert isinstance(resp.response_stages[_i],
                              CoefficientsTypeResponseStage)

    def test_nrl_class_str_method(self):
        out = str(self.nrl_local)
        # The local NRL is not going to chance so it is fine to test this.
        assert out.strip() == """
NRL library at %s
  Sensors: 20 manufacturers
    'CEA-DASE', 'CME', 'Chaparral Physics', 'Eentec', 'Generic',
    'Geo Space/OYO', 'Geodevice', 'Geotech', 'Guralp', 'Hyperion',
    'IESE', 'Kinemetrics', 'Lennartz', 'Metrozet', 'Nanometrics',
    'REF TEK', 'Sercel/Mark Products', 'SolGeo',
    'Sprengnether (now Eentec)', 'Streckeisen'
  Dataloggers: 13 manufacturers
    'Agecodagis', 'DAQ Systems (NetDAS)', 'Earth Data', 'Eentec',
    'Generic', 'Geodevice', 'Geotech', 'Guralp', 'Kinemetrics',
    'Nanometrics', 'Quanterra', 'REF TEK', 'SolGeo'
        """.strip() % self.local_nrl_root

    def test_nrl_dict_str_method(self):
        out = str(self.nrl_local.sensors)
        assert out.strip() == """
Select the sensor manufacturer (20 items):
  'CEA-DASE', 'CME', 'Chaparral Physics', 'Eentec', 'Generic',
  'Geo Space/OYO', 'Geodevice', 'Geotech', 'Guralp', 'Hyperion',
  'IESE', 'Kinemetrics', 'Lennartz', 'Metrozet', 'Nanometrics',
  'REF TEK', 'Sercel/Mark Products', 'SolGeo',
  'Sprengnether (now Eentec)', 'Streckeisen'""".strip()

    def test_error_handling_invalid_path(self):
        msg = \
            "Provided path '/some/really/random/path' seems to be a local " \
            "file path but the directory does not exist."
        with pytest.raises(ValueError, match=msg):
            NRL("/some/really/random/path")


class TestNRLLocalV2RESP():
    """
    NRL test suite using stripped down local NRL version 2 in RESP format
    without network usage.
    """
    @pytest.fixture(autouse=True, scope="function")
    def setup(self, datapath):
        # Longer diffs in the test assertions.
        self.maxDiff = None
        # Small subset of NRL included in tests/data
        self.local_dl_key = ['REFTEK', '130-01', '1', '1 Hz']
        self.local_sensor_key = ['Guralp', 'CMG-3T', '120 s', '50 Hz', '1500']
        self.local_nrl_root = str(datapath / 'IRIS_v2_resp')
        self.nrl_local = NRL(root=self.local_nrl_root)

    def test_nrl_type(self):
        assert isinstance(self.nrl_local, LocalNRL)
        assert self.nrl_local._nrl_version == 2

    def test_get_response(self):
        # Get only the sensor response.
        sensor_resp = self.nrl_local.get_sensor_response(self.local_sensor_key)

        # Get only the datalogger response.
        msg = r"First stage input units are 'COUNTS'\. When requesting a .*"
        with pytest.warns(UserWarning, match=msg):
            dl_resp = self.nrl_local.get_datalogger_response(self.local_dl_key)

        # Get full response.
        resp = self.nrl_local.get_response(
            datalogger_keys=self.local_dl_key,
            sensor_keys=self.local_sensor_key)

        # Make sure that NRL.get_response() has overall instrument sensitivity
        # correctly recalculated after combining sensor and datalogger
        # information, see #3099.
        # Before fixing this bug the result was 945089653.7285056 which is a
        # relative deviation of 0.00104
        assert resp.instrument_sensitivity.value == pytest.approx(
            945084144.2013303, abs=0, rel=1e-4)

        # All of them should be Response objects.
        assert isinstance(resp, Response)
        assert isinstance(dl_resp, Response)
        assert isinstance(sensor_resp, Response)

        # Changed in NRL v2, the full response is the all stages from the
        # sensor followed by all response stages from the datalogger, so there
        # is no dummy stage 1 anymore in the datalogger RESP files.
        # That means that the stage sequence numbers get changed during
        # assembling the full response and they have to be adjusted here
        # accordingly
        num_sensor_stages = len(sensor_resp.response_stages)
        # we also fix the missing units of first datalogger stage when a
        # combined sensor+datalogger response is requested, which doesn't work
        # when fetching the datalogger-only response, so we need to adjust
        # these units here for the test manually
        dl_resp.response_stages[0].input_units = 'V'
        dl_resp.response_stages[0].input_units_description = 'Volts'
        dl_resp.response_stages[0].output_units_description = 'Volts'

        for stage in dl_resp.response_stages:
            stage.stage_sequence_number += num_sensor_stages
        assert resp.response_stages[:num_sensor_stages] == \
            sensor_resp.response_stages[:]
        assert resp.response_stages[num_sensor_stages:] == \
            dl_resp.response_stages[:]

        # Test the actual responses. Testing the parsing of the exact values
        # and what not is done in obspy.io.xseed.
        paz = sensor_resp.response_stages[0]
        assert isinstance(paz, PolesZerosResponseStage)
        np.testing.assert_allclose(
            paz.poles, [(-0.037008 - 0.037008j), (-0.037008 + 0.037008j),
                        (-502.65 + 0j), (-1005 + 0j), (-1131 + 0j)])
        np.testing.assert_allclose(paz.zeros, [0j, 0j])

        assert len(dl_resp.response_stages) == 14
        assert len(resp.response_stages) == 15

        assert isinstance(resp.response_stages[1], ResponseStage)
        for _i in range(2, 15):
            assert isinstance(resp.response_stages[_i],
                              CoefficientsTypeResponseStage)

    def test_nrl_class_str_method(self):
        out = str(self.nrl_local)
        # The local NRL is not going to chance so it is fine to test this.
        assert out.strip() == """
NRL library at %s
  Sensors: 38 manufacturers
    'ASIR', 'CEADASE', 'Chaparral', 'DTCC', 'EQMet', 'Eentec',
    'GEObit', 'GaiaCode', 'GenericUnity', 'GeoDevice', 'GeoSIG',
    'GeoSpace', 'Geotech', 'Guralp', 'HGSProducts', 'HighTech',
    'Hyperion', 'IESE', 'JeffreyBJohnson', 'Kinemetrics', 'LaHusen',
    'Lennartz', 'Lunitek', 'MagseisFairfield', 'Nanometrics', 'REFTEK'
    'RSensors', 'RTClark', 'SARA', 'SeismoWave', 'SensorNederland',
    'Sercel', 'SiliconAudio', 'SolGeo', 'Sprengnether', 'Streckeisen'
    'Sunfull', 'iTem'
  Dataloggers: 28 manufacturers
    'Agecodagis', 'CNSN', 'DAQSystems', 'DTCC', 'DiGOSOmnirecs',
    'EQMet', 'EarthData', 'Eentec', 'GEObit', 'GenericUnity',
    'GeoDevice', 'GeoSIG', 'Geotech', 'Guralp', 'Kinemetrics',
    'Lunitek', 'MagseisFairfield', 'Nanometrics', 'Quanterra',
    'REFTEK', 'RSensors', 'SARA', 'STANEO', 'SeismicSource',
    'SeismologyResearchCentre', 'Sercel', 'SolGeo', 'WorldSensing'
        """.strip() % self.local_nrl_root

    def test_nrl_dict_str_method(self):
        out = str(self.nrl_local.sensors)
        assert out.strip() == """
Select the sensor manufacturer (38 items):
  'ASIR', 'CEADASE', 'Chaparral', 'DTCC', 'EQMet', 'Eentec', 'GEObit'
  'GaiaCode', 'GenericUnity', 'GeoDevice', 'GeoSIG', 'GeoSpace',
  'Geotech', 'Guralp', 'HGSProducts', 'HighTech', 'Hyperion', 'IESE',
  'JeffreyBJohnson', 'Kinemetrics', 'LaHusen', 'Lennartz', 'Lunitek',
  'MagseisFairfield', 'Nanometrics', 'REFTEK', 'RSensors', 'RTClark',
  'SARA', 'SeismoWave', 'SensorNederland', 'Sercel', 'SiliconAudio',
  'SolGeo', 'Sprengnether', 'Streckeisen', 'Sunfull', 'iTem'""".strip()


class TestNRLLocalV2StationXML():
    """
    NRL test suite using stripped down local NRL version 2 in StationXML format
    without network usage.
    """
    @pytest.fixture(autouse=True, scope="function")
    def setup(self, datapath):
        # Longer diffs in the test assertions.
        self.maxDiff = None
        # Small subset of NRL included in tests/data
        self.local_dl_key = ['REFTEK', '130-01', '1', '1 Hz']
        self.local_sensor_key = ['Guralp', 'CMG-3T', '120 s', '50 Hz', '1500']
        self.local_nrl_root = str(datapath / 'IRIS_v2_stationxml')
        self.nrl_local = NRL(root=self.local_nrl_root)

    def test_nrl_type(self):
        assert isinstance(self.nrl_local, LocalNRL)
        assert self.nrl_local._nrl_version == 2

    def test_get_response(self):
        # Get only the sensor response.
        sensor_resp = self.nrl_local.get_sensor_response(self.local_sensor_key)

        # Get only the datalogger response.
        msg = r"Undefined input units in stage one\. Most datalogger-only.*"
        with pytest.warns(UserWarning, match=msg):
            dl_resp = self.nrl_local.get_datalogger_response(self.local_dl_key)

        # Get full response.
        resp = self.nrl_local.get_response(
            datalogger_keys=self.local_dl_key,
            sensor_keys=self.local_sensor_key)

        # Make sure that NRL.get_response() has overall instrument sensitivity
        # correctly recalculated after combining sensor and datalogger
        # information, see #3099.
        # Before fixing this bug the result was 945089653.7285056 which is a
        # relative deviation of 0.00104
        assert resp.instrument_sensitivity.value == pytest.approx(
            945084144.2013303, abs=0, rel=1e-4)

        # All of them should be Response objects.
        assert isinstance(resp, Response)
        assert isinstance(dl_resp, Response)
        assert isinstance(sensor_resp, Response)

        # Changed in NRL v2, the full response is the all stages from the
        # sensor followed by all response stages from the datalogger, so there
        # is no dummy stage 1 anymore in the datalogger RESP files.
        # That means that the stage sequence numbers get changed during
        # assembling the full response and they have to be adjusted here
        # accordingly
        num_sensor_stages = len(sensor_resp.response_stages)
        for stage in dl_resp.response_stages:
            stage.stage_sequence_number += num_sensor_stages
        # we also fix the missing units of first datalogger stage when a
        # combined sensor+datalogger response is requested, which doesn't work
        # when fetching the datalogger-only response, so we need to adjust
        # these units here for the test manually
        dl_resp.response_stages[0].input_units = 'V'
        dl_resp.response_stages[0].input_units_description = 'Volts'
        # dl_resp.response_stages[0].output_units_description = 'Volts'

        assert resp.response_stages[:num_sensor_stages] == \
            sensor_resp.response_stages[:]
        assert resp.response_stages[num_sensor_stages:] == \
            dl_resp.response_stages[:]

        # Test the actual responses. Testing the parsing of the exact values
        # and what not is done in obspy.io.xseed.
        paz = sensor_resp.response_stages[0]
        assert isinstance(paz, PolesZerosResponseStage)
        np.testing.assert_allclose(
            paz.poles, [(-0.037008 - 0.037008j), (-0.037008 + 0.037008j),
                        (-502.65 + 0j), (-1005 + 0j), (-1131 + 0j)])
        np.testing.assert_allclose(paz.zeros, [0j, 0j])

        assert len(dl_resp.response_stages) == 14
        assert len(resp.response_stages) == 15

        assert isinstance(resp.response_stages[1], ResponseStage)
        for _i in range(2, 15):
            assert isinstance(resp.response_stages[_i],
                              CoefficientsTypeResponseStage)

    def test_nrl_class_str_method(self):
        out = str(self.nrl_local)
        # The local NRL is not going to chance so it is fine to test this.
        assert out.strip() == """
NRL library at %s
  Sensors: 38 manufacturers
    'ASIR', 'CEADASE', 'Chaparral', 'DTCC', 'EQMet', 'Eentec',
    'GEObit', 'GaiaCode', 'GenericUnity', 'GeoDevice', 'GeoSIG',
    'GeoSpace', 'Geotech', 'Guralp', 'HGSProducts', 'HighTech',
    'Hyperion', 'IESE', 'JeffreyBJohnson', 'Kinemetrics', 'LaHusen',
    'Lennartz', 'Lunitek', 'MagseisFairfield', 'Nanometrics', 'REFTEK'
    'RSensors', 'RTClark', 'SARA', 'SeismoWave', 'SensorNederland',
    'Sercel', 'SiliconAudio', 'SolGeo', 'Sprengnether', 'Streckeisen'
    'Sunfull', 'iTem'
  Dataloggers: 28 manufacturers
    'Agecodagis', 'CNSN', 'DAQSystems', 'DTCC', 'DiGOSOmnirecs',
    'EQMet', 'EarthData', 'Eentec', 'GEObit', 'GenericUnity',
    'GeoDevice', 'GeoSIG', 'Geotech', 'Guralp', 'Kinemetrics',
    'Lunitek', 'MagseisFairfield', 'Nanometrics', 'Quanterra',
    'REFTEK', 'RSensors', 'SARA', 'STANEO', 'SeismicSource',
    'SeismologyResearchCentre', 'Sercel', 'SolGeo', 'WorldSensing'
        """.strip() % self.local_nrl_root

    def test_nrl_dict_str_method(self):
        out = str(self.nrl_local.sensors)
        assert out.strip() == """
Select the sensor manufacturer (38 items):
  'ASIR', 'CEADASE', 'Chaparral', 'DTCC', 'EQMet', 'Eentec', 'GEObit'
  'GaiaCode', 'GenericUnity', 'GeoDevice', 'GeoSIG', 'GeoSpace',
  'Geotech', 'Guralp', 'HGSProducts', 'HighTech', 'Hyperion', 'IESE',
  'JeffreyBJohnson', 'Kinemetrics', 'LaHusen', 'Lennartz', 'Lunitek',
  'MagseisFairfield', 'Nanometrics', 'REFTEK', 'RSensors', 'RTClark',
  'SARA', 'SeismoWave', 'SensorNederland', 'Sercel', 'SiliconAudio',
  'SolGeo', 'Sprengnether', 'Streckeisen', 'Sunfull', 'iTem'""".strip()

    def test_get_integrated_response(self):
        # Get an integrated response
        resp = self.nrl_local.get_integrated_response(
            ['GeoSIG', 'ela-GMSseries', '0 Hz', '1000 Hz', '100 Hz', '1'])

        assert len(resp.response_stages) == 5
        assert resp.instrument_sensitivity.value == 778678
        assert resp.instrument_sensitivity.input_units == 'm/s**2'
        assert resp.instrument_sensitivity.output_units == 'counts'

    def test_get_soh_response(self):
        # Get an integrated response
        resp = self.nrl_local.get_soh_response(
            ['Quanterra', 'Q330PacketBaler', 'MassPosition', '10', '0.1 Hz'])

        assert len(resp.response_stages) == 1
        assert resp.instrument_sensitivity.value == 0.0
        assert resp.instrument_sensitivity.input_units == 'V'
        assert resp.instrument_sensitivity.output_units == 'counts'
