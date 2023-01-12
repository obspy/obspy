# -*- coding: utf-8 -*-
import collections
import io
import itertools
import os
import warnings

import numpy as np

from obspy.core.inventory.response import PolynomialResponseStage
from obspy.core.util.base import CatchAndAssertWarnings, NamedTemporaryFile
from obspy.io.xseed import Parser, InvalidResponseError
from obspy.io.xseed.core import _is_resp, _is_xseed, _is_seed, _read_resp, \
    _read_seed, _read_xseed
import obspy.io.xseed.parser
from obspy.signal.invsim import evalresp_for_frequencies
import pytest


class TestCore():
    """
    Test integration with ObsPy's inventory objects.
    """
    @pytest.fixture(autouse=True, scope="function")
    def setup(self, testdata, root):
        self.seed_files = [
            "AI.ESPZ._.BHE.dataless",
            "AI.ESPZ._.BH_.dataless",
            "BN.LPW._.BHE.dataless",
            "CL.AIO.dataless",
            "G.SPB.dataless",
            "arclink_full.seed",
            "bug165.dataless",
            "dataless.seed.BW_DHFO",
            "dataless.seed.BW_FURT",
            "dataless.seed.BW_MANZ",
            "dataless.seed.BW_RJOB",
            "dataless.seed.BW_ROTZ",
            "dataless.seed.BW_ZUGS",
            "dataless.seed.II_COCO"
        ]
        self.xseed_files = ["dataless.seed.BW_FURT.xml"]

        self.resp_files = ["RESP.BW.FURT..EHZ",
                           "RESP.XX.NR008..HHZ.130.1.100",
                           "RESP.XX.NS085..BHZ.STS2_gen3.120.1500",
                           "RESP.BK.BRIB..BV1",
                           "RESP.BK.DANT.00.LCL",
                           # Has only a stage 0 with blkts 53 and 58.
                           "RESP.BN.WR0..SHZ",
                           # Defines the input units only in stage 2.
                           "RESP.SG.MEMB..BDI"
                           ]
        self.other_files = ["II_COCO_three_channel_borehole.mseed",
                            "xml-seed-1.0.xsd",
                            "xml-seed-1.1.xsd"]

        self.seed_files = [testdata[_i] for _i in self.seed_files]
        self.xseed_files = [testdata[_i] for _i in self.xseed_files]
        self.resp_files = [testdata[_i] for _i in self.resp_files]
        self.other_files = [testdata[_i] for _i in self.other_files]

        # There are a couple more SEED files in the core test suite.
        core_data = root / 'core' / 'tests' / 'data'
        self.seed_files.append(core_data / "AU.MEEK.seed")
        self.seed_files.append(
            core_data / "IRIS_single_channel_with_response.seed")

        for _i in itertools.chain.from_iterable([
                self.seed_files, self.xseed_files, self.resp_files,
                self.other_files]):
            assert os.path.exists(_i), _i

    def test_is_seed(self):
        for filename in self.seed_files:
            assert _is_seed(filename), filename

        for filename in self.xseed_files:
            assert not _is_seed(filename), filename

        for filename in self.resp_files:
            assert not _is_seed(filename), filename

        for filename in self.other_files:
            assert not _is_seed(filename), filename

    def test_is_xseed(self):
        for filename in self.seed_files:
            assert not _is_xseed(filename), filename

        for filename in self.xseed_files:
            assert _is_xseed(filename), filename

        for filename in self.resp_files:
            assert not _is_xseed(filename), filename

        for filename in self.other_files:
            assert not _is_xseed(filename), filename

    def test_is_resp(self):
        for filename in self.seed_files:
            assert not _is_resp(filename), filename

        for filename in self.xseed_files:
            assert not _is_resp(filename), filename

        for filename in self.resp_files:
            assert _is_resp(filename), filename

        for filename in self.other_files:
            assert not _is_resp(filename), filename

    def test_simple_read_resp(self):
        """
        Currently just tests that all test RESP files can be read without an
        error.
        """
        for f in self.resp_files:
            _read_resp(f)

    def test_simple_read_seed(self):
        """
        Currently just tests that all test SEED files can be read without an
        error.
        """
        # One seed file is a bit faulty and thus raises a warning when read
        # - catch it.
        with CatchAndAssertWarnings():
            for f in self.seed_files:
                _read_seed(f)

    def test_simple_read_xseed(self):
        """
        Currently just tests that all test X(SEED) files can be read without
        an error.
        """
        for f in self.xseed_files:
            _read_xseed(f)

    def test_read_resp_metadata(self, testdata):
        """
        Manually assert that the meta-data is read correctly for all the
        RESP files.
        """
        # File A
        filename_a = testdata["RESP.BW.FURT..EHZ"]
        inv = obspy.read_inventory(filename_a)
        assert len(inv) == 1
        assert len(inv[0]) == 1
        assert len(inv[0][0]) == 1

        network = inv[0]
        station = inv[0][0]
        channel = inv[0][0][0]
        resp = inv[0][0][0].response

        assert network.code == "BW"
        assert station.code == "FURT"
        assert station.start_date is None
        assert station.end_date is None
        assert channel.location_code == ""
        assert channel.code == "EHZ"
        assert channel.start_date == obspy.UTCDateTime(2001, 1, 1)
        assert channel.end_date is None
        # Also check the input and output units.
        assert resp.instrument_sensitivity.input_units == "M/S"
        assert resp.instrument_sensitivity.input_units_description == \
            "Velocity in Meters per Second"
        assert resp.instrument_sensitivity.output_units == "COUNTS"
        assert resp.instrument_sensitivity.output_units_description == \
            "Digital Counts"

        # File B
        filename_b = testdata["RESP.XX.NR008..HHZ.130.1.100"]
        inv = obspy.read_inventory(filename_b)
        assert len(inv) == 1
        assert len(inv[0]) == 1
        assert len(inv[0][0]) == 1

        network = inv[0]
        station = inv[0][0]
        channel = inv[0][0][0]
        resp = inv[0][0][0].response

        assert network.code == "XX"
        assert station.code == "NR008"
        assert station.start_date is None
        assert station.end_date is None
        assert channel.location_code == ""
        assert channel.code == "HHZ"
        assert channel.start_date == obspy.UTCDateTime(2006, 1, 1)
        assert channel.end_date == obspy.UTCDateTime(3000, 1, 1) - 1
        # Also check the input and output units.
        assert resp.instrument_sensitivity.input_units == "M/S"
        assert resp.instrument_sensitivity.input_units_description == \
            "Velocity in Meters per Second"
        assert resp.instrument_sensitivity.output_units == "COUNTS"
        assert resp.instrument_sensitivity.output_units_description == \
            "Digital Counts"

        # File C
        filename_c = testdata["RESP.XX.NS085..BHZ.STS2_gen3.120.1500"]
        inv = obspy.read_inventory(filename_c)
        assert len(inv) == 1
        assert len(inv[0]) == 1
        assert len(inv[0][0]) == 1

        network = inv[0]
        station = inv[0][0]
        channel = inv[0][0][0]
        resp = inv[0][0][0].response

        assert network.code == "XX"
        assert station.code == "NS085"
        assert station.start_date is None
        assert station.end_date is None
        assert channel.location_code == ""
        assert channel.code == "BHZ"
        assert channel.start_date == obspy.UTCDateTime(2006, 1, 1)
        assert channel.end_date is None
        # Also check the input and output units.
        assert resp.instrument_sensitivity.input_units == "M/S"
        assert resp.instrument_sensitivity.input_units_description == \
            "Velocity in Meters per Second"
        assert resp.instrument_sensitivity.output_units == "COUNTS"
        assert resp.instrument_sensitivity.output_units_description == \
            "Digital Counts"

        # Merge A + B to get a multi-response file.
        with NamedTemporaryFile() as tf:
            with io.open(filename_a, "rb") as fh:
                tf.write(fh.read())
            tf.write(b"\n")
            with io.open(filename_b, "rb") as fh:
                tf.write(fh.read())
            tf.seek(0, 0)
            inv = obspy.read_inventory(tf.name)
        assert len(inv) == 2
        assert len(inv[0]) == 1
        assert len(inv[0][0]) == 1
        assert len(inv[1]) == 1
        assert len(inv[1][0]) == 1

        network = inv[0]
        station = inv[0][0]
        channel = inv[0][0][0]
        resp = inv[0][0][0].response

        assert network.code == "BW"
        assert station.code == "FURT"
        assert station.start_date is None
        assert station.end_date is None
        assert channel.location_code == ""
        assert channel.code == "EHZ"
        assert channel.start_date == obspy.UTCDateTime(2001, 1, 1)
        assert channel.end_date is None
        # Also check the input and output units.
        assert resp.instrument_sensitivity.input_units == "M/S"
        assert resp.instrument_sensitivity.input_units_description == \
            "Velocity in Meters per Second"
        assert resp.instrument_sensitivity.output_units == "COUNTS"
        assert resp.instrument_sensitivity.output_units_description == \
            "Digital Counts"

        network = inv[1]
        station = inv[1][0]
        channel = inv[1][0][0]
        resp = inv[1][0][0].response

        assert network.code == "XX"
        assert station.code == "NR008"
        assert station.start_date is None
        assert station.end_date is None
        assert channel.location_code == ""
        assert channel.code == "HHZ"
        assert channel.start_date == obspy.UTCDateTime(2006, 1, 1)
        assert channel.end_date == obspy.UTCDateTime(3000, 1, 1) - 1
        # Also check the input and output units.
        assert resp.instrument_sensitivity.input_units == "M/S"
        assert resp.instrument_sensitivity.input_units_description == \
            "Velocity in Meters per Second"
        assert resp.instrument_sensitivity.output_units == "COUNTS"
        assert resp.instrument_sensitivity.output_units_description == \
            "Digital Counts"

    def test_read_seed_metainformation(self, testdata):
        """
        Test the mapping of meta-information for SEED files. This will be
        exactly identical for XML-SEED files so no need to test these as well.
        """
        filename = testdata["dataless.seed.BW_ROTZ"]
        inv = obspy.read_inventory(filename)

        assert len(inv) == 1
        assert len(inv[0]) == 1
        assert len(inv[0][0]) == 3

        network = inv[0]
        assert network.code == "BW"
        assert network.description == "BayernNetz"

        station = inv[0][0]
        assert station.code == "ROTZ"
        assert round(abs(station.latitude-49.766899), 7) == 0
        assert round(abs(station.longitude-12.207), 7) == 0
        assert round(abs(station.elevation-430.0), 7) == 0
        assert station.site.name == "Rotzenmuhle,Bavaria, BW-Net"
        assert station.start_date == \
            obspy.UTCDateTime("2006-06-04T00:00:00.000000Z")
        assert station.end_date is None

        # First channel.
        channel = inv[0][0][0]
        assert channel.code == "EHZ"
        assert channel.location_code == ""
        assert round(abs(channel.latitude-49.766899), 7) == 0
        assert round(abs(channel.longitude-12.207), 7) == 0
        assert round(abs(channel.elevation-430.0), 7) == 0
        assert round(abs(channel.depth-0.0), 7) == 0
        assert channel.azimuth == 0.0
        assert channel.dip == -90.0
        assert channel.start_date == \
            obspy.UTCDateTime("2006-06-04T00:00:00.000000Z")
        assert channel.end_date is None
        assert channel.sample_rate == 200.0
        assert channel.sensor.type == "Streckeisen STS-2/N seismometer"
        resp = channel.response
        assert resp.instrument_sensitivity.input_units == "M/S"
        assert resp.instrument_sensitivity.input_units_description == \
            "Velocity in Meters per Second"
        assert resp.instrument_sensitivity.output_units == "COUNTS"
        assert resp.instrument_sensitivity.output_units_description == \
            "Digital Counts"

        # Second channel.
        channel = inv[0][0][1]
        assert channel.code == "EHN"
        assert channel.location_code == ""
        assert round(abs(channel.latitude-49.766899), 7) == 0
        assert round(abs(channel.longitude-12.207), 7) == 0
        assert round(abs(channel.elevation-430.0), 7) == 0
        assert round(abs(channel.depth-0.0), 7) == 0
        assert channel.azimuth == 0.0
        assert channel.dip == 0.0
        assert channel.start_date == \
            obspy.UTCDateTime("2006-06-04T00:00:00.000000Z")
        assert channel.end_date is None
        assert channel.sample_rate == 200.0
        assert channel.sensor.type == \
            "Streckeisen STS-2/N seismometer"
        resp = channel.response
        assert resp.instrument_sensitivity.input_units == "M/S"
        assert resp.instrument_sensitivity.input_units_description == \
            "Velocity in Meters per Second"
        assert resp.instrument_sensitivity.output_units == "COUNTS"
        assert resp.instrument_sensitivity.output_units_description == \
            "Digital Counts"

        # Third channel.
        channel = inv[0][0][2]
        assert channel.code == "EHE"
        assert channel.location_code == ""
        assert round(abs(channel.latitude-49.766899), 7) == 0
        assert round(abs(channel.longitude-12.207), 7) == 0
        assert round(abs(channel.elevation-430.0), 7) == 0
        assert round(abs(channel.depth-0.0), 7) == 0
        assert channel.azimuth == 90.0
        assert channel.dip == 0.0
        assert channel.start_date == \
            obspy.UTCDateTime("2006-06-04T00:00:00.000000Z")
        assert channel.end_date is None
        assert channel.sample_rate == 200.0
        assert channel.sensor.type == \
            "Streckeisen STS-2/N seismometer"
        resp = channel.response
        assert resp.instrument_sensitivity.input_units == "M/S"
        assert resp.instrument_sensitivity.input_units_description == \
            "Velocity in Meters per Second"
        assert resp.instrument_sensitivity.output_units == "COUNTS"
        assert resp.instrument_sensitivity.output_units_description == \
            "Digital Counts"

    def test_response_calculation_from_resp_files(self):
        """
        Test the response calculations with the obspy.core interface.

        Compares with directly calling evalresp.
        """
        # Very broad range but the responses should be exactly identical as
        # they use the same code under the hood so it should prove no issue.
        frequencies = np.logspace(-3, 3, 20)

        for filename in self.resp_files:
            # Set the times for the response.
            t = obspy.UTCDateTime(2008, 1, 1)
            if "AZ.DHL..BS1" in str(filename):
                t = obspy.UTCDateTime(1999, julday=351)
            elif "BK.DANT.00.LCL" in str(filename):
                t = obspy.UTCDateTime(2017, 1, 1)
            elif "BN.WR0..SHZ" in str(filename):
                t = obspy.UTCDateTime(1998, 1, 1)

            for unit in ("DISP", "VEL", "ACC"):
                r = obspy.read_inventory(filename)[0][0][0].response
                e_r = evalresp_for_frequencies(
                    t_samp=None, frequencies=frequencies, filename=filename,
                    date=t, units=unit)
                i_r = r.get_evalresp_response_for_frequencies(
                    frequencies=frequencies, output=unit)
                # This is in general very dangerous for floating point numbers
                # but they use exactly the same code under the hood here so it
                # is okay - if we ever have our own response calculation code
                # this will have to be changed.
                np.testing.assert_equal(e_r, i_r, "%s - %s" % (filename, unit))

    def test_response_calculation_from_seed_and_xseed(self):
        """
        Test the response calculations with the obspy.core interface.

        It does it by converting whatever it gets to RESP files and then
        uses evalresp to get the response. This is compared to using the
        ObsPy Response object - this also uses evalresp but the actual flow
        of the data is very different.

        This is an expensive test but worth it for the trust it builds and
        the bugs it found and prevents.
        """
        # Very broad range but the responses should be exactly identical as
        # they use the same code under the hood so it should prove no issue.
        frequencies = np.logspace(-3, 3, 20)

        for filename in self.seed_files + self.xseed_files:
            # Parse the files using the Parser object.
            with CatchAndAssertWarnings():
                p = Parser(filename)
                p_resp = {_i[0]: _i[1] for _i in p.get_resp()}
                # Also read using the core routines.
                inv = obspy.read_inventory(filename)

            # Get all the channels and epochs.
            channels = collections.defaultdict(list)
            for c in p.get_inventory()["channels"]:
                channels[c["channel_id"]].append(
                    (c["start_date"], c["end_date"]))

            # Loop over each.
            for channel, epochs in channels.items():
                with NamedTemporaryFile() as tf:
                    r = p_resp["RESP.%s" % channel]
                    r.seek(0, 0)
                    tf.write(r.read())

                    # Now loop over the epochs.
                    for start, end in epochs:
                        if end:
                            t = start + (end - start) / 2
                        else:
                            t = start + 10

                        # Find response
                        n, s, l, c = channel.split(".")
                        _inv_t = inv.select(network=n, station=s,
                                            location=l, channel=c,
                                            starttime=t - 1, endtime=t + 1)
                        # Should now only be a single channel.
                        assert _inv_t.get_contents()["channels"] == [channel]
                        inv_r = _inv_t[0][0][0].response

                        for unit in ("DISP", "VEL", "ACC"):
                            # Directly call evalresp.
                            e_r = evalresp_for_frequencies(
                                t_samp=None, frequencies=frequencies,
                                filename=tf.name, date=t, units=unit)
                            i_r = inv_r.get_evalresp_response_for_frequencies(
                                frequencies=frequencies, output=unit)
                            # Adaptive absolute tolerance to deal with very
                            # small values.
                            atol = 1E-7 * max(np.abs(e_r).max(),
                                              np.abs(i_r).max())
                            np.testing.assert_allclose(
                                e_r.real, i_r.real,
                                err_msg="real - %s - %s" % (filename, unit),
                                rtol=1E-6, atol=atol)
                            np.testing.assert_allclose(
                                e_r.imag, i_r.imag,
                                err_msg="imag - %s - %s" % (filename, unit),
                                rtol=1E-6, atol=atol)

                            # Bonus: Also read the RESP file directly with
                            # obspy.core and test the response.
                            i_r_r = obspy.read_inventory(tf.name).select(
                                starttime=t - 1,
                                endtime=t + 1)[0][0][0].response\
                                .get_evalresp_response_for_frequencies(
                                frequencies=frequencies, output=unit)
                            np.testing.assert_allclose(
                                e_r.real, i_r_r.real,
                                err_msg="RESP real - %s - %s" % (filename,
                                                                 unit),
                                rtol=1E-6, atol=atol)
                            np.testing.assert_allclose(
                                e_r.imag, i_r_r.imag,
                                err_msg="RESP imag - %s - %s" % (filename,
                                                                 unit),
                                rtol=1E-6, atol=atol)

    def test_warning_when_blockette_54_is_not_followed_by_57(self, testdata):
        filename = testdata["RESP.SG.ST..LDO"]
        # Fail if responses are explicitly not skipped.
        msg = \
            "Stage 2: Invalid response specification. A blockette 54 " \
            "must always be followed by a blockette 57 which is missing."
        with CatchAndAssertWarnings():
            with pytest.raises(InvalidResponseError, match=msg):
                obspy.read_inventory(filename, skip_invalid_responses=False)

        # Delete warning registry to reliably trigger the warning.
        if hasattr(obspy.io.xseed.parser, "__warningregistry__"):
            obspy.io.xseed.parser.__warningregistry__.clear()

        # Otherwise continue, but raise a warning.
        msg1 = (
            r"Epoch SG.ST..LDO "
            r"\[1997-02-01T00:00:00.000000Z - 2599-12-31T23:59:59.000000Z\]: "
            r"Response stage 2 does not end with blockette 58. Proceed at "
            r"your own risk.")
        msg2 = (
            r"Failed to calculate response for SG.ST..LDO with epoch "
            r"1997-02-01T00:00:00.000000Z - 2599-12-31T23:59:59.000000Z "
            r"because: Stage 2: Invalid response specification. A blockette "
            r"54 must always be followed by a blockette 57 which is missing.")
        # This actually raises two warnings - one that blockette 58 is
        # missing on stage 2 - this is a generic error that can be
        # potentially recovered from - then a second one that it actually
        # cannot recover.
        with CatchAndAssertWarnings(
                expected=[(Warning, msg1), (Warning, msg2)]):
            obspy.read_inventory(filename)

    def test_warning_when_blockette_57_is_not_followed_by_58(self, testdata):
        filename = testdata["RESP.decimation_without_gain"]
        # Fail if responses are explicitly not skipped.
        msg = \
            "Stage 1: A decimation stage with blockette 57 must be followed " \
            "by a blockette 58 which is missing here."
        with CatchAndAssertWarnings():
            with pytest.raises(InvalidResponseError, match=msg):
                obspy.read_inventory(filename, skip_invalid_responses=False)
        # Otherwise continue, but raise a warning.
        msg = ("Failed to calculate response for XX.ABC..BHX with epoch "
               "1999-12-16T02:14:00.000000Z - 1999-12-21T19:10:59.000000Z "
               "because: Stage 1: A decimation stage with blockette 57 must "
               "be followed by a blockette 58 which is missing here.")
        # This triggers a number of warnings as the file is constructed and
        # misses all kinds of information.
        with CatchAndAssertWarnings(expected=[(Warning, msg)]):
            obspy.read_inventory(filename)

    def test_warning_with_multiple_blockettes_58_in_stage_0(self, testdata):
        filename = testdata["RESP.repeated_stage_0"]
        msg = (r"Epoch BN.WR0..SHZ "
               r"\[1996-03-01T00:00:00.000000Z - "
               r"1999-01-03T00:00:00.000000Z\]: "
               r"Channel has multiple \(but identical\) blockettes 58 for "
               r"stage 0. Only one will be used.")
        with CatchAndAssertWarnings(expected=[(Warning, msg)]):
            obspy.read_inventory(filename)

    def test_warning_with_multiple_differing_blockettes_58_in_stage_0(
            self, testdata):
        filename = testdata["RESP.repeated_differing_stage_0"]
        msg = (r"Epoch BN.WR0..SHZ "
               r"\[1996-03-01T00:00:00.000000Z - "
               r"1999-01-03T00:00:00.000000Z\]: "
               r"Channel has multiple different blockettes 58 for stage 0. "
               r"The last one will be chosen - this is a faulty file - try to "
               r"fix it!")
        with CatchAndAssertWarnings(expected=[(Warning, msg)]):
            obspy.read_inventory(filename)

    def test_blkts_53_and_54_in_one_stage(self, testdata):
        """
        This should naturally raise.
        """
        filename = testdata["RESP.blkt53_and_54_in_one_stage"]
        msg = \
            "Stage 1 has both, blockette 53 and 54. This is not valid."
        with pytest.raises(InvalidResponseError, match=msg):
            obspy.read_inventory(filename, skip_invalid_responses=False)
        # If invalid responses are skipped, check the warning.
        msg = (
            r"Failed to calculate response for BN.WR0..SHZ with epoch "
            r"1996-03-01T00:00:00.000000Z - 1999-01-03T00:00:00.000000Z "
            r"because: Stage 1 has both, blockette 53 and 54. "
            r"This is not valid.")
        with CatchAndAssertWarnings(expected=[(Warning, msg)]):
            inv = obspy.read_inventory(filename)
        assert inv[0][0][0].response is None

    def test_reconstructing_stage_0_from_other_blockettes(self, testdata):
        # This file has no stage 0 but a bunch of other blockettes 58 from
        # other stages. Try to reconstruct stage 0.
        filename = testdata["RESP.JM.NMIA0.00.HHN"]

        frequencies = np.logspace(-3, 3, 100)
        t = obspy.UTCDateTime(2015, 1, 1)

        # Should raise no warnings.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inv = obspy.read_inventory(filename)
        assert len(w) == 0

        assert inv.get_contents()["channels"] == ["JM.NMIA0.00.HNN"]

        # Also check the responses via the inventory objects and by directly
        # calling evalresp.
        for unit in ("DISP", "VEL", "ACC"):
            e_r = evalresp_for_frequencies(
                t_samp=None, frequencies=frequencies, filename=filename,
                date=t, units=unit)
            i_r = inv[0][0][0].response.get_evalresp_response_for_frequencies(
                frequencies=frequencies, output=unit)
            np.testing.assert_equal(e_r, i_r)

    def test_parsing_blockette_62(self, testdata):
        filename = testdata["RESP.blockette_62"]
        inv = obspy.read_inventory(filename)
        assert inv.get_contents()["channels"] == ["XH.DR01.30.LDO"]
        r = inv[0][0][0].response
        assert isinstance(r.response_stages[0], PolynomialResponseStage)
        p = r.response_stages[0]
        assert p == PolynomialResponseStage(
            stage_sequence_number=1,
            stage_gain=1.0,
            stage_gain_frequency=0.002,
            input_units="PA",
            output_units="COUNTS",
            output_units_description="Digital Counts",
            frequency_lower_bound=0.001,
            frequency_upper_bound=10.0,
            approximation_lower_bound=600,
            approximation_upper_bound=1100,
            maximum_error=0.0,
            coefficients=[8e2, 1.5e-4]
        )

    def test_parsing_blockette_62_as_stage_0(self, testdata):
        filename = testdata["RESP.blockette_62_as_stage_0"]
        # Make sure there are no warnings.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inv = obspy.read_inventory(filename)
        assert len(w) == 0
        assert inv.get_contents()["channels"] == ["UO.BEER.EP.BDO"]
        r = inv[0][0][0].response
        # Should have no sensitivity.
        assert r.instrument_sensitivity is None
        # For now just test that it is actually read.
        assert isinstance(r.response_stages[0], PolynomialResponseStage)

    def test_warning_with_no_blockettes_58(self, testdata):
        filename = testdata["RESP.repeated_stage_0"]
        msg = (r"Epoch BN.WR0..SHZ "
               r"\[1996-03-01T00:00:00.000000Z - "
               r"1999-01-03T00:00:00.000000Z\]: "
               r"Channel has multiple \(but identical\) blockettes 58 for "
               r"stage 0. Only one will be used.")
        with CatchAndAssertWarnings(expected=[(Warning, msg)]):
            obspy.read_inventory(filename)

    def test_paz_with_no_actual_values(self, testdata):
        filename = testdata["RESP.paz_with_no_values"]
        frequencies = np.logspace(-3, 3, 20)

        # Set the times for the response.
        t = obspy.UTCDateTime(2008, 1, 1)

        for unit in ("DISP", "VEL", "ACC"):
            r = obspy.read_inventory(filename)[0][0][0].response
            e_r = evalresp_for_frequencies(
                t_samp=None, frequencies=frequencies, filename=filename,
                date=t, units=unit)
            i_r = r.get_evalresp_response_for_frequencies(
                frequencies=frequencies, output=unit)
            np.testing.assert_equal(e_r, i_r, "%s - %s" % (filename, unit))

    def test_response_of_strain_meter(self, testdata):
        filename = testdata["RESP.strain_meter"]
        frequencies = np.logspace(-3, 3, 20)

        # Set the times for the response.
        t = obspy.UTCDateTime(2012, 1, 1)

        for unit in ("DISP", "VEL", "ACC"):
            r = obspy.read_inventory(filename)[0][0][0].response
            e_r = evalresp_for_frequencies(
                t_samp=None, frequencies=frequencies, filename=filename,
                date=t, units=unit)
            i_r = r.get_evalresp_response_for_frequencies(
                frequencies=frequencies, output=unit)
            np.testing.assert_equal(e_r, i_r, "%s - %s" % (filename, unit))

    def test_response_multiple_gain_blockettes(self, testdata):
        """
        Evalresp chooses the last one - make sure we do the same.
        """
        filename = testdata["RESP.multiple_gain_blockettes"]
        frequencies = np.logspace(-3, 3, 20)

        # Set the times for the response.
        t = obspy.UTCDateTime(1996, 1, 1)

        for unit in ("DISP", "VEL", "ACC"):
            # This raises a warning that it has multiple gain blockettes.
            with CatchAndAssertWarnings():
                r = obspy.read_inventory(filename)[0][0][0].response
            e_r = evalresp_for_frequencies(
                t_samp=None, frequencies=frequencies, filename=filename,
                date=t, units=unit)
            i_r = r.get_evalresp_response_for_frequencies(
                frequencies=frequencies, output=unit)
            np.testing.assert_equal(e_r, i_r, "%s - %s" % (filename, unit))

    def test_response_regression_1(self, testdata):
        """
        Regression test as fixing one issue broke something else.
        """
        filename = testdata["RESP.regression_1"]
        frequencies = np.logspace(-3, 3, 20)

        # Set the times for the response.
        t = obspy.UTCDateTime(2010, 1, 1)

        for unit in ("DISP", "VEL", "ACC"):
            r = obspy.read_inventory(filename)[0][0][0].response
            e_r = evalresp_for_frequencies(
                t_samp=None, frequencies=frequencies, filename=filename,
                date=t, units=unit)
            i_r = r.get_evalresp_response_for_frequencies(
                frequencies=frequencies, output=unit)
            np.testing.assert_equal(e_r, i_r, "%s - %s" % (filename, unit))

    def test_response_regression_2(self, testdata):
        """
        Another regression test.
        """
        filename = testdata["RESP.regression_2"]
        frequencies = np.logspace(-3, 3, 20)

        # Set the times for the response.
        t = obspy.UTCDateTime(2013, 1, 1)

        for unit in ("DISP", "VEL", "ACC"):
            e_r = evalresp_for_frequencies(
                t_samp=None, frequencies=frequencies, filename=filename,
                date=t, units=unit)
            r = obspy.read_inventory(filename)[0][0][0].response
            i_r = r.get_evalresp_response_for_frequencies(
                frequencies=frequencies, output=unit)
            np.testing.assert_equal(e_r, i_r, "%s - %s" % (filename, unit))

    def test_response_regression_segfault(self, testdata):
        """
        Another regression test.
        """
        filename = testdata["RESP.regression_segfault"]
        frequencies = np.logspace(-3, 3, 20)

        # Set the times for the response.
        # t = obspy.UTCDateTime(2009, 1, 23)

        for unit in ("DISP", "VEL", "ACC"):
            # This still sometimes triggers a segfault - but not if called
            # via our internal evalresp mapping via obspy.core.inventory.
            # e_r = evalresp_for_frequencies(
            #     t_samp=None, frequencies=frequencies, filename=filename,
            #     date=t, units=unit)

            with CatchAndAssertWarnings():
                r = obspy.read_inventory(filename)[0][0][0].response
            r.get_evalresp_response_for_frequencies(
                frequencies=frequencies, output=unit)

    def test_response_lots_zero_frequency_gains(self, testdata):
        """
        Test a RESP file with many zero frequency gains in file.
        """
        filename = testdata["RESP.many_zero_frequencies"]
        frequencies = np.logspace(-3, 3, 20)

        # Set the times for the response.
        t = obspy.UTCDateTime(2005, 1, 1)

        for unit in ("DISP", "VEL", "ACC"):
            e_r = evalresp_for_frequencies(
                t_samp=None, frequencies=frequencies, filename=filename,
                date=t, units=unit)
            r = obspy.read_inventory(filename)[0][0][0].response
            i_r = r.get_evalresp_response_for_frequencies(
                frequencies=frequencies, output=unit)
            np.testing.assert_equal(e_r, i_r, "%s - %s" % (filename, unit))
