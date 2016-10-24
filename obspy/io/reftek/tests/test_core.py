#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import inspect
import os
import unittest

import numpy as np

import obspy
from obspy.core.util import NamedTemporaryFile
from obspy.io.reftek.core import (
    _read_reftek130, _is_reftek130, _parse_next_packet)


class ReftekTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.reftek
    """
    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.datapath = os.path.join(self.path, "data")
        self.reftek_filename = "225051000_00008656"
        self.reftek_file = os.path.join(self.datapath, self.reftek_filename)
        self.mseed_filenames = [
            "2015282_225051_0ae4c_1_1.msd",
            "2015282_225051_0ae4c_1_2.msd", "2015282_225051_0ae4c_1_3.msd"]
        self.mseed_files = [os.path.join(self.datapath, filename)
                            for filename in self.mseed_filenames]
        # files "2015282_225051_0ae4c_1_[123].msd" contain miniseed data
        # converted with "rt_mseed" tool of Reftek utilities.

        # information on the data packets in the test file:
        # (omitting leading/trailing EH/ET packet)
        #   >>> for p in packets[1:-1]:
        #   ...     print(
        #   ...         "{:02d}".format(p.packet_sequence),
        #   ...         p.channel_number,
        #   ...         "{:03d}".format(p.number_of_samples),
        #   ...         p.time)
        #   01 0 549 2015-10-09T22:50:51.000000Z
        #   02 1 447 2015-10-09T22:50:51.000000Z
        #   03 2 805 2015-10-09T22:50:51.000000Z
        #   04 0 876 2015-10-09T22:50:53.745000Z
        #   05 1 482 2015-10-09T22:50:53.235000Z
        #   06 1 618 2015-10-09T22:50:55.645000Z
        #   07 2 872 2015-10-09T22:50:55.025000Z
        #   08 0 892 2015-10-09T22:50:58.125000Z
        #   09 1 770 2015-10-09T22:50:58.735000Z
        #   10 2 884 2015-10-09T22:50:59.385000Z
        #   11 0 848 2015-10-09T22:51:02.585000Z
        #   12 1 790 2015-10-09T22:51:02.585000Z
        #   13 2 844 2015-10-09T22:51:03.805000Z
        #   14 0 892 2015-10-09T22:51:06.215000Z
        #   15 1 768 2015-10-09T22:51:05.925000Z
        #   16 2 884 2015-10-09T22:51:08.415000Z
        #   17 1 778 2015-10-09T22:51:10.765000Z
        #   18 0 892 2015-10-09T22:51:11.675000Z
        #   19 2 892 2015-10-09T22:51:12.835000Z
        #   20 1 736 2015-10-09T22:51:14.655000Z
        #   21 0 892 2015-10-09T22:51:16.135000Z
        #   22 2 860 2015-10-09T22:51:17.295000Z
        #   23 1 738 2015-10-09T22:51:18.335000Z
        #   24 0 892 2015-10-09T22:51:20.595000Z
        #   25 1 673 2015-10-09T22:51:22.025000Z
        #   26 2 759 2015-10-09T22:51:21.595000Z
        #   27 0 067 2015-10-09T22:51:25.055000Z

    def test_read_reftek130(self):
        """
        Test original reftek 130 data file against miniseed files converted
        using "rt_mseed" utility from Trimble/Reftek.

        rt_mseed fills in network as "XX", location as "01" and channels as
        "001", "002", "003".
        """
        st_reftek = _read_reftek130(
            self.reftek_file, network="XX", location="01",
            component_codes=["1", "2", "3"])
        st_mseed = obspy.Stream()
        for file_ in self.mseed_files:
            st_mseed += obspy.read(file_, "MSEED")
        # reftek reader correctly fills in band+instrument code but rt_mseed
        # does not apparently, so set it now for the comparison
        for tr in st_mseed:
            tr.stats.channel = "EH" + tr.stats.channel[-1]
            tr.stats.pop("_format")
            tr.stats.pop("mseed")
        # check reftek130 low-level headers separately:
        for tr in st_reftek:
            self.assertTrue("reftek130" in tr.stats)
            # XXX TODO check reftek specific headers
            tr.stats.pop("reftek130")
        # sort streams
        st_reftek = st_reftek.sort()
        st_mseed = st_mseed.sort()
        # check amount of traces
        self.assertEqual(len(st_reftek), len(st_mseed))
        # check equality of headers
        for tr_got, tr_expected in zip(st_reftek, st_mseed):
            self.assertEqual(tr_got.stats, tr_expected.stats)
        # check equality of data
        for tr_got, tr_expected in zip(st_reftek, st_mseed):
            np.testing.assert_array_equal(tr_got.data, tr_expected.data)

    def test_is_reftek130(self):
        """
        Test checking whether file is REFTEK130 format or not.
        """
        self.assertTrue(_is_reftek130(self.reftek_file))
        for file_ in self.mseed_files:
            self.assertFalse(_is_reftek130(file_))

    def test_integration_with_obspy_core(self):
        """
        Test the integration with ObsPy core.
        """
        st_reftek = obspy.read(
            self.reftek_file, network="XX", location="01",
            component_codes=["1", "2", "3"])
        st_mseed = obspy.Stream()
        for file_ in self.mseed_files:
            st_mseed += obspy.read(file_, "MSEED")
        # sort streams
        st_reftek = st_reftek.sort()
        st_mseed = st_mseed.sort()
        # check amount of traces
        self.assertEqual(len(st_reftek), len(st_mseed))
        # check equality of data
        for tr_got, tr_expected in zip(st_reftek, st_mseed):
            np.testing.assert_array_equal(tr_got.data, tr_expected.data)

    def test_error_no_packets_read(self):
        """
        Test error message when no packets could be read from file.
        """
        with NamedTemporaryFile() as fh:
            # try to read empty file, finding no packets
            self.assertRaises(Exception, _read_reftek130, fh.name)
        # try to read mseed file, finding no packets
        self.assertRaises(Exception, _read_reftek130, self.mseed_files[0])

    def test_error_disturbed_packet_sequence(self):
        """
        Test error message when packet sequence is non-contiguous (one packet
        missing).
        """
        with NamedTemporaryFile() as fh:
            with open(self.reftek_file, 'rb') as fh2:
                # write packages to the file and omit one packet
                # (packets are 1024 byte each)
                fh.write(fh2.read(1024 * 5))
                fh2.seek(1024 * 6)
                fh.write(fh2.read())
            fh.seek(0)
            # try to read file, finding a non-contiguous packet sequence
            self.assertRaises(NotImplementedError, _read_reftek130, fh.name)

    def test_string_formatting_of_packet(self):
        """
        Check print formatting of packets
        """
        expected_eh_first_lines = (
            'EH Packet',
            '\texperiment_number: 00',
            '\tunit_id: AE4C',
            '\ttime: 2015-10-09T22:50:51.000000Z',
            '\tbyte_count: 416',
            '\tpacket_sequence: 0',
            '\t--------------------',
            '\tevent_number: 427',
            '\tdata_stream_number: 0',
            )
        some_other_eh_lines = (
            '\tevent_number: 427',
            '\tdata_stream_number: 0',
            '\tdata_format: C0',
            '\ttrigger_time_message: Trigger Time = 2015282225051000',
            '\ttime_source: 1',
            '\ttime_quality: ?',
            '\tstation_name_extension:  ',
            '\tstation_name: KW1 ',
            '\tstream_name: EH              ',
            '\tsampling_rate: 200',
            '\ttrigger_type: CON ',
            '\ttrigger_time: 2015-10-09T22:50:51.000000Z',
            '\tfirst_sample_time: 2015-10-09T22:50:51.000000Z',
            '\tdetrigger_time: None',
            '\tlast_sample_time: None',
            "\tchannel_adjusted_nominal_bit_weights: (u'104.2 mV', "
            "u'104.2 mV', u'104.2 mV', None, None, None, None, None, None, "
            "None, None, None, None, None, None, None)",
            "\tchannel_true_bit_weights: (u'1.585 uV', u'1.587 uV', "
            "u'1.587 uV', None, None, None, None, None, None, None, None, "
            "None, None, None, None, None)",
            "\tchannel_gain_code: (u'1', u'1', u'1', None, None, None, None, "
            "None, None, None, None, None, None, None, None, None)",
            "\tchannel_ad_resolution_code: (u'3', u'3', u'3', None, None, "
            "None, None, None, None, None, None, None, None, None, None, "
            "None)",
            "\tchannel_fsa_code: (u'3', u'3', u'3', None, None, None, None, "
            "None, None, None, None, None, None, None, None, None)",
            '\tchannel_code: (None, None, None, None, None, None, None, None, '
            'None, None, None, None, None, None, None, None)',
            '\ttotal_installed_channels: 3',
            '\tstation_comment: STATION COMMENT                         ',
            )
        expected_dt_first_lines = (
            'DT Packet',
            '\texperiment_number: 00',
            '\tunit_id: AE4C',
            '\ttime: 2015-10-09T22:50:51.000000Z',
            '\tbyte_count: 1024',
            '\tpacket_sequence: 1',
            '\t--------------------',
            '\tevent_number: 0427',
            '\tdata_stream_number: 0',
            '\tchannel_number: 0',
            '\tnumber_of_samples: 549')

        with open(self.reftek_file, 'rb') as fh:
            eh_packet = _parse_next_packet(fh)
            dt_packet = _parse_next_packet(fh)
        eh_lines = str(eh_packet).splitlines()
        dt_lines = str(dt_packet).splitlines()

        for expected, got in zip(expected_eh_first_lines, eh_lines):
            self.assertEqual(got, expected)
        for line in some_other_eh_lines:
            self.assertTrue(line in eh_lines)
        for expected, got in zip(expected_dt_first_lines, dt_lines):
            self.assertEqual(got, expected)


def suite():
    return unittest.makeSuite(ReftekTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
