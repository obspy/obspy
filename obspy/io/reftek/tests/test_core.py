#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import io
import os
import re
import unittest
import warnings

import numpy as np

import obspy
from obspy.core.util import NamedTemporaryFile
from obspy.io.reftek.core import (
    _read_reftek130, _is_reftek130, Reftek130, Reftek130Exception)
from obspy.io.reftek.packet import (
    _unpack_C0_C2_data_fast, _unpack_C0_C2_data_safe, _unpack_C0_C2_data,
    EHPacket, _initial_unpack_packets)


class ReftekTestCase(unittest.TestCase):
    """
    Test suite for obspy.io.reftek
    """
    def setUp(self):
        try:
            # doctests of __init__.py produce warnings that get caught. if we
            # don't raze the slate out the registry here, we can't test those
            # warnings in the unit tests (if doctests run before unittests)..
            from obspy.io.reftek.core import __warningregistry__
            __warningregistry__.clear()
        except ImportError:
            # import error means no warning has been issued
            # before, so nothing to do.
            pass
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.datapath = os.path.join(self.path, "data")
        self.reftek_filename = "225051000_00008656"
        self.reftek_file = os.path.join(self.datapath, self.reftek_filename)
        self.reftek_file_steim2 = os.path.join(self.datapath,
                                               '104800000_000093F8')
        self.reftek_file_16 = os.path.join(
            self.datapath, '065520000_013EE8A0.rt130')
        self.reftek_file_16_npz = os.path.join(
            self.datapath, '065520000_013EE8A0.npz')
        self.reftek_file_32 = os.path.join(
            self.datapath, '230000005_0036EE80_cropped.rt130')
        self.reftek_file_32_npz = os.path.join(
            self.datapath, '230000005_0036EE80_cropped.npz')
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
        self.reftek_file_vpu = os.path.join(self.datapath,
                                            '221935615_00000000')

    def _assert_reftek130_test_stream(self, st_reftek):
        """
        Test reftek 130 data read into a stream object against miniseed files
        converted using "rt_mseed" utility from Trimble/Reftek.

        Note that rt_mseed fills in network as "XX", location as "01" and
        channels as "001", "002", "003".
        """
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
            tr.stats.pop("_format", None)
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

    def test_read_reftek130_steim1(self):
        """
        Test original reftek 130 data file against miniseed files converted
        using "rt_mseed" utility from Trimble/Reftek.

        rt_mseed fills in network as "XX", location as "01" and channels as
        "001", "002", "003".
        """
        st_reftek = _read_reftek130(
            self.reftek_file, network="XX", location="01",
            component_codes=["1", "2", "3"],
            sort_permuted_package_sequence=True)
        self._assert_reftek130_test_stream(st_reftek)

    def test_read_reftek130_steim2(self):
        """
        Test reading a steim2 encoded data file.

        Unpacking of data is tested separately so just checking a few samples
        at the start should suffice.
        """
        st = _read_reftek130(
            self.reftek_file_steim2, network="XX", location="01",
            component_codes=["1", "2", "3"],
            sort_permuted_package_sequence=True)
        # note: test data has stream name defined as 'DS 1', so we end up with
        # non-SEED conforming channel codes which is expected
        self.assertEqual(len(st), 3)
        self.assertEqual(len(st[0]), 3788)
        self.assertEqual(len(st[1]), 3788)
        self.assertEqual(len(st[2]), 3788)
        self.assertEqual(st[0].id, 'XX.TL01.01.DS 11')
        self.assertEqual(st[1].id, 'XX.TL01.01.DS 12')
        self.assertEqual(st[2].id, 'XX.TL01.01.DS 13')
        np.testing.assert_array_equal(
            st[0].data[:5], [26814, 26823, 26878, 26941, 26942])
        np.testing.assert_array_equal(
            st[1].data[:5], [-1987, -1984, -1959, -1966, -1978])
        np.testing.assert_array_equal(
            st[2].data[:5], [-2404, -2376, -2427, -2452, -2452])

    def test_read_reftek130_no_component_codes_specified(self):
        """
        Test reading reftek 130 data file not providing component codes
        (relying on the information in header packet).

        rt_mseed fills in network as "XX", location as "01" and channels as
        "001", "002", "003".
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            st_reftek = _read_reftek130(
                self.reftek_file, network="XX", location="01",
                sort_permuted_package_sequence=True)
        self.assertEqual(len(w), 8)
        for w_ in w:
            self.assertEqual(
                str(w_.message),
                'No channel code specified in the data file and no component '
                'codes specified. Using stream label and number of channel in '
                'file as channel codes.')
        # check that channel codes are set with stream label from EH packet +
        # enumerated channel number starting at 0
        for tr, cha in zip(st_reftek, ('EH0', 'EH0', 'EH0', 'EH1', 'EH1',
                                       'EH1', 'EH2', 'EH2')):
            self.assertEqual(tr.stats.channel, cha)
        for tr in st_reftek:
            # need to adapt channel codes to compare against mseed stream now..
            tr.stats.channel = (
                'EH' + {'0': '1', '1': '2', '2': '3'}[tr.stats.channel[-1]])
        self._assert_reftek130_test_stream(st_reftek)

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
        self._assert_reftek130_test_stream(st_reftek)

    def test_error_no_packets_read(self):
        """
        Test error message when no packets could be read from file.
        """
        with NamedTemporaryFile() as fh:
            # try to read empty file, finding no packets
            self.assertRaises(Reftek130Exception, _read_reftek130, fh.name)
        # try to read mseed file, finding no packets
        self.assertRaises(Reftek130Exception, _read_reftek130,
                          self.mseed_files[0],
                          sort_permuted_package_sequence=True)

    def test_warning_disturbed_packet_sequence(self):
        """
        Test warning message when packet sequence is non-contiguous (one packet
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
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _read_reftek130(fh.name, network="XX", location="01",
                                component_codes=["1", "2", "3"],
                                sort_permuted_package_sequence=True)
        self.assertEqual(len(w), 1)
        self.assertEqual(str(w[0].message),
                         'Detected a non-contiguous packet sequence!')

    def test_read_file_perturbed_packet_sequence(self):
        """
        Test data read from file when packet sequence is perturbed. This makes
        sure that data is read correctly even when the array storing the
        packets gets permuted and thus becomes incontiguous for C.
        """
        with NamedTemporaryFile() as fh:
            with open(self.reftek_file, 'rb') as fh2:
                # write packages to the file and move some packets around in
                # the file
                # (packets are 1024 byte each)
                tmp1 = fh2.read(1024 * 2)
                tmp2 = fh2.read(1024 * 4)
                tmp3 = fh2.read(1024 * 1)
                tmp4 = fh2.read(1024 * 2)
                tmp5 = fh2.read(1024 * 3)
                tmp6 = fh2.read()
            fh.write(tmp1)
            fh.write(tmp3)
            fh.write(tmp5)
            fh.write(tmp6)
            fh.write(tmp2)
            fh.write(tmp4)
            fh.seek(0)
            # try to read file, finding a non-contiguous packet sequence
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                st_reftek = _read_reftek130(
                    fh.name, network="XX", location="01",
                    component_codes=["1", "2", "3"],
                    sort_permuted_package_sequence=True)
        st_reftek.merge(-1)
        self.assertEqual(len(w), 1)
        self.assertEqual(str(w[0].message),
                         'Detected permuted packet sequence, sorting.')
        self._assert_reftek130_test_stream(st_reftek)

    def test_drop_not_implemented_packets(self):
        """
        Test error message when some not implemented packet types are dropped
        """
        with NamedTemporaryFile() as fh:
            with open(self.reftek_file, 'rb') as fh2:
                # write packages to the file and write the last three packets
                # with a different packet type
                # (packets are 1024 byte each)
                tmp = fh2.read()[:-(1024 * 3)]
                fh2.seek(-(1024 * 3), 2)
                tmp2 = fh2.read(1024)
                tmp3 = fh2.read(1024)
                tmp4 = fh2.read(1024)
            # write last three packages with some different packet types
            fh.write(tmp)
            fh.write(b"AA")
            fh.write(tmp2[2:])
            fh.write(b"AA")
            fh.write(tmp3[2:])
            fh.write(b"BB")
            fh.write(tmp4[2:])
            fh.seek(0)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _read_reftek130(
                    fh.name, network="XX", location="01",
                    component_codes=["1", "2", "3"],
                    sort_permuted_package_sequence=True)
        self.assertEqual(len(w), 2)
        self.assertTrue(
            re.match(
                r"Encountered some packets of types that are not implemented "
                r"yet \(types: \[b?'AA', b?'BB'\]\). Dropped 3 packets "
                r"overall.",
                str(w[0].message)))
        # this message we get because ET packet at end is now missing
        self.assertEqual(
            str(w[1].message),
            'No event trailer (ET) packets in packet sequence. File might be '
            'truncated.')

    def test_missing_event_trailer_packet(self):
        """
        Test that reading the file if the ET packet is missing works and a
        warning is shown.
        """
        with NamedTemporaryFile() as fh:
            with open(self.reftek_file, 'rb') as fh2:
                # write packages to the file and omit last (ET) packet
                # (packets are 1024 byte each)
                tmp = fh2.read()
                tmp = tmp[:-1024]
            fh.write(tmp)
            fh.seek(0)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                st_reftek = _read_reftek130(
                    fh.name, network="XX", location="01",
                    component_codes=["1", "2", "3"],
                    sort_permuted_package_sequence=True)
        self.assertEqual(len(w), 1)
        self.assertEqual(
            str(w[0].message),
            'No event trailer (ET) packets in packet sequence. File might be '
            'truncated.')
        self._assert_reftek130_test_stream(st_reftek)

    def test_truncated_last_packet(self):
        """
        Test that reading the file works, if the ET packet at the end is
        truncated and thus omitted.
        """
        with NamedTemporaryFile() as fh:
            with open(self.reftek_file, 'rb') as fh2:
                # write packages to the file and truncate last (ET) packet
                # (packets are 1024 byte each)
                tmp = fh2.read()
                tmp = tmp[:-10]
            fh.write(tmp)
            fh.seek(0)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                st_reftek = _read_reftek130(
                    fh.name, network="XX", location="01",
                    component_codes=["1", "2", "3"],
                    sort_permuted_package_sequence=True)
        self.assertEqual(len(w), 2)
        # we get two warnings, one about the truncated packet and one about the
        # missing last (ET) packet
        self.assertEqual(str(w[0].message), 'Length of data not a multiple of '
                         '1024. Data might be truncated. Dropping 1014 '
                         'byte(s) at the end.')
        self.assertEqual(
            str(w[1].message),
            'No event trailer (ET) packets in packet sequence. File might be '
            'truncated.')
        # data should be read OK aside from the warnings
        self._assert_reftek130_test_stream(st_reftek)

    def test_no_eh_et_packet(self):
        """
        Test error messages when reading a file without any EH/ET packet.
        """
        with NamedTemporaryFile() as fh:
            with open(self.reftek_file, 'rb') as fh2:
                # write packages to the file and omit first and last (EH/ET)
                # packet
                # (packets are 1024 byte each)
                tmp = fh2.read()
            fh.write(tmp[1024:-1024])
            fh.seek(0)
            with self.assertRaises(Reftek130Exception) as context:
                _read_reftek130(
                    fh.name, network="XX", location="01",
                    component_codes=["1", "2", "3"],
                    sort_permuted_package_sequence=True)
        self.assertEqual(
            str(context.exception),
            "Reftek data contains data packets without corresponding header "
            "or trailer packet.")

    def test_data_unpacking_steim1(self):
        """
        Test both unpacking routines for C0 data coding (STEIM1)
        """
        rt = Reftek130.from_file(self.reftek_file)
        expected = np.load(os.path.join(self.datapath,
                                        "unpacked_data_steim1.npy"))
        packets = rt._data[rt._data['packet_type'] == b'DT'][:10]
        for func in (_unpack_C0_C2_data, _unpack_C0_C2_data_fast,
                     _unpack_C0_C2_data_safe):
            got = func(packets, encoding='C0')
            np.testing.assert_array_equal(got, expected)

    def test_data_unpacking_steim2(self):
        """
        Test both unpacking routines for C2 data coding (STEIM2)
        """
        rt = Reftek130.from_file(self.reftek_file_steim2)
        expected = np.load(os.path.join(self.datapath,
                                        "unpacked_data_steim2.npy"))
        packets = rt._data[rt._data['packet_type'] == b'DT'][:10]
        for func in (_unpack_C0_C2_data, _unpack_C0_C2_data_fast,
                     _unpack_C0_C2_data_safe):
            got = func(packets, encoding='C2')
            np.testing.assert_array_equal(got, expected)

    def test_string_representations(self):
        """
        Test string representations of Reftek object and Packets
        """
        expected = [
            "Reftek130 (29 packets, file: {})".format(self.reftek_file),
            "Packet Sequence  Byte Count  Data Fmt  Sampling Rate      Time",
            "  | Packet Type   |  Event #  | Station | Channel #         |",
            "  |   |  Unit ID  |    | Data Stream #  |   |  # of samples |",
            "  |   |   |  Exper.#   |   |  |  |      |   |    |          |",
            "0000 EH AE4C  0  416  427  0 C0 KW1    200.        "
            "2015-10-09T22:50:51.000000Z",
            "0001 DT AE4C  0 1024  427  0 C0             0  549 "
            "2015-10-09T22:50:51.000000Z",
            "0002 DT AE4C  0 1024  427  0 C0             1  447 "
            "2015-10-09T22:50:51.000000Z",
            "0003 DT AE4C  0 1024  427  0 C0             2  805 "
            "2015-10-09T22:50:51.000000Z",
            "0004 DT AE4C  0 1024  427  0 C0             0  876 "
            "2015-10-09T22:50:53.745000Z",
            "0005 DT AE4C  0 1024  427  0 C0             1  482 "
            "2015-10-09T22:50:53.235000Z",
            "0006 DT AE4C  0 1024  427  0 C0             1  618 "
            "2015-10-09T22:50:55.645000Z",
            "0007 DT AE4C  0 1024  427  0 C0             2  872 "
            "2015-10-09T22:50:55.025000Z",
            "0008 DT AE4C  0 1024  427  0 C0             0  892 "
            "2015-10-09T22:50:58.125000Z",
            "0009 DT AE4C  0 1024  427  0 C0             1  770 "
            "2015-10-09T22:50:58.735000Z",
            "0010 DT AE4C  0 1024  427  0 C0             2  884 "
            "2015-10-09T22:50:59.385000Z",
            "0011 DT AE4C  0 1024  427  0 C0             0  848 "
            "2015-10-09T22:51:02.585000Z",
            "0012 DT AE4C  0 1024  427  0 C0             1  790 "
            "2015-10-09T22:51:02.585000Z",
            "0013 DT AE4C  0 1024  427  0 C0             2  844 "
            "2015-10-09T22:51:03.805000Z",
            "0014 DT AE4C  0 1024  427  0 C0             0  892 "
            "2015-10-09T22:51:06.215000Z",
            "0015 DT AE4C  0 1024  427  0 C0             1  768 "
            "2015-10-09T22:51:05.925000Z",
            "0016 DT AE4C  0 1024  427  0 C0             2  884 "
            "2015-10-09T22:51:08.415000Z",
            "0017 DT AE4C  0 1024  427  0 C0             1  778 "
            "2015-10-09T22:51:10.765000Z",
            "0018 DT AE4C  0 1024  427  0 C0             0  892 "
            "2015-10-09T22:51:11.675000Z",
            "0019 DT AE4C  0 1024  427  0 C0             2  892 "
            "2015-10-09T22:51:12.835000Z",
            "0020 DT AE4C  0 1024  427  0 C0             1  736 "
            "2015-10-09T22:51:14.655000Z",
            "0021 DT AE4C  0 1024  427  0 C0             0  892 "
            "2015-10-09T22:51:16.135000Z",
            "0022 DT AE4C  0 1024  427  0 C0             2  860 "
            "2015-10-09T22:51:17.295000Z",
            "0023 DT AE4C  0 1024  427  0 C0             1  738 "
            "2015-10-09T22:51:18.335000Z",
            "0024 DT AE4C  0 1024  427  0 C0             0  892 "
            "2015-10-09T22:51:20.595000Z",
            "0025 DT AE4C  0 1024  427  0 C0             1  673 "
            "2015-10-09T22:51:22.025000Z",
            "0026 DT AE4C  0 1024  427  0 C0             2  759 "
            "2015-10-09T22:51:21.595000Z",
            "0027 DT AE4C  0 1024  427  0 C0             0   67 "
            "2015-10-09T22:51:25.055000Z",
            "0028 ET AE4C  0  416  427  0 C0 KW1    200.        "
            "2015-10-09T22:50:51.000000Z",
            "(detailed packet information with: "
            "'print(Reftek130.__str__(compact=False))')"]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            rt130 = Reftek130.from_file(self.reftek_file)
        self.assertEqual(expected, str(rt130).splitlines())

    def test_reading_packet_with_vpu_float_string(self):
        """
        Test reading a data stream with VPU floating point in header, see #1632
        """
        with open(self.reftek_file_vpu, 'rb') as fh:
            data = fh.read(1024)
        data = _initial_unpack_packets(data)
        eh = EHPacket(data[0])
        self.assertEqual(
            eh.channel_sensor_vpu,
            (2.4, 2.4, 2.4, None, None, None, None, None, None, None, None,
             None, None, None, None, None))
        # reading the file should work..
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = obspy.read(self.reftek_file_vpu,
                            sort_permuted_package_sequence=True)
        self.assertEqual(len(st), 2)
        self.assertEqual(len(st[0]), 890)
        self.assertEqual(len(st[1]), 890)
        np.testing.assert_array_equal(
            st[0][:10], [210, 212, 208, 211, 211, 220, 216, 215, 219, 218])

    def test_reading_file_with_multiple_events(self):
        """
        Test reading a "multiplexed" file with multiple "events" in it.

        Simply reuse the existing test data in one read operation.
        """
        with open(self.reftek_file_vpu, 'rb') as fh:
            data = fh.read()
        with open(self.reftek_file, 'rb') as fh:
            data += fh.read()
        bytes_ = io.BytesIO(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = obspy.read(bytes_, format='REFTEK130',
                            sort_permuted_package_sequence=True)
        self.assertEqual(len(st), 10)
        # we should have data from two different files/stations in there
        for tr in st[:8]:
            self.assertEqual(tr.stats.station, 'KW1')
        for tr in st[8:]:
            self.assertEqual(tr.stats.station, 'TL02')

    def test_reading_file_with_no_data_in_channel_zero(self):
        """
        Test reading a file that has no data packets in channel zero (e.g.
        6-channel Reftek and only recording on channels 4-6)

        Simply reuse the existing test data omitting the data packet that has
        channel zero.
        """
        with open(self.reftek_file_vpu, 'rb') as fh:
            data = fh.read()
        # only use first packet (the EH packet) and last packet (a DT packet
        # for channel number 1, i.e. channel 2)
        data = data[:1024] + data[-1024:]
        bio = io.BytesIO(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = obspy.read(bio, format='REFTEK130')
        self.assertEqual(len(st), 1)
        # just a few basic checks, reading data is covered in other tests
        tr = st[0]
        self.assertEqual(tr.id, ".TL02..DS 11")
        self.assertEqual(len(tr), 890)

    def test_reading_file_with_encoding_32(self):
        """
        Test reading a file with encoding '32' (uncompressed 32 bit integer)

        Only tests unpacked sample data, everything else should be covered by
        existing tests.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = _read_reftek130(self.reftek_file_32)
        # read expected data
        npz = np.load(self.reftek_file_32_npz)
        # compare it
        self.assertEqual(len(st), 3)
        for tr, (_, expected) in zip(st, sorted(npz.items())):
            np.testing.assert_array_equal(expected, tr.data)

    def test_reading_file_with_encoding_16(self):
        """
        Test reading a file with encoding '16' (uncompressed 16 bit integer)

        Only tests unpacked sample data, everything else should be covered by
        existing tests.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = _read_reftek130(self.reftek_file_16)
        # read expected data
        npz = np.load(self.reftek_file_16_npz)
        # compare it
        self.assertEqual(len(st), 3)
        self.assertEqual(len(st[0]), 2090)
        self.assertEqual(len(st[1]), 2090)
        self.assertEqual(len(st[2]), 2090)
        for tr, (_, expected) in zip(st, sorted(npz.items())):
            np.testing.assert_array_equal(expected, tr.data)
