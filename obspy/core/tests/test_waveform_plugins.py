# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io
import os
import threading
import time
import unittest
import warnings
from copy import deepcopy

from pkg_resources import load_entry_point
import numpy as np

from obspy import Trace, read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.base import NamedTemporaryFile, _get_entry_points


class WaveformPluginsTestCase(unittest.TestCase):
    """
    Test suite for all waveform plug-ins.
    """
    longMessage = True

    def test_raiseOnEmptyFile(self):
        """
        Test case ensures that empty files do raise warnings.
        """
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # create empty file
            open(tmpfile, 'wb').close()
            formats_ep = _get_entry_points('obspy.plugin.waveform',
                                           'readFormat')
            # using format keyword
            for ep in formats_ep.values():
                is_format = load_entry_point(
                    ep.dist.key, 'obspy.plugin.waveform.' + ep.name,
                    'isFormat')
                self.assertFalse(False, is_format(tmpfile))

    def test_readAndWrite(self):
        """
        Tests read and write methods for all waveform plug-ins.
        """
        data = np.arange(0, 2000)
        start = UTCDateTime(2009, 1, 13, 12, 1, 2, 999000)
        formats = _get_entry_points('obspy.plugin.waveform', 'writeFormat')
        for format in formats:
            # XXX: skip SEGY and SU formats for now as they need some special
            # headers.
            if format in ['SEGY', 'SU', 'SEG2']:
                continue
            for native_byteorder in ['<', '>']:
                for byteorder in ['<', '>', '=']:
                    if format == 'SAC' and byteorder == '=':
                        # SAC file format enforces '<' or '>'
                        # byteorder on writing
                        continue
                    # new trace object in native byte order
                    dt = np.dtype(np.int_).newbyteorder(native_byteorder)
                    if format in ('MSEED', 'GSE2'):
                        # MiniSEED and GSE2 cannot write int64, enforce type
                        dt = np.int32
                    tr = Trace(data=data.astype(dt))
                    tr.stats.network = "BW"
                    tr.stats.station = "MANZ1"
                    tr.stats.location = "00"
                    tr.stats.channel = "EHE"
                    tr.stats.calib = 0.199999
                    tr.stats.delta = 0.005
                    tr.stats.starttime = start
                    # create waveform file with given format and byte order
                    with NamedTemporaryFile() as tf:
                        outfile = tf.name
                        tr.write(outfile, format=format, byteorder=byteorder)
                        if format == 'Q':
                            outfile += '.QHD'
                        # read in again using auto detection
                        st = read(outfile)
                        self.assertEqual(len(st), 1)
                        self.assertEqual(st[0].stats._format, format)
                        # read in using format argument
                        st = read(outfile, format=format)
                        self.assertEqual(len(st), 1)
                        self.assertEqual(st[0].stats._format, format)
                        # read in using a BytesIO instances, skip Q files as
                        # it needs multiple files
                        if format not in ['Q']:
                            # file handler without format
                            with open(outfile, 'rb') as fp:
                                st = read(fp)
                            self.assertEqual(len(st), 1)
                            self.assertEqual(st[0].stats._format, format)
                            # file handler with format
                            with open(outfile, 'rb') as fp:
                                st = read(fp, format=format)
                            self.assertEqual(len(st), 1)
                            self.assertEqual(st[0].stats._format, format)
                            # BytesIO without format
                            with open(outfile, 'rb') as fp:
                                temp = io.BytesIO(fp.read())
                            st = read(temp)
                            self.assertEqual(len(st), 1)
                            self.assertEqual(st[0].stats._format, format)
                            # BytesIO with format
                            with open(outfile, 'rb') as fp:
                                temp = io.BytesIO(fp.read())
                            st = read(temp, format=format)
                            self.assertEqual(len(st), 1)
                            self.assertEqual(st[0].stats._format, format)
                        # Q files consist of two files - deleting additional
                        # file
                        if format == 'Q':
                            os.remove(outfile[:-4] + '.QBN')
                            os.remove(outfile[:-4] + '.QHD')
                    # check byte order
                    if format == 'SAC':
                        # SAC format preserves byteorder on writing
                        self.assertTrue(st[0].data.dtype.byteorder
                                        in ('=', byteorder))
                    else:
                        self.assertEqual(st[0].data.dtype.byteorder, '=')
                    # check meta data
                    # some formats do not contain a calibration factor
                    if format not in ['MSEED', 'WAV', 'TSPAIR', 'SLIST']:
                        self.assertAlmostEqual(st[0].stats.calib, 0.199999, 5)
                    else:
                        self.assertEqual(st[0].stats.calib, 1.0)
                    if format not in ['WAV']:
                        self.assertEqual(st[0].stats.starttime, start)
                        self.assertEqual(st[0].stats.endtime, start + 9.995)
                        self.assertEqual(st[0].stats.delta, 0.005)
                        self.assertEqual(st[0].stats.sampling_rate, 200.0)
                    # network/station/location/channel codes
                    if format in ['Q', 'SH_ASC', 'GSE2']:
                        # no network or location code in Q, SH_ASC, GSE2
                        self.assertEqual(st[0].id, ".MANZ1..EHE")
                    elif format not in ['WAV']:
                        self.assertEqual(st[0].id, "BW.MANZ1.00.EHE")

    def test_isFormat(self):
        """
        Tests all isFormat methods against all data test files from the other
        modules for false positives.
        """
        KNOWN_FALSE = [
            os.path.join('seisan', 'tests', 'data', 'SEISAN_Bug',
                         '2011-09-06-1311-36S.A1032_001BH_Z_MSEED'),
            os.path.join('core', 'tests', 'data',
                         'IU_ULN_00_LH1_2015-07-18T02.mseed'),
        ]
        formats_ep = _get_entry_points('obspy.plugin.waveform', 'isFormat')
        formats = list(formats_ep.values())
        # Collect all false positives.
        false_positives = []
        # Big loop over every format.
        for format in formats:
            # search isFormat for given entry point
            is_format = load_entry_point(
                format.dist.key, 'obspy.plugin.waveform.' + format.name,
                'isFormat')
            # get all the test directories.
            paths = [os.path.join(f.dist.location, 'obspy',
                                  f.module_name.split('.')[1], 'tests', 'data')
                     for f in formats
                     if f.module_name.split('.')[1] !=
                     format.module_name.split('.')[1]]
            # Remove double paths because some modules can have two file
            # formats.
            paths = set(paths)
            # Remove path if one module defines two file formats.
            for path in paths:
                # Collect all files found.
                filelist = []
                # Walk every path.
                for directory, _, files in os.walk(path):
                    filelist.extend([os.path.join(directory, _i) for _i in
                                     files])
                for file in filelist:
                    if any([n in file for n in KNOWN_FALSE]):
                        continue
                    if is_format(file) is True:  # pragma: no cover
                        false_positives.append((format.name, file))
        # Use try except to produce a meaningful error message.
        try:
            self.assertEqual(len(false_positives), 0)
        except:  # pragma: no cover
            msg = 'False positives for isFormat:\n'
            msg += '\n'.join(['\tFormat %s: %s' % (_i[0], _i[1]) for _i in
                              false_positives])
            raise Exception(msg)

    def test_readThreadSafe(self):
        """
        Tests for race conditions. Reading n_threads (currently 30) times
        the same waveform file in parallel and compare the results which must
        be all the same.
        """
        data = np.arange(0, 500)
        start = UTCDateTime(2009, 1, 13, 12, 1, 2, 999000)
        formats = _get_entry_points('obspy.plugin.waveform', 'writeFormat')
        for format in formats:
            # XXX: skip SEGY and SU formats for now as they need some special
            # headers.
            if format in ['SEGY', 'SU', 'SEG2']:
                continue

            dt = np.int_
            if format in ('MSEED', 'GSE2'):
                dt = np.int32
            tr = Trace(data=data.astype(dt))
            tr.stats.network = "BW"
            tr.stats.station = "MANZ1"
            tr.stats.location = "00"
            tr.stats.channel = "EHE"
            tr.stats.calib = 0.999999
            tr.stats.delta = 0.005
            tr.stats.starttime = start
            # create waveform file with given format and byte order
            with NamedTemporaryFile() as tf:
                outfile = tf.name
                tr.write(outfile, format=format)
                if format == 'Q':
                    outfile += '.QHD'
                n_threads = 30
                streams = []
                timeout = 120
                if 'TRAVIS' in os.environ:
                    timeout = 570  # 30 seconds under Travis' limit
                cond = threading.Condition()

                def testFunction(streams, cond):
                    st = read(outfile, format=format)
                    streams.append(st)
                    with cond:
                        cond.notify()
                # Read the ten files at one and save the output in the just
                # created class.
                our_threads = []
                for _i in range(n_threads):
                    thread = threading.Thread(target=testFunction,
                                              args=(streams, cond))
                    thread.start()
                    our_threads.append(thread)
                our_threads = set(our_threads)
                # Loop until all threads are finished.
                start = time.time()
                while True:
                    with cond:
                        cond.wait(1)
                    remaining_threads = set(threading.enumerate())
                    if len(remaining_threads & our_threads) == 0:
                        break
                    # Avoid infinite loop and leave after some time; such a
                    # long time is needed for debugging with valgrind or Travis
                    elif time.time() - start >= timeout:  # pragma: no cover
                        msg = 'Not all threads finished after %d seconds!' % (
                            timeout)
                        raise Warning(msg)
                # Compare all values which should be identical and clean up
                # files
                for st in streams:
                    np.testing.assert_array_equal(st[0].data, tr.data)
                if format == 'Q':
                    os.remove(outfile[:-4] + '.QBN')
                    os.remove(outfile[:-4] + '.QHD')

    def test_issue193(self):
        """
        Test for issue #193: if non-contiguous array is written correctly.
        """
        warnings.filterwarnings("ignore", "Detected non contiguous data")
        # test all plugins with both read and write method
        formats_write = \
            set(_get_entry_points('obspy.plugin.waveform', 'writeFormat'))
        formats_read = \
            set(_get_entry_points('obspy.plugin.waveform', 'readFormat'))
        formats = set.intersection(formats_write, formats_read)
        # mseed will raise exception for int64 data, thus use int32 only
        data = np.arange(10, dtype=np.int32)
        # make array non-contiguous
        data = data[::2]
        tr = Trace(data=data)
        for format in formats:
            # XXX: skip SEGY and SU formats for now as they need some special
            # headers.
            if format in ['SEGY', 'SU', 'SEG2']:
                continue
            with NamedTemporaryFile() as tf:
                tempfile = tf.name
                tr.write(tempfile, format)
                if format == "Q":
                    tempfile = tempfile + ".QHD"
                tr_test = read(tempfile, format)[0]
                if format == 'Q':
                    os.remove(tempfile[:-4] + '.QBN')
                    os.remove(tempfile[:-4] + '.QHD')
            np.testing.assert_array_equal(tr.data, tr_test.data)

    def test_readGzip2File(self):
        """
        Tests reading gzip compressed waveforms.
        """
        path = os.path.dirname(__file__)
        ascii_path = os.path.join(path, "..", "..", "io", "ascii",
                                  "tests", "data")
        st1 = read(os.path.join(ascii_path, 'tspair.ascii.gz'))
        st2 = read(os.path.join(ascii_path, 'tspair.ascii'))
        self.assertEqual(st1, st2)

    def test_readBzip2File(self):
        """
        Tests reading bzip2 compressed waveforms.
        """
        path = os.path.dirname(__file__)
        ascii_path = os.path.join(path, "..", "..", "io", "ascii",
                                  "tests", "data")
        st1 = read(os.path.join(ascii_path, 'slist.ascii.bz2'))
        st2 = read(os.path.join(ascii_path, 'slist.ascii'))
        self.assertEqual(st1, st2)

    def test_readTarArchive(self):
        """
        Tests reading tar compressed waveforms.
        """
        path = os.path.dirname(__file__)
        ascii_path = os.path.join(path, "..", "..", "io", "ascii",
                                  "tests", "data")
        # tar
        st1 = read(os.path.join(path, "data", "test.tar"))
        st2 = read(os.path.join(ascii_path, "slist.ascii"))
        self.assertEqual(st1, st2)
        # tar.gz
        st1 = read(os.path.join(path, "data", "test.tar.gz"))
        st2 = read(os.path.join(ascii_path, "slist.ascii"))
        self.assertEqual(st1, st2)
        # tar.bz2
        st1 = read(os.path.join(path, "data", "test.tar.bz2"))
        st2 = read(os.path.join(ascii_path, "slist.ascii"))
        self.assertEqual(st1, st2)
        # tgz
        st1 = read(os.path.join(path, "data", "test.tgz"))
        st2 = read(os.path.join(ascii_path, "slist.ascii"))
        self.assertEqual(st1, st2)

    def test_readZipArchive(self):
        """
        Tests reading zip compressed waveforms.
        """
        path = os.path.dirname(__file__)
        ascii_path = os.path.join(path, "..", "..", "io", "ascii",
                                  "tests", "data")
        st1 = read(os.path.join(path, 'data', 'test.zip'))
        st2 = read(os.path.join(ascii_path, 'slist.ascii'))
        self.assertEqual(st1, st2)

    def test_raiseOnUnknownFormat(self):
        """
        Test case for issue #338:
        """
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # create empty file
            open(tmpfile, 'wb').close()
            # using format keyword
            self.assertRaises(TypeError, read, tmpfile)

    def test_deepcopy(self):
        """
        Test for issue #689: deepcopy did not work for segy. In order to
        avoid complicated code to find test data for each waveform pluging,
        which read OK and have no errors we simply test by first writing
        the waveform and then reading it in. Thus test is limited to
        formats which we can also write.
        """
        # find all plugins with both read and write method
        formats_write = \
            set(_get_entry_points('obspy.plugin.waveform', 'writeFormat'))
        formats_read = \
            set(_get_entry_points('obspy.plugin.waveform', 'readFormat'))
        formats = set.intersection(formats_write, formats_read)
        stream_orig = read()
        for format in formats:
            # TODO: these formats error in read and writing, not in
            # deepcopy
            if format in ('SAC', 'SACXY', 'SEG2', 'Q', 'WAV'):
                continue
            stream = deepcopy(stream_orig)
            # set some data
            dt = np.float32
            if format in ('GSE2', 'MSEED'):
                dt = np.int32
            for tr in stream:
                tr.data = np.arange(tr.stats.npts).astype(dt)
            with NamedTemporaryFile() as tf:
                tmpfile = tf.name
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stream.write(format=format, filename=tmpfile)
                st = read(tmpfile, format=format)
            st.sort()
            st_deepcopy = deepcopy(st)
            st_deepcopy.sort()
            msg = "Error in wavform format=%s" % format
            self.assertEqual(str(st), str(st_deepcopy), msg=msg)


def suite():
    return unittest.makeSuite(WaveformPluginsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
