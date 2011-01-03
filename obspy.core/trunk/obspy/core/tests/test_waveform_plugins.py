# -*- coding: utf-8 -*-

from obspy.core import Trace, read
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util import NamedTemporaryFile, _getPlugins
from pkg_resources import load_entry_point
import numpy as np
import os
import threading
import time
import unittest
import warnings


class WaveformPluginsTestCase(unittest.TestCase):
    """
    Test suite for all installed waveform plug-ins.
    """

    def test_raiseOnEmptyFile(self):
        """
        Test case ensures that empty files do raise
        warnings. 
        """
        tmpfile = NamedTemporaryFile().name
        # generate empty file
        f = open(tmpfile, 'wb')
        f.write("")
        f.close()
        formats_ep = _getPlugins('obspy.plugin.waveform', 'readFormat')
        for ep in formats_ep.values():
            isFormat = load_entry_point(ep.dist.key,
                                        'obspy.plugin.waveform.' + ep.name,
                                        'isFormat')
            self.assertFalse(False, isFormat(tmpfile))
        os.remove(tmpfile)

    def test_readAndWriteAllInstalledWaveformPlugins(self):
        """
        Tests read and write methods for all installed waveform plug-ins.
        """
        data = np.arange(0, 2000)
        start = UTCDateTime(2009, 1, 13, 12, 1, 2, 999000)
        formats = _getPlugins('obspy.plugin.waveform', 'writeFormat')
        for format in formats:
            # XXX: skip SEGY for now as it needs some special headers
            if 'SEGY' in format:
                continue
            for native_byteorder in ['<', '>']:
                for byteorder in ['<', '>', '=']:
                    # new trace object in native byte order
                    dt = np.dtype("int").newbyteorder(native_byteorder)
                    if format in ('MSEED', 'GSE2'):
                        # MiniSEED and GSE2 cannot write int64, enforce type
                        dt = "int32"
                    tr = Trace(data=data.astype(dt))
                    tr.stats.network = "BW"
                    tr.stats.station = "MANZ1"
                    tr.stats.location = "00"
                    tr.stats.channel = "EHE"
                    tr.stats.calib = 0.199999
                    tr.stats.delta = 0.005
                    tr.stats.starttime = start
                    # create waveform file with given format and byte order
                    outfile = NamedTemporaryFile().name
                    tr.write(outfile, format=format, byteorder=byteorder)
                    if format == 'Q':
                        outfile += '.QHD'
                    # read in again using auto detection
                    st = read(outfile)
                    self.assertEquals(len(st), 1)
                    self.assertEquals(st[0].stats._format, format)
                    # read in using format argument
                    st = read(outfile, format=format)
                    self.assertEquals(len(st), 1)
                    self.assertEquals(st[0].stats._format, format)
                    # check byte order
                    self.assertEquals(st[0].data.dtype.byteorder, '=')
                    # check meta data
                    # MSEED and WAV do not contain the calibration factor
                    if format not in ['MSEED', 'WAV']:
                        self.assertAlmostEquals(st[0].stats.calib, 0.199999, 5)
                    else:
                        self.assertEquals(st[0].stats.calib, 1.0)
                    if format not in ['WAV']:
                        self.assertEquals(st[0].stats.starttime, start)
                        self.assertEquals(st[0].stats.endtime, start + 9.995)
                        self.assertEquals(st[0].stats.delta, 0.005)
                        self.assertEquals(st[0].stats.sampling_rate, 200.0)
                    # network/station/location/channel codes
                    if format in ['Q', 'SH_ASC', 'GSE2']:
                        # no network or location code in Q, SH_ASC, GSE2
                        self.assertEquals(st[0].id, ".MANZ1..EHE")
                    elif format not in ['WAV']:
                        self.assertEquals(st[0].id, "BW.MANZ1.00.EHE")
                    # remove temporary files
                    # XXX: temporary SAC file is locked on Windows ?!?
                    try:
                        os.remove(outfile)
                    except:
                        pass
                    if format == 'Q':
                        os.remove(outfile[:-4] + '.QBN')
                        os.remove(outfile[:-4])

    def test_read_thread_safe(self):
        """
        Tests for race conditions. Reading n_threads (currently 30) times
        the same waveform file in parallel and compare the results which must
        be all the same.
        """
        data = np.arange(0, 20000)
        start = UTCDateTime(2009, 1, 13, 12, 1, 2, 999000)
        formats = _getPlugins('obspy.plugin.waveform', 'writeFormat')
        for format in formats:
            # XXX: skip SEGY for now as it needs some special headers
            if 'SEGY' in format:
                continue

            dt = np.dtype("int")
            if format in ('MSEED', 'GSE2'):
                dt = "int32"
            tr = Trace(data=data.astype(dt))
            tr.stats.network = "BW"
            tr.stats.station = "MANZ1"
            tr.stats.location = "00"
            tr.stats.channel = "EHE"
            tr.stats.calib = 0.999999
            tr.stats.delta = 0.005
            tr.stats.starttime = start
            # create waveform file with given format and byte order
            outfile = NamedTemporaryFile().name
            tr.write(outfile, format=format)
            if format == 'Q':
                outfile += '.QHD'
            n_threads = 10
            streams = []
            def test_function(streams):
                st = read(outfile, format=format)
                streams.append(st)
            # Read the ten files at one and save the output in the just created
            # class.
            for _i in xrange(n_threads):
                thread = threading.Thread(target=test_function,
                                          args=(streams,))
                thread.start()
            # Loop until all threads are finished.
            start = time.time()
            while True:
                if threading.activeCount() == 1:
                    break
                # Avoid infinite loop and leave after 120 seconds 
                # such a long time is needed for debugging with valgrind
                elif time.time() - start >= 120:
                    msg = 'Not all threads finished!'
                    raise Warning(msg)
                    break
                else:
                    continue
            # Compare all values which should be identical and clean up files
            #for data in :
            #    np.testing.assert_array_equal(values, original)
            try:
                os.remove(outfile)
            except:
                pass
            if format == 'Q':
                os.remove(outfile[:-4] + '.QBN')
                os.remove(outfile[:-4])

    def test_issue193_noncontiguous(self):
        """
        Test if non-contiguous array is written correctly.
        """
        warnings.filterwarnings("ignore", "Detected non contiguous data")
        # test all plugins with both read and write method
        formats_write = set(_getPlugins('obspy.plugin.waveform', 'writeFormat'))
        formats_read = set(_getPlugins('obspy.plugin.waveform', 'readFormat'))
        formats = set.intersection(formats_write, formats_read)
        # mseed will raise exception for int64 data, thus use int32 only
        data = np.arange(10, dtype='int32')
        # make array non-contiguous
        data = data[::2]
        tr = Trace(data=data)
        for format in formats:
            # XXX: skip SEGY for now as it needs some special headers
            if 'SEGY' in format:
                continue
            tempfile = NamedTemporaryFile().name
            tr.write(tempfile, format)
            if format == "Q":
                tempfile = tempfile + ".QHD"
            tr_test = read(tempfile, format)[0]
            # clean up
            os.remove(tempfile)
            if format == 'Q':
                os.remove(tempfile[:-4] + '.QBN')
                os.remove(tempfile[:-4])
            np.testing.assert_array_equal(tr.data, tr_test.data)


def suite():
    return unittest.makeSuite(WaveformPluginsTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
