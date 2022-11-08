# -*- coding: utf-8 -*-
import io
import os
import threading
import time
import warnings
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import obspy
from obspy import Trace, read
from obspy.io.mseed.core import _write_mseed
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.base import (NamedTemporaryFile, _get_entry_points,
                                  DEFAULT_MODULES, WAVEFORM_ACCEPT_BYTEORDER)
from obspy.core.util.misc import buffered_load_entry_point, _ENTRY_POINT_CACHE


def _get_default_eps(group, subgroup=None):
    eps = _get_entry_points(group, subgroup=subgroup)
    eps = {ep: f for ep, f in eps.items()
           if any(m in f.module_name for m in DEFAULT_MODULES)}
    return eps


class TestWaveformPlugins:
    """
    Test suite for all waveform plug-ins.
    """
    longMessage = True

    def test_raise_on_empty_file(self):
        """
        Test case ensures that empty files do raise warnings.
        """
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # create empty file
            open(tmpfile, 'wb').close()
            formats_ep = _get_default_eps('obspy.plugin.waveform',
                                          'readFormat')
            # using format keyword
            for ep in formats_ep.values():
                is_format = buffered_load_entry_point(
                    ep.dist.key, 'obspy.plugin.waveform.' + ep.name,
                    'isFormat')
                assert not False, is_format(tmpfile)

    def test_read_and_write(self):
        """
        Tests read and write methods for all waveform plug-ins.
        """
        np.random.seed(1234)
        data = np.random.randint(-500, 500, 2000)
        formats = _get_default_eps('obspy.plugin.waveform', 'writeFormat')
        for format in formats:
            start = UTCDateTime(2009, 1, 13, 12, 1, 2, 999000)
            # XXX: skip SEGY and SU formats for now as they need some special
            # headers.
            if format in ['SEGY', 'SU', 'SEG2']:
                continue
            elif format in ['GCF']:
                # XXX: GCF format does not support fractional start time for
                # sampling rates <= 250 Hz, hence set to integer sec start
                start = UTCDateTime(2009, 1, 13, 12, 1, 3)
            for native_byteorder in ['<', '>']:
                for byteorder in (['<', '>', '='] if format in
                                  WAVEFORM_ACCEPT_BYTEORDER else [None]):
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
                    tr.stats.delta = 0.25
                    tr.stats.starttime = start
                    # create waveform file with given format and byte order
                    with NamedTemporaryFile() as tf:
                        outfile = tf.name
                        if byteorder is None:
                            tr.write(outfile, format=format)
                        else:
                            tr.write(outfile, format=format,
                                     byteorder=byteorder)
                        if format == 'Q':
                            outfile += '.QHD'
                        # read in again using auto detection
                        st = read(outfile)
                        assert len(st) == 1
                        assert st[0].stats._format == format
                        # read in using format argument
                        st = read(outfile, format=format)
                        assert len(st) == 1
                        assert st[0].stats._format == format
                        # read in using a BytesIO instances, skip Q files as
                        # it needs multiple files
                        if format not in ['Q']:
                            # file handler without format
                            with open(outfile, 'rb') as fp:
                                st = read(fp)
                            assert len(st) == 1
                            assert st[0].stats._format == format
                            # file handler with format
                            with open(outfile, 'rb') as fp:
                                st = read(fp, format=format)
                            assert len(st) == 1
                            assert st[0].stats._format == format
                            # BytesIO without format
                            with open(outfile, 'rb') as fp:
                                temp = io.BytesIO(fp.read())
                            st = read(temp)
                            assert len(st) == 1
                            assert st[0].stats._format == format
                            # BytesIO with format
                            with open(outfile, 'rb') as fp:
                                temp = io.BytesIO(fp.read())
                            st = read(temp, format=format)
                            assert len(st) == 1
                            assert st[0].stats._format == format
                            # BytesIO with an offset (additional data in front
                            # but file pointer at right position in file), with
                            # and without autodetection
                            for autodetect in (format, None):
                                temp.seek(0)
                                temp2 = io.BytesIO()
                                dummy_bytes = b'123456'
                                temp2.write(dummy_bytes)
                                temp2.write(temp.read())
                                temp2.seek(len(dummy_bytes))
                                st = read(outfile, format=autodetect)
                                assert len(st) == 1
                                assert st[0].stats._format == format
                        # Q files consist of two files - deleting additional
                        # file
                        if format == 'Q':
                            os.remove(outfile[:-4] + '.QBN')
                            os.remove(outfile[:-4] + '.QHD')
                    # check byte order
                    if format == 'SAC':
                        # SAC format preserves byteorder on writing
                        assert st[0].data.dtype.byteorder \
                                        in ('=', byteorder)
                    else:
                        assert st[0].data.dtype.byteorder == '='
                    # check meta data
                    # some formats do not contain a calibration factor
                    if format not in ['MSEED', 'WAV', 'TSPAIR', 'SLIST', 'AH',
                                      'GCF']:
                        assert round(abs(st[0].stats.calib-0.199999), 5) == 0
                    else:
                        assert st[0].stats.calib == 1.0
                    if format not in ['WAV']:
                        assert st[0].stats.starttime == start
                        assert st[0].stats.delta == 0.25
                        assert st[0].stats.endtime == start + 499.75
                        assert st[0].stats.sampling_rate == 4.0

                    # network/station/location/channel codes
                    if format in ['GCF']:
                        # no network, station or location code in GCF, however
                        #  first 4 characters in station code will be set in
                        #  current implementation if stream_id is not set.
                        #  Further no bandcode or instrumentcode, if not set
                        #  by argument in call to read function both default
                        #  to H
                        assert st[0].id == ".MANZ..HHE"
                    elif format in ['Q', 'SH_ASC', 'AH']:
                        # no network or location code in Q, SH_ASC
                        assert st[0].id == ".MANZ1..EHE"
                    elif format == "GSE2":
                        # no location code in GSE2
                        assert st[0].id == "BW.MANZ1..EHE"
                    elif format not in ['WAV']:
                        assert st[0].id == "BW.MANZ1.00.EHE"

    def test_is_format(self):
        """
        Tests all isFormat methods against all data test files from the other
        modules for false positives.
        """
        known_false = [
            os.path.join('seisan', 'tests', 'data',
                         '2011-09-06-1311-36S.A1032_001BH_Z.mseed'),
            os.path.join('seisan', 'tests', 'data',
                         'D1360930.203.mseed'),
            os.path.join('seisan', 'tests', 'data',
                         '2005-07-23-1452-04S.CER___030.mseed'),
            os.path.join('seisan', 'tests', 'data',
                         '9701-30-1048-54S.MVO_21_1.ascii'),
            os.path.join('core', 'tests', 'data',
                         'IU_ULN_00_LH1_2015-07-18T02.mseed'),
            # That file is not in obspy.io.mseed as it is used to test an
            # issue with the uncompress_data() decorator.
            os.path.join('core', 'tests', 'data',
                         'tarfile_impostor.mseed'),
            # these files are not in /mseed because they hold the data to
            # validate the read output of the reftek file
            os.path.join('io', 'reftek', 'tests', 'data',
                         '2015282_225051_0ae4c_1_1.msd'),
            os.path.join('io', 'reftek', 'tests', 'data',
                         '2015282_225051_0ae4c_1_2.msd'),
            os.path.join('io', 'reftek', 'tests', 'data',
                         '2015282_225051_0ae4c_1_3.msd'),
            os.path.join('core', 'tests', 'data', 'ffbx_unrotated_gaps.mseed'),
            os.path.join('core', 'tests', 'data', 'ffbx_rotated.slist'),
            os.path.join('io', 'ascii', 'tests', 'data',
                         'miniseed_record.mseed'),
        ]
        formats_ep = _get_default_eps('obspy.plugin.waveform', 'isFormat')
        formats = list(formats_ep.values())
        # Get all the test directories.
        paths = {}
        all_paths = []
        # dont rely on f.dist.location to lookup install path, as this seems to
        # recently not be able to follow a pip editable install correctly. this
        # seems safe unless custom installed plugins come into play, but we can
        # not test these here properly anyway
        install_dir = Path(obspy.__file__).parent.parent
        for f in formats:
            path = os.path.join(install_dir,
                                *f.module_name.split('.')[:-1])
            path = os.path.join(path, 'tests', 'data')
            all_paths.append(path)
            if os.path.exists(path):
                paths[f.name] = path

        msg = 'Test data directories do not exist:\n    '
        assert len(paths) > 0, msg + '\n    '.join(all_paths)
        # Collect all false positives.
        false_positives = []
        # Big loop over every format.
        for format in formats:
            # search isFormat for given entry point
            is_format = buffered_load_entry_point(
                format.dist.key, 'obspy.plugin.waveform.' + format.name,
                'isFormat')
            for f, path in paths.items():
                if format.name in paths and paths[f] == paths[format.name]:
                    continue
                # Collect all files found.
                filelist = []
                # Walk every path.
                for directory, _, files in os.walk(path):
                    filelist.extend([os.path.join(directory, _i) for _i in
                                     files])
                for file in filelist:
                    if any([n in file for n in known_false]):
                        continue
                    if is_format(file) is True:  # pragma: no cover
                        false_positives.append((format.name, file))
        # Use try except to produce a meaningful error message.
        try:
            assert len(false_positives) == 0
        except Exception:  # pragma: no cover
            msg = 'False positives for isFormat:\n'
            msg += '\n'.join(['\tFormat %s: %s' % (_i[0], _i[1]) for _i in
                              false_positives])
            raise Exception(msg)

    def test_read_thread_safe(self):
        """
        Tests for race conditions. Reading n_threads (currently 30) times
        the same waveform file in parallel and compare the results which must
        be all the same.
        """
        data = np.arange(0, 500)
        formats = _get_default_eps('obspy.plugin.waveform', 'writeFormat')
        for format in formats:
            start = UTCDateTime(2009, 1, 13, 12, 1, 2, 999000)
            # XXX: skip SEGY and SU formats for now as they need some special
            # headers. Also skip GCF as format does not permitt fractional
            # start time for sampling rates < 250
            if format in ['SEGY', 'SU', 'SEG2']:
                continue
            elif format in ['GCF']:
                # XXX: GCF format does not support fractional for sampling
                # rates <= 250 Hz
                start = UTCDateTime(2009, 1, 13, 12, 1, 3)
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

                def test_functions(streams, cond):
                    st = read(outfile, format=format)
                    streams.append(st)
                    with cond:
                        cond.notify()
                # Read the ten files at one and save the output in the just
                # created class.
                our_threads = []
                for _i in range(n_threads):
                    thread = threading.Thread(target=test_functions,
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

    @pytest.mark.filterwarnings('ignore:Detected non contiguous data array')
    def test_issue_193(self):
        """
        Test for issue #193: if non-contiguous array is written correctly.
        """
        # test all plugins with both read and write method
        formats_write = \
            set(_get_default_eps('obspy.plugin.waveform', 'writeFormat'))
        formats_read = \
            set(_get_default_eps('obspy.plugin.waveform', 'readFormat'))
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

    def test_read_gzip2_file(self):
        """
        Tests reading gzip compressed waveforms.
        """
        path = os.path.dirname(__file__)
        ascii_path = os.path.join(path, "..", "..", "io", "ascii",
                                  "tests", "data")
        st1 = read(os.path.join(ascii_path, 'tspair.ascii.gz'))
        st2 = read(os.path.join(ascii_path, 'tspair.ascii'))
        assert st1 == st2

    def test_read_bzip2_file(self):
        """
        Tests reading bzip2 compressed waveforms.
        """
        path = os.path.dirname(__file__)
        ascii_path = os.path.join(path, "..", "..", "io", "ascii",
                                  "tests", "data")
        st1 = read(os.path.join(ascii_path, 'slist.ascii.bz2'))
        st2 = read(os.path.join(ascii_path, 'slist.ascii'))
        assert st1 == st2

    def test_read_tar_archive(self):
        """
        Tests reading tar compressed waveforms.
        """
        path = os.path.dirname(__file__)
        ascii_path = os.path.join(path, "..", "..", "io", "ascii",
                                  "tests", "data")
        # tar
        st1 = read(os.path.join(path, "data", "test.tar"))
        st2 = read(os.path.join(ascii_path, "slist.ascii"))
        assert st1 == st2
        # tar.gz
        st1 = read(os.path.join(path, "data", "test.tar.gz"))
        st2 = read(os.path.join(ascii_path, "slist.ascii"))
        assert st1 == st2
        # tar.bz2
        st1 = read(os.path.join(path, "data", "test.tar.bz2"))
        st2 = read(os.path.join(ascii_path, "slist.ascii"))
        assert st1 == st2
        # tgz
        st1 = read(os.path.join(path, "data", "test.tgz"))
        st2 = read(os.path.join(ascii_path, "slist.ascii"))
        assert st1 == st2

    def test_read_zip_archive(self):
        """
        Tests reading zip compressed waveforms.
        """
        path = os.path.dirname(__file__)
        ascii_path = os.path.join(path, "..", "..", "io", "ascii",
                                  "tests", "data")
        st1 = read(os.path.join(path, 'data', 'test.zip'))
        st2 = read(os.path.join(ascii_path, 'slist.ascii'))
        assert st1 == st2

    def test_raise_on_unknown_format(self):
        """
        Test case for issue #338:
        """
        with NamedTemporaryFile() as tf:
            tmpfile = tf.name
            # create empty file
            open(tmpfile, 'wb').close()
            # using format keyword
            with pytest.raises(TypeError):
                read(tmpfile)

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
            set(_get_default_eps('obspy.plugin.waveform', 'writeFormat'))
        formats_read = \
            set(_get_default_eps('obspy.plugin.waveform', 'readFormat'))
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
            assert str(st) == str(st_deepcopy), msg

    def test_auto_file_format_during_writing(self):
        """
        The file format is either determined by directly specifying the
        format or deduced from the filename. The former overwrites the latter.
        """
        # Get format name and name of the write function.
        formats = [(key, value.module_name) for key, value in
                   _get_default_eps('obspy.plugin.waveform',
                                    'writeFormat').items()
                   # Only test plugins that are actually part of ObsPy.
                   if value.dist.key == "obspy"]

        # Test for stream as well as for trace.
        stream_trace = [read(), read()[0]]

        # get mseed cache name and mseed function
        mseed_name = "obspy/obspy.plugin.waveform.MSEED/writeFormat"
        mseed_func = _ENTRY_POINT_CACHE.get(mseed_name, _write_mseed)

        for suffix, module_name in formats:
            # get a list of dist, group, name.
            entry_point_list = ["obspy", "obspy.plugin.waveform.%s" % suffix,
                                "writeFormat"]
            # load entry point to make sure it is in the cache.
            buffered_load_entry_point(*entry_point_list)
            # get the cache name for monkey patching.
            entry_point_name = '/'.join(entry_point_list)
            # For stream and trace.
            for obj in stream_trace:
                # Various versions of the suffix.
                for s in [suffix.capitalize(), suffix.lower(), suffix.upper()]:
                    # create a mock function and patch the entry point cache.
                    write_func = _ENTRY_POINT_CACHE[entry_point_name]
                    mocked_func = mock.MagicMock(write_func)
                    mock_dict = {entry_point_name: mocked_func}
                    with mock.patch.dict(_ENTRY_POINT_CACHE, mock_dict):
                        obj.write("temp." + s)
                    # Make sure the fct has actually been called.
                    assert mocked_func.call_count == 1

                    # Specifying the format name should overwrite this.
                    mocked_mseed_func = mock.MagicMock(mseed_func)
                    mseed_mock_dict = {mseed_name: mocked_mseed_func}
                    with mock.patch.dict(_ENTRY_POINT_CACHE, mseed_mock_dict):
                        obj.write("temp." + s, format="mseed")
                    assert mocked_mseed_func.call_count == 1
                    assert mocked_func.call_count == 1

        # An unknown suffix should raise.
        with pytest.raises(ValueError):
            for obj in stream_trace:
                obj.write("temp.random_suffix")

    def test_reading_tarfile_impostor(self):
        """
        Tests that a file, that by chance is interpreted as a valid tar file
        can be read by ObsPy and is not treated as a tar file.

        See #1436.
        """
        st = read("/path/to/tarfile_impostor.mseed")
        assert st[0].id == "10.864.1B.004"

    def test_read_invalid_filename(self):
        """
        Tests that we get a sane error message when calling read()
        with a filename that doesn't exist
        """
        doesnt_exist = 'dsfhjkfs'
        for i in range(10):
            if os.path.exists(doesnt_exist):
                doesnt_exist += doesnt_exist
                continue
            break
        else:
            self.fail('unable to get invalid file path')

        exception_msg = "No such file or directory: '{}'"

        formats = _get_entry_points(
            'obspy.plugin.catalog', 'readFormat').keys()
        # try read_inventory() with invalid filename for all registered read
        # plugins and also for filetype autodiscovery
        formats = [None] + list(formats)
        for format in formats:
            msg = exception_msg.format(doesnt_exist)
            with pytest.raises(FileNotFoundError, match=msg):
                read(doesnt_exist, format=format)
