#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions for testing the obspy.io.nordic functions
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import inspect
import io
import os
import unittest
import warnings
from itertools import cycle
import numpy as np

from obspy import read_events, Catalog, UTCDateTime, read
from obspy.core.event import (
    Pick, WaveformStreamID, Arrival, Amplitude, Event, Origin, Magnitude,
    OriginQuality, EventDescription, CreationInfo, OriginUncertainty,
    ConfidenceEllipsoid, QuantityError, FocalMechanism, MomentTensor,
    NodalPlane, NodalPlanes, ResourceIdentifier, Tensor)
from obspy.core.util.base import NamedTemporaryFile
from obspy.core.util.misc import TemporaryWorkingDirectory
from obspy.core.util.testing import ImageComparison

from obspy.io.nordic import NordicParsingError
from obspy.io.nordic.core import (
    _is_sfile, read_spectral_info, read_nordic, readwavename, blanksfile,
    _write_nordic, nordpick, readheader, _readheader, write_select)
from obspy.io.nordic.utils import (
    _int_conv, _float_conv, _str_conv, _nortoevmag, _evmagtonor,
    _get_line_tags)
from obspy.io.nordic.ellipse import Ellipse


class TestNordicMethods(unittest.TestCase):
    """
    Test suite for nordic io operations.
    """
    def setUp(self):
        self.path = os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe())))
        self.testing_path = os.path.join(self.path, "data")

    def test_read_write(self):
        """
        Function to test the read and write capabilities of sfile_util.
        """
        # Set-up a test event
        test_event = full_test_event()
        # Sort the magnitudes - they are sorted on writing and we need to check
        # like-for-like
        test_event.magnitudes.sort(key=lambda obj: obj['mag'], reverse=True)
        # Add the event to a catalogue which can be used for QuakeML testing
        test_cat = Catalog()
        test_cat += test_event
        # Check the read-write s-file functionality
        with TemporaryWorkingDirectory():
            with warnings.catch_warnings():
                # Evaluation mode mapping warning
                warnings.simplefilter('ignore', UserWarning)
                sfile = _write_nordic(
                    test_cat[0], filename=None, userid='TEST', evtype='L',
                    outdir='.', wavefiles='test', explosion=True,
                    overwrite=True)
            self.assertEqual(readwavename(sfile), ['test'])
            read_cat = Catalog()
            # raises "UserWarning: AIN in header, currently unsupported"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                read_cat += read_nordic(sfile)
        read_ev = read_cat[0]
        test_ev = test_cat[0]
        for read_pick, test_pick in zip(read_ev.picks, test_ev.picks):
            self.assertEqual(read_pick.time, test_pick.time)
            self.assertEqual(read_pick.backazimuth, test_pick.backazimuth)
            self.assertEqual(read_pick.onset, test_pick.onset)
            self.assertEqual(read_pick.phase_hint, test_pick.phase_hint)
            if test_pick.polarity == "undecidable":
                self.assertIsNone(read_pick.polarity)
            elif read_pick.polarity == "undecidable":
                self.assertIsNone(test_pick.polarity)
            else:
                self.assertEqual(read_pick.polarity, test_pick.polarity)
            self.assertEqual(read_pick.waveform_id.station_code,
                             test_pick.waveform_id.station_code)
            self.assertEqual(read_pick.waveform_id.channel_code[-1],
                             test_pick.waveform_id.channel_code[-1])
        # assert read_ev.origins[0].resource_id ==\
        #     test_ev.origins[0].resource_id
        self.assertEqual(read_ev.origins[0].time,
                         test_ev.origins[0].time)
        # Note that time_residual_RMS is not a quakeML format
        self.assertEqual(read_ev.origins[0].longitude,
                         test_ev.origins[0].longitude)
        self.assertEqual(read_ev.origins[0].latitude,
                         test_ev.origins[0].latitude)
        self.assertEqual(read_ev.origins[0].depth,
                         test_ev.origins[0].depth)
        self.assertEqual(read_ev.magnitudes[0].mag,
                         test_ev.magnitudes[0].mag)
        self.assertEqual(read_ev.magnitudes[1].mag,
                         test_ev.magnitudes[1].mag)
        self.assertEqual(read_ev.magnitudes[2].mag,
                         test_ev.magnitudes[2].mag)
        self.assertEqual(read_ev.magnitudes[0].creation_info,
                         test_ev.magnitudes[0].creation_info)
        self.assertEqual(read_ev.magnitudes[1].creation_info,
                         test_ev.magnitudes[1].creation_info)
        self.assertEqual(read_ev.magnitudes[2].creation_info,
                         test_ev.magnitudes[2].creation_info)
        self.assertEqual(read_ev.magnitudes[0].magnitude_type,
                         test_ev.magnitudes[0].magnitude_type)
        self.assertEqual(read_ev.magnitudes[1].magnitude_type,
                         test_ev.magnitudes[1].magnitude_type)
        self.assertEqual(read_ev.magnitudes[2].magnitude_type,
                         test_ev.magnitudes[2].magnitude_type)
        self.assertEqual(read_ev.event_descriptions,
                         test_ev.event_descriptions)
        # assert read_ev.amplitudes[0].resource_id ==\
        #     test_ev.amplitudes[0].resource_id
        self.assertEqual(read_ev.amplitudes[0].period,
                         test_ev.amplitudes[0].period)
        self.assertEqual(read_ev.amplitudes[0].snr,
                         test_ev.amplitudes[0].snr)
        self.assertEqual(read_ev.amplitudes[2].period,
                         test_ev.amplitudes[2].period)
        self.assertEqual(read_ev.amplitudes[2].snr,
                         test_ev.amplitudes[2].snr)
        # Check coda magnitude pick
        # Resource ids get overwritten because you can't have two the same in
        # memory
        # self.assertEqual(read_ev.amplitudes[1].resource_id,
        #                  test_ev.amplitudes[1].resource_id)
        self.assertEqual(read_ev.amplitudes[1].type,
                         test_ev.amplitudes[1].type)
        self.assertEqual(read_ev.amplitudes[1].unit,
                         test_ev.amplitudes[1].unit)
        self.assertEqual(read_ev.amplitudes[1].generic_amplitude,
                         test_ev.amplitudes[1].generic_amplitude)
        # Resource ids get overwritten because you can't have two the same in
        # memory
        # self.assertEqual(read_ev.amplitudes[1].pick_id,
        #                  test_ev.amplitudes[1].pick_id)
        self.assertEqual(read_ev.amplitudes[1].waveform_id.station_code,
                         test_ev.amplitudes[1].waveform_id.station_code)
        self.assertEqual(read_ev.amplitudes[1].waveform_id.channel_code,
                         test_ev.amplitudes[1].waveform_id.channel_code[0] +
                         test_ev.amplitudes[1].waveform_id.channel_code[-1])
        self.assertEqual(read_ev.amplitudes[1].magnitude_hint,
                         test_ev.amplitudes[1].magnitude_hint)
        # snr is not supported in s-file
        # self.assertEqual(read_ev.amplitudes[1].snr,
        #                  test_ev.amplitudes[1].snr)
        self.assertEqual(read_ev.amplitudes[1].category,
                         test_ev.amplitudes[1].category)

    def test_fail_writing(self):
        """
        Test a deliberate fail.
        """
        test_event = full_test_event()
        # Add the event to a catalogue which can be used for QuakeML testing
        test_cat = Catalog()
        test_cat += test_event
        test_ev = test_cat[0]
        test_cat.append(full_test_event())
        with self.assertRaises(NordicParsingError):
            # Raises error due to multiple events in catalog
            with warnings.catch_warnings():
                # Evaluation mode mapping warning
                warnings.simplefilter('ignore', UserWarning)
                _write_nordic(test_cat, filename=None, userid='TEST',
                              evtype='L', outdir='.',
                              wavefiles='test', explosion=True,
                              overwrite=True)
        with self.assertRaises(NordicParsingError):
            # Raises error due to too long userid
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                _write_nordic(test_ev, filename=None, userid='TESTICLE',
                              evtype='L', outdir='.',
                              wavefiles='test', explosion=True,
                              overwrite=True)
        with self.assertRaises(NordicParsingError):
            # Raises error due to unrecognised event type
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                _write_nordic(test_ev, filename=None, userid='TEST',
                              evtype='U', outdir='.',
                              wavefiles='test', explosion=True,
                              overwrite=True)
        with self.assertRaises(NordicParsingError):
            # Raises error due to no output directory
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                _write_nordic(test_ev, filename=None, userid='TEST',
                              evtype='L', outdir='albatross',
                              wavefiles='test', explosion=True,
                              overwrite=True)
        invalid_origin = test_ev.copy()

        invalid_origin.origins = []
        with self.assertRaises(NordicParsingError):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                _write_nordic(invalid_origin, filename=None, userid='TEST',
                              evtype='L', outdir='.', wavefiles='test',
                              explosion=True, overwrite=True)
        invalid_origin = test_ev.copy()
        invalid_origin.origins[0].time = None
        with self.assertRaises(NordicParsingError):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                _write_nordic(invalid_origin, filename=None, userid='TEST',
                              evtype='L', outdir='.', wavefiles='test',
                              explosion=True, overwrite=True)
        # Write a near empty origin
        valid_origin = test_ev.copy()
        valid_origin.origins[0].latitude = None
        valid_origin.origins[0].longitude = None
        valid_origin.origins[0].depth = None
        with NamedTemporaryFile() as tf:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                _write_nordic(
                    valid_origin, filename=tf.name, userid='TEST',
                    evtype='L', outdir='.', wavefiles='test',
                    explosion=True, overwrite=True)
            self.assertTrue(os.path.isfile(tf.name))

    def test_blanksfile(self):
        st = read()
        with TemporaryWorkingDirectory():
            testing_path = 'Temporary_wavefile'
            st.write(testing_path, format='MSEED')
            sfile = blanksfile(testing_path, 'L', 'TEST', overwrite=True)
            self.assertTrue(os.path.isfile(sfile))
            os.remove(sfile)
            sfile = blanksfile(testing_path, 'L', 'TEST', overwrite=True,
                               evtime=UTCDateTime())
            self.assertTrue(os.path.isfile(sfile))
            os.remove(sfile)
            with self.assertRaises(NordicParsingError):
                # No wavefile
                blanksfile('albert', 'L', 'TEST', overwrite=True)
            with self.assertRaises(NordicParsingError):
                # USER ID too long
                blanksfile(testing_path, 'L', 'TESTICLE', overwrite=True)
            with self.assertRaises(NordicParsingError):
                # Unknown event type
                blanksfile(testing_path, 'U', 'TEST', overwrite=True)
            # Check that it breaks when writing multiple versions
            sfiles = []
            for _i in range(10):
                # raises UserWarning: Desired sfile exists, will not overwrite
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    sfiles.append(blanksfile(testing_path, 'L', 'TEST'))
            with self.assertRaises(NordicParsingError):
                # raises UserWarning: Desired sfile exists, will not overwrite
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    blanksfile(testing_path, 'L', 'TEST')
            for sfile in sfiles:
                self.assertTrue(os.path.isfile(sfile))

    def test_write_empty(self):
        """
        Function to check that writing a blank event works as it should.
        """
        test_event = Event()
        with self.assertRaises(NordicParsingError):
            _write_nordic(test_event, filename=None, userid='TEST', evtype='L',
                          outdir='.', wavefiles='test')
        test_event.origins.append(Origin())
        with self.assertRaises(NordicParsingError):
            _write_nordic(test_event, filename=None, userid='TEST', evtype='L',
                          outdir='.', wavefiles='test')
        test_event.origins[0].time = UTCDateTime()
        with TemporaryWorkingDirectory():
            test_sfile = _write_nordic(test_event, filename=None,
                                       userid='TEST', evtype='L', outdir='.',
                                       wavefiles='test')
            self.assertTrue(os.path.isfile(test_sfile))

    def test_read_empty_header(self):
        """
        Function to check a known issue, empty header info S-file: Bug found \
        by Dominic Evanzia.
        """
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            test_event = read_nordic(os.path.join(self.testing_path,
                                                  'Sfile_no_location'))[0]
        self.assertFalse(test_event.origins[0].latitude)
        self.assertFalse(test_event.origins[0].longitude)
        self.assertFalse(test_event.origins[0].depth)

    def test_read_extra_header(self):
        testing_path = os.path.join(self.testing_path, 'Sfile_extra_header')
        not_extra_header = os.path.join(self.testing_path,
                                        '01-0411-15L.S201309')
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            test_event = read_nordic(testing_path)[0]
            header_event = read_nordic(not_extra_header)[0]
        self.assertEqual(len(header_event.origins), 2)
        self.assertEqual(test_event.origins[0].time,
                         header_event.origins[0].time)
        self.assertEqual(test_event.origins[0].latitude,
                         header_event.origins[0].latitude)
        self.assertEqual(test_event.origins[0].longitude,
                         header_event.origins[0].longitude)
        self.assertEqual(test_event.origins[0].depth,
                         header_event.origins[0].depth)

    def test_header_mapping(self):
        # Raise "UserWarning: Lines of type I..."
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            head_1 = readheader(os.path.join(self.testing_path,
                                             '01-0411-15L.S201309'))
        with open(os.path.join(self.testing_path,
                               '01-0411-15L.S201309'), 'r') as f:
            # raises "UserWarning: AIN in header, currently unsupported"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                tagged_lines = _get_line_tags(f=f)
                head_2 = _readheader(head_lines=tagged_lines['1'])
        _assert_similarity(head_1, head_2)

    def test_missing_header(self):
        # Check that a suitable error is raised
        with self.assertRaises(NordicParsingError):
            # Raises AIN warning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                readheader(os.path.join(self.testing_path, 'Sfile_no_header'))

    def test_reading_string_io(self):
        filename = os.path.join(self.testing_path, '01-0411-15L.S201309')
        with open(filename, "rt") as fh:
            file_object = io.StringIO(fh.read())

        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            cat = read_events(file_object)
            file_object.close()

            ref_cat = read_events(filename)
            _assert_similarity(cat[0], ref_cat[0])

    def test_reading_bytes_io(self):
        filename = os.path.join(self.testing_path, '01-0411-15L.S201309')
        with open(filename, "rb") as fh:
            file_object = io.BytesIO(fh.read())

        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            cat = read_events(file_object)
            file_object.close()

            ref_cat = read_events(filename)
            _assert_similarity(cat[0], ref_cat[0])

    def test_corrupt_header(self):
        filename = os.path.join(self.testing_path, '01-0411-15L.S201309')
        f = open(filename, 'r')
        with NamedTemporaryFile(suffix='.sfile') as tmp_file:
            fout = open(tmp_file.name, 'w')
            for line in f:
                fout.write(line[0:78])
            f.close()
            fout.close()
            with self.assertRaises(NordicParsingError):
                readheader(tmp_file.name)

    def test_multi_writing(self):
        event = full_test_event()
        # Try to write the same event multiple times, but not overwrite
        sfiles = []
        with TemporaryWorkingDirectory():
            for _i in range(59):
                with warnings.catch_warnings():
                    # Evaluation mode mapping warning
                    warnings.simplefilter('ignore', UserWarning)
                    sfiles.append(_write_nordic(event=event, filename=None,
                                                overwrite=False))
            with self.assertRaises(NordicParsingError):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    _write_nordic(event=event, filename=None, overwrite=False)

    def test_mag_conv(self):
        """
        Check that we convert magnitudes as we should!
        """
        magnitude_map = [
            ('L', 'ML'), ('B', 'mB'), ('S', 'MS'), ('W', 'MW'), ('G', 'MbLg'),
            ('C', 'Mc'), ('s', 'Ms')]
        for magnitude in magnitude_map:
            self.assertEqual(magnitude[0], _evmagtonor(magnitude[1]))
            self.assertEqual(_nortoevmag(magnitude[0]), magnitude[1])

    def test_str_conv(self):
        """
        Test the simple string conversions.
        """
        self.assertEqual(_int_conv('albert'), None)
        self.assertEqual(_float_conv('albert'), None)
        self.assertEqual(_str_conv('albert'), 'albert')
        self.assertEqual(_int_conv('1'), 1)
        self.assertEqual(_float_conv('1'), 1.0)
        self.assertEqual(_str_conv(1), '1')
        self.assertEqual(_int_conv('1.0256'), None)
        self.assertEqual(_float_conv('1.0256'), 1.0256)
        self.assertEqual(_str_conv(1.0256), '1.0256')

    def test_read_wavename(self):
        testing_path = os.path.join(self.testing_path, '01-0411-15L.S201309')
        wavefiles = readwavename(testing_path)
        self.assertEqual(len(wavefiles), 1)
        # Check that read_nordic reads wavname when return_wavnames=True
        cat, wavefiles = read_nordic(testing_path, return_wavnames=True)
        self.assertEqual(wavefiles, [['2013-09-01-0410-35.DFDPC_024_00']])
        # Test that full paths are handled
        test_event = full_test_event()
        # Add the event to a catalogue which can be used for QuakeML testing
        test_cat = Catalog()
        test_cat += test_event
        # Check the read-write s-file functionality
        with TemporaryWorkingDirectory():
            with warnings.catch_warnings():
                # Evaluation mode mapping warning
                warnings.simplefilter('ignore', UserWarning)
                sfile = _write_nordic(
                    test_cat[0], filename=None, userid='TEST', evtype='L',
                    outdir='.', wavefiles=['walrus/test'], explosion=True,
                    overwrite=True)
            self.assertEqual(readwavename(sfile), ['test'])
        # Check that multiple wavefiles are read properly
        with TemporaryWorkingDirectory():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                sfile = _write_nordic(
                    test_cat[0], filename=None, userid='TEST', evtype='L',
                    outdir='.', wavefiles=['walrus/test', 'albert'],
                    explosion=True, overwrite=True)
            self.assertEqual(readwavename(sfile), ['test', 'albert'])

    def test_read_event(self):
        """
        Test the wrapper.
        """
        testing_path = os.path.join(self.testing_path, '01-0411-15L.S201309')
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            event = read_nordic(testing_path)[0]
        self.assertEqual(len(event.origins), 2)
        self.assertEqual(len(event.picks), 17)

    def test_read_latin1(self):
        """
        Check that we can read dos formatted, latin1 encoded files.
        """
        with warnings.catch_warnings():
            # Lots of unsupported line warnings
            warnings.simplefilter('ignore', UserWarning)
            dos_file = os.path.join(self.testing_path, 'dos-file.sfile')
            self.assertTrue(_is_sfile(dos_file))
            event = readheader(dos_file)
            self.assertEqual(event.origins[0].latitude, 60.328)
            cat = read_events(dos_file)
            self.assertEqual(cat[0].origins[0].latitude, 60.328)
            wavefiles = readwavename(dos_file)
            self.assertEqual(wavefiles[0], "90121311.0851W41")
            spectral_info = read_spectral_info(dos_file)
            self.assertEqual(len(spectral_info.keys()), 10)
            self.assertEqual(spectral_info[('AVERAGE', '')]['stress_drop'],
                             27.7)
            with self.assertRaises(UnicodeDecodeError):
                readheader(dos_file, 'ASCII')

    def test_read_many_events(self):
        testing_path = os.path.join(self.testing_path, 'select.out')
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            catalog = read_nordic(testing_path)
        self.assertEqual(len(catalog), 50)

    def test_write_select(self):
        cat = read_events()
        with NamedTemporaryFile(suffix='.out') as tf:
            # raises "UserWarning: mb is not convertible"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                write_select(cat, filename=tf.name)
            self.assertTrue(_is_sfile(tf.name))
            with warnings.catch_warnings():
                # Type I warning
                warnings.simplefilter('ignore', UserWarning)
                cat_back = read_events(tf.name)
        for event_1, event_2 in zip(cat, cat_back):
            _assert_similarity(event_1=event_1, event_2=event_2)

    def test_write_plugin(self):
        cat = read_events()
        cat.append(full_test_event())
        with NamedTemporaryFile(suffix='.out') as tf:
            # raises UserWarning: mb is not convertible
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                cat.write(tf.name, format='nordic')
            # raises "UserWarning: AIN in header, currently unsupported"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                cat_back = read_events(tf.name)
            for event_1, event_2 in zip(cat, cat_back):
                _assert_similarity(event_1=event_1, event_2=event_2)

    def test_more_than_three_mags(self):
        cat = Catalog()
        cat += full_test_event()
        cat[0].magnitudes.append(Magnitude(
            mag=0.9, magnitude_type='MS', creation_info=CreationInfo('TES'),
            origin_id=cat[0].origins[0].resource_id))
        with NamedTemporaryFile(suffix='.out') as tf:
            # raises UserWarning: mb is not convertible
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                cat.write(tf.name, format='nordic')
            # raises "UserWarning: AIN in header, currently unsupported"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                cat_back = read_events(tf.name)
            for event_1, event_2 in zip(cat, cat_back):
                self.assertTrue(
                    len(event_1.magnitudes) == len(event_2.magnitudes))
                _assert_similarity(event_1, event_2)

    def test_inaccurate_picks(self):
        testing_path = os.path.join(self.testing_path, 'bad_picks.sfile')
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            cat = read_nordic(testing_path)
        pick_string = nordpick(cat[0])
        for pick in pick_string:
            self.assertEqual(len(pick), 80)

    def test_round_len(self):
        testing_path = os.path.join(self.testing_path, 'round_len_undef.sfile')
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            event = read_nordic(testing_path)[0]
        pick_string = nordpick(event)
        for pick in pick_string:
            self.assertEqual(len(pick), 80)

    def test_read_moment(self):
        """
        Test the reading of seismic moment from the s-file.
        """
        testing_path = os.path.join(self.testing_path, 'automag.out')
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            event = read_nordic(testing_path)[0]
        mag = [m for m in event.magnitudes if m.magnitude_type == 'MW']
        self.assertEqual(len(mag), 1)
        self.assertEqual(mag[0].mag, 0.7)

    def test_read_moment_info(self):
        """
        Test reading the info from spectral analysis.
        """
        testing_path = os.path.join(self.testing_path, 'automag.out')
        with warnings.catch_warnings():
            # Userwarning, type I
            warnings.simplefilter('ignore', UserWarning)
            spec_inf = read_spectral_info(testing_path)
        self.assertEqual(len(spec_inf), 5)
        # This should actually test that what we are reading in is correct.
        average = spec_inf[('AVERAGE', '')]
        check_av = {u'channel': '', u'corner_freq': 5.97, u'decay': 0.0,
                    u'moment': 12589254117.941662, u'moment_mag': 0.7,
                    u'source_radius': 0.231,
                    u'spectral_level': 0.3981071705534972,
                    u'station': 'AVERAGE', u'stress_drop': 0.006,
                    u'window_length': 1.6}
        for key in average.keys():
            if isinstance(average.get(key), str):
                self.assertEqual(average.get(key), check_av.get(key))
            else:
                self.assertEqual(round(average.get(key), 4),
                                 round(check_av.get(key), 4))

    def test_is_sfile(self):
        sfiles = ['01-0411-15L.S201309', 'automag.out', 'bad_picks.sfile',
                  'round_len_undef.sfile', 'Sfile_extra_header',
                  'Sfile_no_location']
        for sfile in sfiles:
            self.assertTrue(_is_sfile(os.path.join(self.testing_path, sfile)))
        self.assertFalse(
            _is_sfile(os.path.join(self.testing_path, 'Sfile_no_header')))
        self.assertFalse(_is_sfile(os.path.join(
            self.path, '..', '..', 'nlloc', 'tests', 'data', 'nlloc.hyp')))

    def test_read_picks_across_day_end(self):
        testing_path = os.path.join(self.testing_path, 'sfile_over_day')
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            event = read_nordic(testing_path)[0]
        pick_times = [pick.time for pick in event.picks]
        for pick in event.picks:
            # Pick should come after origin time
            self.assertGreater(pick.time, event.origins[0].time)
            # All picks in this event are within 60s of origin time
            self.assertLessEqual((pick.time - event.origins[0].time), 60)
        # Make sure zero hours and 24 hour picks are handled the same.
        testing_path = os.path.join(self.testing_path, 'sfile_over_day_zeros')
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            event_2 = read_nordic(testing_path)[0]
        for pick in event_2.picks:
            # Pick should come after origin time
            self.assertGreater(pick.time, event_2.origins[0].time)
            # All picks in this event are within 60s of origin time
            self.assertLessEqual((pick.time - event_2.origins[0].time), 60)
            # Each pick should be the same as one pick in the previous event
            self.assertTrue(pick.time in pick_times)
        self.assertEqual(event_2.origins[0].time, event.origins[0].time)

    def test_distance_conversion(self):
        """
        Check that distances are converted properly.
        """
        testing_path = os.path.join(self.testing_path, '01-0411-15L.S201309')
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            cat = read_events(testing_path)
        event = cat[0]
        self.assertAlmostEqual(
            sorted(event.origins[0].arrivals,
                   key=lambda x: x.distance)[0].distance, 0.035972864236749225)
        pick_strings = nordpick(event)
        self.assertEqual(
            int([p for p in pick_strings if p.split()[0] == 'GCSZ' and
                 p.split()[1] == 'SZ'][0].split()[-1]), 304)
        self.assertEqual(
            int([p for p in pick_strings if p.split()[0] == 'WZ11' and
                 p.split()[1] == 'HZ'][0].split()[-1]), 30)

    def test_large_negative_longitude(self):
        event = full_test_event()
        event.origins[0].longitude = -120
        with NamedTemporaryFile(suffix=".out") as tf:
            with warnings.catch_warnings():
                # Evaluation mode mapping warning
                warnings.simplefilter('ignore', UserWarning)
                event.write(tf.name, format="NORDIC")
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                event_back = read_events(tf.name)
        _assert_similarity(event, event_back[0])

    def test_write_preferred_origin(self):
        event = full_test_event()
        preferred_origin = Origin(
            time=UTCDateTime("2012-03-26") + 2.2, latitude=47.0,
            longitude=35.0, depth=18000)
        event.origins.append(preferred_origin)
        event.preferred_origin_id = preferred_origin.resource_id
        with NamedTemporaryFile(suffix=".out") as tf:
            with warnings.catch_warnings():
                # Evaluation mode mapping warning
                warnings.simplefilter('ignore', UserWarning)
                event.write(tf.name, format="NORDIC")
            with warnings.catch_warnings():
                # Type I warning
                warnings.simplefilter('ignore', UserWarning)
                event_back = read_events(tf.name)
        self.assertEqual(preferred_origin.latitude,
                         event_back[0].origins[0].latitude)
        self.assertEqual(preferred_origin.longitude,
                         event_back[0].origins[0].longitude)
        self.assertEqual(preferred_origin.depth,
                         event_back[0].origins[0].depth)
        self.assertEqual(preferred_origin.time,
                         event_back[0].origins[0].time)

    def test_read_high_precision_pick(self):
        """
        Nordic supports writing to milliseconds in high-precision mode,
        obspy < 1.2.0 did not properly read this, see #2348.
        """
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            cat = read_events(
                os.path.join(self.testing_path, "sfile_high_precision_picks"))
        event = cat[0]
        pick_times = {
            "LSd1": UTCDateTime(2010, 11, 26, 1, 28, 46.859),
            "LSd3": UTCDateTime(2010, 11, 26, 1, 28, 48.132),
            "LSd2": UTCDateTime(2010, 11, 26, 1, 28, 48.183),
            "LSd4": UTCDateTime(2010, 11, 26, 1, 28, 49.744)}
        for key, value in pick_times.items():
            pick = [p for p in event.picks
                    if p.waveform_id.station_code == key]
            self.assertEqual(len(pick), 1)
            self.assertEqual(pick[0].time, value)

    def test_high_precision_read_write(self):
        """ Test that high-precision writing works. """
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            cat = read_events(
                os.path.join(self.testing_path, "sfile_high_precision_picks"))
        event = cat[0]
        pick_times = {
            "LSd1": UTCDateTime(2010, 11, 26, 1, 28, 46.859),
            "LSd3": UTCDateTime(2010, 11, 26, 1, 28, 48.132),
            "LSd2": UTCDateTime(2010, 11, 26, 1, 28, 48.183),
            "LSd4": UTCDateTime(2010, 11, 26, 1, 28, 49.744)}
        for key, value in pick_times.items():
            pick = [p for p in event.picks
                    if p.waveform_id.station_code == key]
            self.assertEqual(len(pick), 1)
            self.assertEqual(pick[0].time, value)
        with NamedTemporaryFile(suffix=".out") as tf:
            write_select(cat, filename=tf.name)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                cat_back = read_events(tf.name)
        self.assertEqual(len(cat_back), 1)
        for key, value in pick_times.items():
            pick = [p for p in cat_back[0].picks
                    if p.waveform_id.station_code == key]
            self.assertEqual(len(pick), 1)
            self.assertEqual(pick[0].time, value)
        # Check that writing to standard accuracy just gives a rounded version
        with NamedTemporaryFile(suffix=".out") as tf:
            cat.write(format="NORDIC", filename=tf.name, high_accuracy=False)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                cat_back = read_events(tf.name)
        self.assertEqual(len(cat_back), 1)
        for key, value in pick_times.items():
            pick = [p for p in cat_back[0].picks
                    if p.waveform_id.station_code == key]
            self.assertEqual(len(pick), 1)
            rounded_pick_time = UTCDateTime(
                value.year, value.month, value.day, value.hour, value.minute)
            rounded_pick_time += round(
                value.second + (value.microsecond / 1e6), 2)
            self.assertEqual(pick[0].time, rounded_pick_time)

    def test_long_phase_name(self):
        """ Nordic format supports 8 char phase names, sometimes. """
        # raises "UserWarning: AIN in header, currently unsupported"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            cat = read_events(
                os.path.join(self.testing_path, "sfile_long_phase"))
        # This file has one event with one pick
        pick = cat[0].picks[0]
        arrival = cat[0].origins[0].arrivals[0]
        self.assertEqual(pick.phase_hint, "PKiKP")
        self.assertEqual(arrival.time_weight, 1)
        with NamedTemporaryFile(suffix=".out") as tf:
            write_select(cat, filename=tf.name)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                cat_back = read_events(tf.name)
        pick = cat_back[0].picks[0]
        arrival = cat_back[0].origins[0].arrivals[0]
        self.assertEqual(pick.phase_hint, "PKiKP")
        self.assertEqual(arrival.time_weight, 1)

    def test_read_write_over_day(self):
        """
        Nordic picks are relative to origin time - check that this works
        over day boundaries.
        """
        event = full_test_event()
        event.origins[0].time -= 3600
        self.assertGreater(
            event.picks[0].time.date, event.origins[0].time.date)
        with NamedTemporaryFile(suffix=".out") as tf:
            with warnings.catch_warnings():
                # Evaluation mode mapping warning
                warnings.simplefilter('ignore', UserWarning)
                write_select(Catalog([event]), filename=tf.name)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                event_back = read_events(tf.name)[0]
        _assert_similarity(event, event_back)

    def test_seconds_overflow(self):
        """
        #2348 indicates that SEISAN sometimes overflows seconds into column 29.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            cat = read_events(
                os.path.join(self.testing_path, "sfile_seconds_overflow"))
        event = cat[0]
        pick_times = {
            "LSb2": UTCDateTime(2009, 7, 2, 6, 49) + 100.24}
        for key, value in pick_times.items():
            pick = [p for p in event.picks
                    if p.waveform_id.station_code == key]
            self.assertEqual(len(pick), 1)
            self.assertEqual(pick[0].time, value)
        with NamedTemporaryFile(suffix=".out") as tf:
            write_select(cat, filename=tf.name)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                cat_back = read_events(tf.name)
        self.assertEqual(len(cat_back), 1)
        for key, value in pick_times.items():
            pick = [p for p in cat_back[0].picks
                    if p.waveform_id.station_code == key]
            self.assertEqual(len(pick), 1)
            self.assertEqual(pick[0].time, value)

    def test_read_bad_covariance(self):
        """
        Verify graceful exit if covariance matrix is not positive-definite
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            cat = read_events(
                os.path.join(self.testing_path, "sfile_bad_covariance"))
        self.assertIs(cat[0].origins[0].origin_uncertainty, None)

    def test_read_high_accuracy(self):
        """
        Verify that high-accuracy locations are read, if present
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            cat = read_events(
                os.path.join(self.testing_path, "sfile_highaccuracy"))
        event = cat[0]
        event_time = event.origins[0].time
        event_lat = event.origins[0].latitude
        event_lon = event.origins[0].longitude
        event_depth = event.origins[0].depth
        event_rms = event.origins[0].quality.standard_error
        self.assertEqual(event_time,
                         UTCDateTime(2015, 4, 24, 15, 25, 37) + 0.676)
        self.assertEqual(event_lat, 37.29242)
        self.assertEqual(event_lon, -32.26983)
        self.assertEqual(event_depth, 1969.)
        self.assertEqual(event_rms, 0.051)

    def test_ellipse_from__to_uncerts(self):
        """
        Verify ellipse is properly calculated and inverted using uncertainties

        tests Ellipse.from_uncerts and Ellipse.to_uncerts()
        """
        center = (20, 30)
        # First try simple cases without correlation
        x_errs = (0.5, 1.33, 1.0)
        y_errs = (1.33, 0.5, 1.0)
        for c_xy in [0, 0.2, 0.4, 0.6]:
            for (x_err, y_err) in zip(x_errs, y_errs):
                ell = Ellipse.from_uncerts(x_err, y_err, c_xy, center)
                (x_err_out, y_err_out, c_xy_out, center_out) = ell.to_uncerts()
                self.assertAlmostEqual(x_err, x_err_out)
                self.assertAlmostEqual(y_err, y_err_out)
                self.assertAlmostEqual(c_xy, c_xy_out)
                self.assertAlmostEqual(center, center_out)
        # Now a specific case with a finite covariance
        x_err = 0.5
        y_err = 1.1
        c_xy = -0.2149
        # Calculate ellipse
        ell = Ellipse.from_uncerts(x_err, y_err, c_xy, center)
        self.assertAlmostEqual(ell.a, 1.120674193646)
        self.assertAlmostEqual(ell.b, 0.451762494786)
        self.assertAlmostEqual(ell.theta, 167.9407699)
        # Calculate covariance error from ellipse
        (x_err_out, y_err_out, c_xy_out, center_out) = ell.to_uncerts()
        self.assertAlmostEqual(x_err, x_err_out)
        self.assertAlmostEqual(y_err, y_err_out)
        self.assertAlmostEqual(c_xy, c_xy_out)
        self.assertAlmostEqual(center, center_out)

    def test_ellipse_from_to_cov(self):
        """
        Verify ellipse is properly calculated and inverted using covariance

        tests Ellipse.from_uncerts and Ellipse.to_uncerts()
        """
        center = (20, 30)
        x_err = 0.5
        y_err = 1.1
        c_xy = -0.2149
        cov = [[x_err**2, c_xy], [c_xy, y_err**2]]
        # Calculate ellipse
        ell = Ellipse.from_cov(cov, center)
        self.assertAlmostEqual(ell.a, 1.120674193646)
        self.assertAlmostEqual(ell.b, 0.451762494786)
        self.assertAlmostEqual(ell.theta, 167.9407699)
        # Calculate covariance error from ellipse
        cov_out, center_out = ell.to_cov()
        self.assertAlmostEqual(cov[0][0], cov_out[0][0])
        self.assertAlmostEqual(cov[0][1], cov_out[0][1])
        self.assertAlmostEqual(cov[1][0], cov_out[1][0])
        self.assertAlmostEqual(cov[1][1], cov_out[1][1])

    def test_ellipse_from_uncerts_baz(self, debug=False):
        """
        Verify alternative ellipse creator

        tests Ellipse.from_uncerts_baz
        """
        # Now a specific case with a finite covariance
        x_err = 0.5
        y_err = 1.1
        c_xy = -0.2149
        dist = 10
        baz = 90
        viewpoint = (5, 5)
        # Calculate ellipse
        ell = Ellipse.from_uncerts_baz(x_err, y_err, c_xy,
                                       dist, baz, viewpoint)
        self.assertAlmostEqual(ell.a, 1.120674193646)
        self.assertAlmostEqual(ell.b, 0.451762494786)
        self.assertAlmostEqual(ell.theta, 167.9407699)
        self.assertAlmostEqual(ell.x, 15)
        self.assertAlmostEqual(ell.y, 5)
        baz = 180
        ell = Ellipse.from_uncerts_baz(x_err, y_err, c_xy,
                                       dist, baz, viewpoint)
        self.assertAlmostEqual(ell.x, 5)
        self.assertAlmostEqual(ell.y, -5)

    def test_ellipse_is_inside(self, debug=False):
        """
        Verify Ellipse.is_inside()
        """
        ell = Ellipse(20, 10, 90)
        self.assertIs(ell.is_inside((0, 0)), True)
        self.assertFalse(ell.is_inside((100, 100)))
        self.assertTrue(ell.is_inside((-19.9, 0)))
        self.assertTrue(ell.is_inside((19.9, 0)))
        self.assertFalse(ell.is_inside((-20.1, 0)))
        self.assertFalse(ell.is_inside((20.1, 0)))
        self.assertTrue(ell.is_inside((0, 9.9)))
        self.assertTrue(ell.is_inside((0, -9.9)))
        self.assertFalse(ell.is_inside((0, 10.1)))
        self.assertFalse(ell.is_inside((0, -10.1)))

    def test_ellipse_is_on(self, debug=False):
        """
        Verify Ellipse.is_on()
        """
        ell = Ellipse(20, 10, 90)
        self.assertFalse(ell.is_on((0, 0)))
        self.assertFalse(ell.is_on((100, 100)))
        self.assertTrue(ell.is_on((-20, 0)))
        self.assertTrue(ell.is_on((20, 0)))
        self.assertFalse(ell.is_on((-20.1, 0)))
        self.assertFalse(ell.is_on((20.1, 0)))
        self.assertTrue(ell.is_on((0, 10)))
        self.assertTrue(ell.is_on((0, -10)))
        self.assertFalse(ell.is_on((0, 10.1)))
        self.assertFalse(ell.is_on((0, -10.1)))

    def test_ellipse_subtended_angle(self, debug=False):
        """
        Verify Ellipse.subtended_angle()
        """
        ell = Ellipse(20, 10, 90)
        self.assertAlmostEqual(ell.subtended_angle((20, 0)), 180.)
        self.assertAlmostEqual(ell.subtended_angle((0, 0)), 360.)
        self.assertAlmostEqual(ell.subtended_angle((40, 0)), 32.2042275039720)
        self.assertAlmostEqual(ell.subtended_angle((0, 40)), 54.6234598480584)
        self.assertAlmostEqual(ell.subtended_angle((20, 10)), 89.9994270422)

    def test_ellipse_plot(self):
        """
        Test Ellipse.plot()

        To generate test figures, used same commands after:
        from ellipse import Ellipse
        import matplotlib.pyplot as plt
        plt.style.use('classic')
        """
        # Test single ellipse
        with ImageComparison(self.testing_path, 'plot_ellipse.png',
                             style='classic', reltol=10) as ic:
            Ellipse(20, 10, 90).plot(outfile=ic.name)
        # Test multi-ellipse figure
        with ImageComparison(self.testing_path, 'plot_ellipses.png',
                             style='classic', reltol=10) as ic:
            fig = Ellipse(20, 10, 90).plot(color='r')
            fig = Ellipse(20, 10, 45).plot(fig=fig, color='b')
            fig = Ellipse(20, 10, 0, center=(10, 10)).plot(fig=fig, color='g')
            fig = Ellipse(20, 10, -45).plot(fig=fig, outfile=ic.name)

    def test_ellipse_plot_tangents(self):
        """
        Test Ellipse.plot_tangents()
        """
        import matplotlib.pyplot as plt
        # Test single ellipse and point
        with ImageComparison(self.testing_path, 'plot_ellipse_tangents.png',
                             style='classic', reltol=10) as ic:
            Ellipse(20, 10, 90).plot_tangents((30, 30),
                                              color='b',
                                              print_angle=True,
                                              ellipse_name='Ellipse',
                                              outfile=ic.name)
        # Test multi-ellipse figure
        dist = 50
        fig = None
        try:
            prop_cycle = plt.rcParams['axes.prop_cycle']
        # prop_cycle was introduced at some point between mpl 1.x and 2.0 it
        # seems
        # XXX workaround can be removed when mpl is bumped to certain version
        except KeyError:
            colors = plt.rcParams['axes.color_cycle']
        else:
            colors = prop_cycle.by_key()['color']
        color_cycle = cycle(colors)
        step = 45
        with ImageComparison(self.testing_path, 'plot_ellipses_tangents.png',
                             style='classic', reltol=15) as ic:
            for angle in range(step, 360 + step - 1, step):
                x = dist * np.sin(np.radians(angle))
                y = dist * np.cos(np.radians(angle))
                ell = Ellipse(20, 10, 90, center=(x, y))
                if angle == 360:
                    outfile = ic.name
                else:
                    outfile = None
                fig = ell.plot_tangents((0, 0),
                                        fig=fig,
                                        color=next(color_cycle),
                                        print_angle=True,
                                        ellipse_name='E{:d}'.format(angle),
                                        outfile=outfile)
        # Test multi-station figure
        fig = None
        color_cycle = cycle(colors)
        with ImageComparison(self.testing_path,
                             'plot_ellipse_tangents_pts.png',
                             style='classic',
                             reltol=15) as ic:
            for angle in range(step, 360 + step - 1, step):
                x = dist * np.sin(np.radians(angle))
                y = dist * np.cos(np.radians(angle))
                ell = Ellipse(20, 10, 90)
                if angle == 360:
                    outfile = ic.name
                else:
                    outfile = None
                fig = ell.plot_tangents((x, y),
                                        fig=fig,
                                        color=next(color_cycle),
                                        print_angle=True,
                                        pt_name='pt{:d}'.format(angle),
                                        outfile=outfile)

    def test_read_uncert_ellipse(self):
        """
        Verify that confidence ellipse is properly read/calculated from nordic
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            cat = read_events(
                os.path.join(self.testing_path, "sfile_highaccuracy"))
        event = cat[0]
        val = event.origins[0].origin_uncertainty
        hor_max = val['max_horizontal_uncertainty']
        hor_min = val['min_horizontal_uncertainty']
        azi_max = val['azimuth_max_horizontal_uncertainty']
        self.assertAlmostEqual(hor_max, 1120.674193646)
        self.assertAlmostEqual(hor_min, 451.762494786)
        self.assertAlmostEqual(azi_max, 167.9407699)


def _assert_similarity(event_1, event_2):
    """
    Raise AssertionError if testing similarity fails
    """
    similarity_output = _test_similarity(event_1, event_2)
    if similarity_output:
        raise AssertionError(similarity_output)


def _test_similarity(event_1, event_2):
    """
    Check the similarity of the components of obspy events, discounting
    resource IDs, which are not maintained in nordic files.

    Raise AssertionError if test fails

    :type event_1: obspy.core.event.Event
    :param event_1: First event
    :type event_2: obspy.core.event.Event
    :param event_2: Comparison event
    :type verbose: bool
    :param verbose: If true and fails will output why it fails.
    """
    # What None maps to.
    pick_default_mapper = {
        "polarity": "undecidable", "evaluation_mode": "manual"}
    # Check origins
    if len(event_1.origins) != len(event_2.origins):
        return False
    for ori_1, ori_2 in zip(event_1.origins, event_2.origins):
        for key in ori_1.keys():
            if key not in ["resource_id", "comments", "arrivals",
                           "method_id", "origin_uncertainty", "depth_type",
                           "quality", "creation_info", "evaluation_mode",
                           "depth_errors", "time_errors"]:
                if ori_1[key] != ori_2[key]:
                    return ('%s is not the same as %s for key %s' %
                            (ori_1[key], ori_2[key], key))
            elif key == "arrivals":
                if len(ori_1[key]) != len(ori_2[key]):
                    return ('%i is not the same as %i for key %s' %
                            (len(ori_1[key]), len(ori_2[key]), key))
                for arr_1, arr_2 in zip(ori_1[key], ori_2[key]):
                    for arr_key in arr_1.keys():
                        if arr_key not in ["resource_id", "pick_id",
                                           "distance"]:
                            if arr_1[arr_key] != arr_2[arr_key]:
                                return ('%s does not match %s for key %s' %
                                        (arr_1[arr_key], arr_2[arr_key],
                                         arr_key))
                    if arr_1["distance"] and round(
                            arr_1["distance"]) != round(arr_2["distance"]):
                        return ('%s does not match %s for key %s' %
                                (arr_1[arr_key], arr_2[arr_key],
                                 arr_key))
    # Check picks
    if len(event_1.picks) != len(event_2.picks):
        return 'Number of picks is not equal'
    for pick_1, pick_2 in zip(event_1.picks, event_2.picks):
        # Assuming same ordering of picks...
        for key in pick_1.keys():
            if key not in ["resource_id", "waveform_id"]:
                if pick_1[key] != pick_2[key]:
                    default = pick_default_mapper.get(key, None)
                    if pick_1[key] is None:
                        if pick_2[key] == default:
                            continue
                    elif pick_2[key] is None:
                        if pick_1[key] == default:
                            continue
                    return ('%s is not the same as %s for key %s' %
                            (pick_1[key], pick_2[key], key))
            elif key == "waveform_id":
                if pick_1[key].station_code != pick_2[key].station_code:
                    return 'Station codes do not match'
                if pick_1[key].channel_code[0] != pick_2[key].channel_code[0]:
                    return 'Channel codes do not match'
                if pick_1[key].channel_code[-1] !=\
                   pick_2[key].channel_code[-1]:
                    return 'Channel codes do not match'
    # Check amplitudes
    if not len(event_1.amplitudes) == len(event_2.amplitudes):
        return 'Not the same number of amplitudes'
    for amp_1, amp_2 in zip(event_1.amplitudes, event_2.amplitudes):
        # Assuming same ordering of amplitudes
        for key in amp_1.keys():
            if key not in ["resource_id", "pick_id", "waveform_id", "snr",
                           "magnitude_hint", 'type']:
                if not amp_1[key] == amp_2[key]:
                    return ("{0} is not the same as {1} for key "
                            "{2}\n{3}\n{4}".format(
                                amp_1[key], amp_2[key], key, amp_1, amp_2))
            elif key == "waveform_id":
                if pick_1[key].station_code != pick_2[key].station_code:
                    return 'Station codes do not match'
                if pick_1[key].channel_code[0] != pick_2[key].channel_code[0]:
                    return 'Channel codes do not match'
                if pick_1[key].channel_code[-1] !=\
                        pick_2[key].channel_code[-1]:
                    return 'Channel codes do not match'
            elif key in ["magnitude_hint", "type"]:
                # Reading back in will define both, but input event might have
                # None
                if amp_1[key] is not None:
                    if not amp_1[key] == amp_2[key]:
                        return ('%s is not the same as %s for key %s' %
                                (amp_1[key], amp_2[key], key))
    return None


def full_test_event():
    """
    Function to generate a basic, full test event
    """
    test_event = Event()
    test_event.origins.append(Origin(
        time=UTCDateTime("2012-03-26") + 1.2, latitude=45.0, longitude=25.0,
        depth=15000))
    test_event.event_descriptions.append(EventDescription())
    test_event.event_descriptions[0].text = 'LE'
    test_event.creation_info = CreationInfo(agency_id='TES')
    test_event.magnitudes.append(Magnitude(
        mag=0.1, magnitude_type='ML', creation_info=CreationInfo('TES'),
        origin_id=test_event.origins[0].resource_id))
    test_event.magnitudes.append(Magnitude(
        mag=0.5, magnitude_type='Mc', creation_info=CreationInfo('TES'),
        origin_id=test_event.origins[0].resource_id))
    test_event.magnitudes.append(Magnitude(
        mag=1.3, magnitude_type='Ms', creation_info=CreationInfo('TES'),
        origin_id=test_event.origins[0].resource_id))

    # Define the test pick
    _waveform_id_1 = WaveformStreamID(station_code='FOZ', channel_code='SHZ',
                                      network_code='NZ')
    _waveform_id_2 = WaveformStreamID(station_code='WTSZ', channel_code='BH1',
                                      network_code=' ')
    # Pick to associate with amplitude - 0
    test_event.picks = [
        Pick(waveform_id=_waveform_id_1, phase_hint='IAML',
             polarity='undecidable', time=UTCDateTime("2012-03-26") + 1.68,
             evaluation_mode="manual"),
        Pick(waveform_id=_waveform_id_1, onset='impulsive', phase_hint='PN',
             polarity='positive', time=UTCDateTime("2012-03-26") + 1.68,
             evaluation_mode="manual"),
        Pick(waveform_id=_waveform_id_1, phase_hint='IAML',
             polarity='undecidable', time=UTCDateTime("2012-03-26") + 1.68,
             evaluation_mode="manual"),
        Pick(waveform_id=_waveform_id_2, onset='impulsive', phase_hint='SG',
             polarity='undecidable', time=UTCDateTime("2012-03-26") + 1.72,
             evaluation_mode="manual"),
        Pick(waveform_id=_waveform_id_2, onset='impulsive', phase_hint='PN',
             polarity='undecidable', time=UTCDateTime("2012-03-26") + 1.62,
             evaluation_mode="automatic"),
        # Missing info shouldn't be an issue
        Pick(waveform_id=_waveform_id_2, onset=None, phase_hint='PN',
             polarity=None, time=UTCDateTime("2012-03-26") + 1.92,
             evaluation_mode=None),
        # Long-phase
        Pick(waveform_id=_waveform_id_2, onset='impulsive', phase_hint='PKiKP',
             polarity=None, time=UTCDateTime("2012-03-26") + 1.92,
             evaluation_mode=None),
    ]
    # Test a generic local magnitude amplitude pick
    test_event.amplitudes = [
        Amplitude(generic_amplitude=2.0, period=0.4,
                  pick_id=test_event.picks[0].resource_id,
                  waveform_id=test_event.picks[0].waveform_id, unit='m',
                  magnitude_hint='ML', category='point', type='AML'),
        Amplitude(generic_amplitude=10,
                  pick_id=test_event.picks[1].resource_id,
                  waveform_id=test_event.picks[1].waveform_id, type='END',
                  category='duration', unit='s', magnitude_hint='Mc',
                  snr=2.3),
        Amplitude(generic_amplitude=5.0, period=0.6,
                  pick_id=test_event.picks[2].resource_id,
                  waveform_id=test_event.picks[0].waveform_id, unit='m',
                  category='point', type='AML')]
    test_event.origins[0].arrivals = [
        Arrival(time_weight=0, phase=test_event.picks[1].phase_hint,
                pick_id=test_event.picks[1].resource_id),
        Arrival(time_weight=2, phase=test_event.picks[3].phase_hint,
                pick_id=test_event.picks[3].resource_id,
                backazimuth_residual=5, time_residual=0.2, distance=15,
                azimuth=25, takeoff_angle=10),
        Arrival(time_weight=2, phase=test_event.picks[4].phase_hint,
                pick_id=test_event.picks[4].resource_id,
                backazimuth_residual=5, time_residual=0.2, distance=15,
                azimuth=25, takeoff_angle=170),
        Arrival(time_weight=2, phase=test_event.picks[5].phase_hint,
                pick_id=test_event.picks[5].resource_id,
                backazimuth_residual=5, time_residual=0.2, distance=15,
                azimuth=25, takeoff_angle=170),
        Arrival(time_weight=0, phase=test_event.picks[6].phase_hint,
                pick_id=test_event.picks[6].resource_id,
                backazimuth_residual=5, time_residual=0.2, distance=15,
                azimuth=25, takeoff_angle=170),
    ]
    # Add in error info (line E)
    test_event.origins[0].quality = OriginQuality(
        standard_error=0.01, azimuthal_gap=36)
    # Origin uncertainty in Seisan is output as long-lat-depth, quakeML has
    # semi-major and semi-minor
    test_event.origins[0].origin_uncertainty = OriginUncertainty(
        confidence_ellipsoid=ConfidenceEllipsoid(
            semi_major_axis_length=3000, semi_minor_axis_length=1000,
            semi_intermediate_axis_length=2000, major_axis_plunge=20,
            major_axis_azimuth=100, major_axis_rotation=4))
    test_event.origins[0].time_errors = QuantityError(uncertainty=0.5)
    # Add in fault-plane solution info (line F) - Note have to check program
    # used to determine which fields are filled....
    test_event.focal_mechanisms.append(FocalMechanism(
        nodal_planes=NodalPlanes(nodal_plane_1=NodalPlane(
            strike=180, dip=20, rake=30, strike_errors=QuantityError(10),
            dip_errors=QuantityError(10), rake_errors=QuantityError(20))),
        method_id=ResourceIdentifier("smi:nc.anss.org/focalMechanism/FPFIT"),
        creation_info=CreationInfo(agency_id="NC"), misfit=0.5,
        station_distribution_ratio=0.8))
    # Need to test high-precision origin and that it is preferred origin.
    # Moment tensor includes another origin
    test_event.origins.append(Origin(
        time=UTCDateTime("2012-03-26") + 1.2, latitude=45.1, longitude=25.2,
        depth=14500))
    test_event.magnitudes.append(Magnitude(
        mag=0.1, magnitude_type='MW', creation_info=CreationInfo('TES'),
        origin_id=test_event.origins[-1].resource_id))
    # Moment tensors go with focal-mechanisms
    test_event.focal_mechanisms.append(FocalMechanism(
        moment_tensor=MomentTensor(
            derived_origin_id=test_event.origins[-1].resource_id,
            moment_magnitude_id=test_event.magnitudes[-1].resource_id,
            scalar_moment=100, tensor=Tensor(
                m_rr=100, m_tt=100, m_pp=10, m_rt=1, m_rp=20, m_tp=15),
            method_id=ResourceIdentifier(
                'smi:nc.anss.org/momentTensor/BLAH'))))
    return test_event


def suite():
    return unittest.makeSuite(TestNordicMethods, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
