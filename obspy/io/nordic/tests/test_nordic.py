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

from obspy import read_events, Catalog, UTCDateTime, read
from obspy.core.event import Pick, WaveformStreamID, Arrival, Amplitude
from obspy.core.event import Event, Origin, Magnitude
from obspy.core.event import EventDescription, CreationInfo
from obspy.core.util.base import NamedTemporaryFile
from obspy.io.nordic.core import _is_sfile, read_spectral_info, read_nordic
from obspy.io.nordic.core import readwavename, blanksfile, _write_nordic
from obspy.io.nordic.core import nordpick, readheader
from obspy.io.nordic.core import _int_conv, _readheader, _evmagtonor
from obspy.io.nordic.core import write_select, NordicParsingError
from obspy.io.nordic.core import _float_conv, _nortoevmag, _str_conv


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
        # Add the event to a catalogue which can be used for QuakeML testing
        test_cat = Catalog()
        test_cat += test_event
        # Check the read-write s-file functionality
        sfile = _write_nordic(test_cat[0], filename=None, userid='TEST',
                              evtype='L', outdir='.',
                              wavefiles='test', explosion=True, overwrite=True)
        self.assertEqual(readwavename(sfile), ['test'])
        read_cat = Catalog()
        read_cat += read_nordic(sfile)
        os.remove(sfile)
        read_ev = read_cat[0]
        test_ev = test_cat[0]
        for read_pick, test_pick in zip(read_ev.picks, test_ev.picks):
            self.assertEqual(read_pick.time, test_pick.time)
            self.assertEqual(read_pick.backazimuth, test_pick.backazimuth)
            self.assertEqual(read_pick.onset, test_pick.onset)
            self.assertEqual(read_pick.phase_hint, test_pick.phase_hint)
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
                         test_ev.amplitudes[1].
                         waveform_id.channel_code[0] +
                         test_ev.amplitudes[1].
                         waveform_id.channel_code[-1])
        self.assertEqual(read_ev.amplitudes[1].magnitude_hint,
                         test_ev.amplitudes[1].magnitude_hint)
        # snr is not supported in s-file
        # self.assertEqual(read_ev.amplitudes[1].snr,
        #                  test_ev.amplitudes[1].snr)
        self.assertEqual(read_ev.amplitudes[1].category,
                         test_ev.amplitudes[1].category)

    def test_fail_writing(self):
        """Test a deliberate fail."""
        test_event = full_test_event()
        # Add the event to a catalogue which can be used for QuakeML testing
        test_cat = Catalog()
        test_cat += test_event
        test_ev = test_cat[0]
        test_cat.append(full_test_event())
        with self.assertRaises(NordicParsingError):
            # Raises error due to multiple events in catalog
            _write_nordic(test_cat, filename=None, userid='TEST',
                          evtype='L', outdir='.',
                          wavefiles='test', explosion=True,
                          overwrite=True)
        with self.assertRaises(NordicParsingError):
            # Raises error due to too long userid
            _write_nordic(test_ev, filename=None, userid='TESTICLE',
                          evtype='L', outdir='.',
                          wavefiles='test', explosion=True,
                          overwrite=True)
        with self.assertRaises(NordicParsingError):
            # Raises error due to unrecognised event type
            _write_nordic(test_ev, filename=None, userid='TEST',
                          evtype='U', outdir='.',
                          wavefiles='test', explosion=True,
                          overwrite=True)
        with self.assertRaises(NordicParsingError):
            # Raises error due to no output directory
            _write_nordic(test_ev, filename=None, userid='TEST',
                          evtype='L', outdir='albatross',
                          wavefiles='test', explosion=True,
                          overwrite=True)
        invalid_origin = test_ev.copy()
        invalid_origin.origins = []
        with self.assertRaises(NordicParsingError):
            _write_nordic(invalid_origin, filename=None, userid='TEST',
                          evtype='L', outdir='.',
                          wavefiles='test', explosion=True,
                          overwrite=True)
        invalid_origin = test_ev.copy()
        invalid_origin.origins[0].time = None
        with self.assertRaises(NordicParsingError):
            _write_nordic(invalid_origin, filename=None, userid='TEST',
                          evtype='L', outdir='.',
                          wavefiles='test', explosion=True,
                          overwrite=True)
        # Write a near empty origin
        valid_origin = test_ev.copy()
        valid_origin.origins[0].latitude = None
        valid_origin.origins[0].longitude = None
        valid_origin.origins[0].depth = None
        with NamedTemporaryFile() as tf:
            _write_nordic(valid_origin, filename=tf.name, userid='TEST',
                          evtype='L', outdir='.', wavefiles='test',
                          explosion=True, overwrite=True)
            self.assertTrue(os.path.isfile(tf.name))

    def test_blanksfile(self):
        st = read()
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
        for i in range(10):
            sfiles.append(blanksfile(testing_path, 'L', 'TEST'))
        with self.assertRaises(NordicParsingError):
            blanksfile(testing_path, 'L', 'TEST')
        for sfile in sfiles:
            self.assertTrue(os.path.isfile(sfile))
            os.remove(sfile)
        os.remove(testing_path)

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
        test_sfile = _write_nordic(test_event, filename=None, userid='TEST',
                                   evtype='L', outdir='.', wavefiles='test')
        self.assertTrue(os.path.isfile(test_sfile))
        os.remove(test_sfile)

    def test_read_empty_header(self):
        """
        Function to check a known issue, empty header info S-file: Bug found \
        by Dominic Evanzia.
        """
        test_event = read_nordic(os.path.join(self.testing_path,
                                              'Sfile_no_location'))[0]
        self.assertFalse(test_event.origins[0].latitude)
        self.assertFalse(test_event.origins[0].longitude)
        self.assertFalse(test_event.origins[0].depth)

    def test_read_extra_header(self):
        testing_path = os.path.join(self.testing_path, 'Sfile_extra_header')
        not_extra_header = os.path.join(self.testing_path,
                                        '01-0411-15L.S201309')
        test_event = read_nordic(testing_path)[0]
        header_event = read_nordic(not_extra_header)[0]
        self.assertEqual(test_event.origins[0].time,
                         header_event.origins[0].time)
        self.assertEqual(test_event.origins[0].latitude,
                         header_event.origins[0].latitude)
        self.assertEqual(test_event.origins[0].longitude,
                         header_event.origins[0].longitude)
        self.assertEqual(test_event.origins[0].depth,
                         header_event.origins[0].depth)

    def test_header_mapping(self):
        head_1 = readheader(os.path.join(self.testing_path,
                                         '01-0411-15L.S201309'))
        with open(os.path.join(self.testing_path,
                               '01-0411-15L.S201309'), 'r') as f:
            head_2 = _readheader(f=f)
        self.assertTrue(test_similarity(head_1, head_2))

    def test_missing_header(self):
        # Check that a suitable error is raised
        with self.assertRaises(NordicParsingError):
            readheader(os.path.join(self.testing_path, 'Sfile_no_header'))

    def test_reading_string_io(self):
        filename = os.path.join(self.testing_path, '01-0411-15L.S201309')
        with open(filename, "rt") as fh:
            file_object = io.StringIO(fh.read())

        cat = read_events(file_object)
        file_object.close()

        ref_cat = read_events(filename)
        self.assertTrue(test_similarity(cat[0], ref_cat[0]))

    def test_reading_bytes_io(self):
        filename = os.path.join(self.testing_path, '01-0411-15L.S201309')
        with open(filename, "rb") as fh:
            file_object = io.BytesIO(fh.read())

        cat = read_events(file_object)
        file_object.close()

        ref_cat = read_events(filename)
        self.assertTrue(test_similarity(cat[0], ref_cat[0]))

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
        for i in range(59):
            sfiles.append(_write_nordic(event=event, filename=None,
                                        overwrite=False))
        with self.assertRaises(NordicParsingError):
            _write_nordic(event=event, filename=None, overwrite=False)
        for sfile in sfiles:
            os.remove(sfile)

    def test_mag_conv(self):
        """Check that we convert magnitudes as we should!"""
        magnitude_map = [('L', 'ML'),
                         ('B', 'mB'),
                         ('S', 'Ms'),
                         ('W', 'MW'),
                         ('G', 'MbLg'),
                         ('C', 'Mc'),
                         ]
        for magnitude in magnitude_map:
            self.assertEqual(magnitude[0], _evmagtonor(magnitude[1]))
            self.assertEqual(_nortoevmag(magnitude[0]), magnitude[1])

    def test_str_conv(self):
        """Test the simple string conversions."""
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

    def test_read_event(self):
        """Test the wrapper."""
        testing_path = os.path.join(self.testing_path, '01-0411-15L.S201309')
        event = read_nordic(testing_path)[0]
        self.assertEqual(len(event.origins), 1)

    def test_read_many_events(self):
        testing_path = os.path.join(self.testing_path, 'select.out')
        catalog = read_nordic(testing_path)
        self.assertEqual(len(catalog), 50)

    def test_write_select(self):
        cat = read_events()
        with NamedTemporaryFile(suffix='.out') as tf:
            write_select(cat, filename=tf.name)
            cat_back = read_events(tf.name)
            for event_1, event_2 in zip(cat, cat_back):
                self.assertTrue(test_similarity(event_1=event_1,
                                                event_2=event_2))

    def test_write_plugin(self):
        cat = read_events()
        cat.append(full_test_event())
        with NamedTemporaryFile(suffix='.out') as tf:
            cat.write(tf.name, format='nordic')
            cat_back = read_events(tf.name)
            for event_1, event_2 in zip(cat, cat_back):
                self.assertTrue(test_similarity(event_1=event_1,
                                                event_2=event_2))

    def test_inaccurate_picks(self):
        testing_path = os.path.join(self.testing_path, 'bad_picks.sfile')
        cat = read_nordic(testing_path)
        pick_string = nordpick(cat[0])
        for pick in pick_string:
            self.assertEqual(len(pick), 80)

    def test_round_len(self):
        testing_path = os.path.join(self.testing_path, 'round_len_undef.sfile')
        event = read_nordic(testing_path)[0]
        pick_string = nordpick(event)
        for pick in pick_string:
            self.assertEqual(len(pick), 80)

    def test_read_moment(self):
        """Test the reading of seismic moment from the s-file."""
        testing_path = os.path.join(self.testing_path, 'automag.out')
        event = read_nordic(testing_path)[0]
        mag = [m for m in event.magnitudes if m.magnitude_type == 'MW']
        self.assertEqual(len(mag), 1)
        self.assertEqual(mag[0].mag, 0.7)

    def test_read_moment_info(self):
        """Test reading the info from spectral analysis."""
        testing_path = os.path.join(self.testing_path, 'automag.out')
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
        self.assertFalse(_is_sfile(os.path.join(self.testing_path,
                                                'Sfile_no_header')))
        self.assertFalse(_is_sfile(os.path.join(self.path, '..', '..',
                                                'nlloc', 'tests', 'data',
                                                'nlloc.hyp')))

    def test_read_picks_across_day_end(self):
        testing_path = os.path.join(self.testing_path, 'sfile_over_day')
        event = read_nordic(testing_path)[0]
        pick_times = [pick.time for pick in event.picks]
        for pick in event.picks:
            # Pick should come after origin time
            self.assertGreater(pick.time, event.origins[0].time)
            # All picks in this event are within 60s of origin time
            self.assertLessEqual((pick.time - event.origins[0].time), 60)
        # Make sure zero hours and 24 hour picks are handled the same.
        testing_path = os.path.join(self.testing_path, 'sfile_over_day_zeros')
        event_2 = read_nordic(testing_path)[0]
        for pick in event_2.picks:
            # Pick should come after origin time
            self.assertGreater(pick.time, event_2.origins[0].time)
            # All picks in this event are within 60s of origin time
            self.assertLessEqual((pick.time - event_2.origins[0].time), 60)
            # Each pick should be the same as one pick in the previous event
            self.assertTrue(pick.time in pick_times)
        self.assertEqual(event_2.origins[0].time, event.origins[0].time)


def test_similarity(event_1, event_2):
    """
    Check the similarity of the components of obspy events, discounting
    resource IDs, which are not maintained in nordic files.

    :type event_1: obspy.core.event.Event
    :param event_1: First event
    :type event_2: obspy.core.event.Event
    :param event_2: Comparison event
    :return: bool
    """
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
                    return False
            elif key == "arrivals":
                if len(ori_1[key]) != len(ori_2[key]):
                    return False
                for arr_1, arr_2 in zip(ori_1[key], ori_2[key]):
                    for arr_key in arr_1.keys():
                        if arr_key not in ["resource_id", "pick_id"]:
                            if arr_1[arr_key] != arr_2[arr_key]:
                                return False
    # Check picks
    if len(event_1.picks) != len(event_2.picks):
        return False
    for pick_1, pick_2 in zip(event_1.picks, event_2.picks):
        # Assuming same ordering of picks...
        for key in pick_1.keys():
            if key not in ["resource_id", "waveform_id"]:
                if pick_1[key] != pick_2[key]:
                    return False
            elif key == "waveform_id":
                if pick_1[key].station_code != pick_2[key].station_code:
                    return False
                if pick_1[key].channel_code[0] != pick_2[key].channel_code[0]:
                    return False
                if pick_1[key].channel_code[-1] !=\
                   pick_2[key].channel_code[-1]:
                    return False
    # Check amplitudes
    if not len(event_1.amplitudes) == len(event_2.amplitudes):
        print('amplitudes')
        return False
    for amp_1, amp_2 in zip(event_1.amplitudes, event_2.amplitudes):
        # Assuming same ordering of amplitudes
        for key in amp_1.keys():
            if key not in ["resource_id", "pick_id", "waveform_id", "snr"]:
                if not amp_1[key] == amp_2[key]:
                    print(key)
                    return False
            elif key == "waveform_id":
                if pick_1[key].station_code != pick_2[key].station_code:
                    return False
                if pick_1[key].channel_code[0] != pick_2[key].channel_code[0]:
                    return False
                if pick_1[key].channel_code[-1] !=\
                   pick_2[key].channel_code[-1]:
                    return False
    return True


def full_test_event():
    """
    Function to generate a basic, full test event
    """
    test_event = Event()
    test_event.origins.append(Origin())
    test_event.origins[0].time = UTCDateTime("2012-03-26") + 1.2
    test_event.event_descriptions.append(EventDescription())
    test_event.event_descriptions[0].text = 'LE'
    test_event.origins[0].latitude = 45.0
    test_event.origins[0].longitude = 25.0
    test_event.origins[0].depth = 15000
    test_event.creation_info = CreationInfo(agency_id='TES')
    test_event.origins[0].time_errors['Time_Residual_RMS'] = 0.01
    test_event.magnitudes.append(Magnitude())
    test_event.magnitudes[0].mag = 0.1
    test_event.magnitudes[0].magnitude_type = 'ML'
    test_event.magnitudes[0].creation_info = CreationInfo('TES')
    test_event.magnitudes[0].origin_id = test_event.origins[0].resource_id
    test_event.magnitudes.append(Magnitude())
    test_event.magnitudes[1].mag = 0.5
    test_event.magnitudes[1].magnitude_type = 'Mc'
    test_event.magnitudes[1].creation_info = CreationInfo('TES')
    test_event.magnitudes[1].origin_id = test_event.origins[0].resource_id
    test_event.magnitudes.append(Magnitude())
    test_event.magnitudes[2].mag = 1.3
    test_event.magnitudes[2].magnitude_type = 'Ms'
    test_event.magnitudes[2].creation_info = CreationInfo('TES')
    test_event.magnitudes[2].origin_id = test_event.origins[0].resource_id

    # Define the test pick
    _waveform_id_1 = WaveformStreamID(station_code='FOZ', channel_code='SHZ',
                                      network_code='NZ')
    _waveform_id_2 = WaveformStreamID(station_code='WTSZ', channel_code='BH1',
                                      network_code=' ')
    # Pick to associate with amplitude
    test_event.picks.append(
        Pick(waveform_id=_waveform_id_1, phase_hint='IAML',
             polarity='undecidable', time=UTCDateTime("2012-03-26") + 1.68,
             evaluation_mode="manual"))
    # Need a second pick for coda
    test_event.picks.append(
        Pick(waveform_id=_waveform_id_1, onset='impulsive', phase_hint='PN',
             polarity='positive', time=UTCDateTime("2012-03-26") + 1.68,
             evaluation_mode="manual"))
    # Unassociated pick
    test_event.picks.append(
        Pick(waveform_id=_waveform_id_2, onset='impulsive', phase_hint='SG',
             polarity='undecidable', time=UTCDateTime("2012-03-26") + 1.72,
             evaluation_mode="manual"))
    # Unassociated pick
    test_event.picks.append(
        Pick(waveform_id=_waveform_id_2, onset='impulsive', phase_hint='PN',
             polarity='undecidable', time=UTCDateTime("2012-03-26") + 1.62,
             evaluation_mode="automatic"))
    # Test a generic local magnitude amplitude pick
    test_event.amplitudes.append(
        Amplitude(generic_amplitude=2.0, period=0.4,
                  pick_id=test_event.picks[0].resource_id,
                  waveform_id=test_event.picks[0].waveform_id, unit='m',
                  magnitude_hint='ML', category='point', type='AML'))
    # Test a coda magnitude pick
    test_event.amplitudes.append(
        Amplitude(generic_amplitude=10,
                  pick_id=test_event.picks[1].resource_id,
                  waveform_id=test_event.picks[1].waveform_id, type='END',
                  category='duration', unit='s', magnitude_hint='Mc',
                  snr=2.3))
    test_event.origins[0].arrivals.append(
        Arrival(time_weight=0, phase=test_event.picks[1].phase_hint,
                pick_id=test_event.picks[1].resource_id))
    test_event.origins[0].arrivals.append(
        Arrival(time_weight=2, phase=test_event.picks[2].phase_hint,
                pick_id=test_event.picks[2].resource_id,
                backazimuth_residual=5, time_residual=0.2, distance=15,
                azimuth=25))
    test_event.origins[0].arrivals.append(
        Arrival(time_weight=2, phase=test_event.picks[3].phase_hint,
                pick_id=test_event.picks[3].resource_id,
                backazimuth_residual=5, time_residual=0.2, distance=15,
                azimuth=25))
    return test_event


def suite():
    return unittest.makeSuite(TestNordicMethods, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
