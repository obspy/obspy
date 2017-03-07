# -*- coding: utf-8 -*-
"""
Nordic file format support for ObsPy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

.. note::

    Currently does not read nor write moment tensors or focal mechanism
    solutions from/to Nordic files.

.. note::

    Pick time-residuals are handled in event.origins[0].arrivals, with
    the arrival.pick_id linking the arrival (which contain calculated
    information) with the pick.resource_id (where the pick contains only
    physical measured information).
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import warnings
import datetime
import os
import io

from obspy import UTCDateTime, read
from obspy.geodetics import kilometers2degrees, degrees2kilometers
from obspy.core.event import Event, Origin, Magnitude, Comment, Catalog
from obspy.core.event import EventDescription, CreationInfo, OriginQuality
from obspy.core.event import Pick, WaveformStreamID, Arrival, Amplitude


mag_mapping = {"ML": "L", "MLv": "L", "mB": "B", "Ms": "S", "MW": "W",
               "MbLg": "G", "Mc": "C"}

onsets = {'I': 'impulsive', 'E': 'emergent'}


class NordicParsingError(Exception):
    """
    Internal general error for IO operations in obspy.core.io.nordic.
    """
    def __init__(self, value):
        self.value = value


def _is_sfile(sfile):
    """
    Basic test of whether the file is nordic format or not.

    Not exhaustive, but checks some of the basics.

    :type sfile: str
    :param sfile: Path to sfile
    :rtype: bool
    """
    if not hasattr(sfile, "readline"):
        try:
            with open(sfile, 'r') as f:
                head_line = _get_headline(f)
        except Exception:
            return False
    else:
        head_line = _get_headline(sfile)
    if head_line is not None:
        try:
            sfile_seconds = int(head_line[16:18])
        except ValueError:
            return False
        if sfile_seconds == 60:
            sfile_seconds = 0
        try:
            UTCDateTime(int(head_line[1:5]), int(head_line[6:8]),
                        int(head_line[8:10]), int(head_line[11:13]),
                        int(head_line[13:15]), sfile_seconds,
                        int(head_line[19:20]) * 100000)
            return True
        except Exception:
            return False
    else:
        return False


def _get_headline(f):
    for i, line in enumerate(f):
        if i == 0 and len(line.rstrip()) != 80:
            return None
        if line[79] == '1':
            return line
    else:
        return None


def _int_conv(string):
    """
    Convenience tool to convert from string to integer.

    If empty string return None rather than an error.

    >>> _int_conv('12')
    12
    >>> _int_conv('')

    """
    try:
        intstring = int(string)
    except Exception:
        intstring = None
    return intstring


def _float_conv(string):
    """
    Convenience tool to convert from string to float.

    If empty string return None rather than an error.

    >>> _float_conv('12')
    12.0
    >>> _float_conv('')
    >>> _float_conv('12.324')
    12.324
    """
    try:
        floatstring = float(string)
    except Exception:
        floatstring = None
    return floatstring


def _str_conv(number, rounded=False):
    """
    Convenience tool to convert a number, either float or int into a string.

    If the int or float is None, returns empty string.

    >>> print(_str_conv(12.3))
    12.3
    >>> print(_str_conv(12.34546, rounded=1))
    12.3
    >>> print(_str_conv(None))
    <BLANKLINE>
    >>> print(_str_conv(1123040))
    11.2e5
    """
    if not number:
        return str(' ')
    if not rounded and isinstance(number, (float, int)):
        if number < 100000:
            string = str(number)
        else:
            exponant = int('{0:.2E}'.format(number).split('E+')[-1]) - 1
            divisor = 10 ** exponant
            string = '{0:.1f}'.format(number / divisor) + 'e' + str(exponant)
    elif rounded == 2 and isinstance(number, (float, int)):
        if number < 100000:
            string = '{0:.2f}'.format(number)
        else:
            exponant = int('{0:.2E}'.format(number).split('E+')[-1]) - 1
            divisor = 10 ** exponant
            string = '{0:.2f}'.format(number / divisor) + 'e' + str(exponant)
    elif rounded == 1 and isinstance(number, (float, int)):
        if number < 100000:
            string = '{0:.1f}'.format(number)
        else:
            exponant = int('{0:.2E}'.format(number).split('E+')[-1]) - 1
            divisor = 10 ** exponant
            string = '{0:.1f}'.format(number / divisor) + 'e' + str(exponant)
    else:
        return str(number)
    return string


def _evmagtonor(mag_type):
    """
    Switch from obspy event magnitude types to seisan syntax.

    >>> print(_evmagtonor('mB'))  # doctest: +SKIP
    B
    >>> print(_evmagtonor('M'))  # doctest: +SKIP
    W
    >>> print(_evmagtonor('bob'))  # doctest: +SKIP
    <BLANKLINE>
    """
    if mag_type == 'M':
        msg = ('Converting generic magnitude to moment magnitude')
        warnings.warn(msg)
        return "W"
    try:
        mag = mag_mapping[mag_type]
    except KeyError:
        warnings.warn(mag_type + ' is not convertible')
        return ''
    return mag


def _nortoevmag(mag_type):
    """
    Switch from nordic type magnitude notation to obspy event magnitudes.

    >>> print(_nortoevmag('b'))  # doctest: +SKIP
    mB
    >>> print(_nortoevmag('bob'))  # doctest: +SKIP
    <BLANKLINE>
    """
    if mag_type.upper() == "L":
        return "ML"
    inv_mag_mapping = {item: key for key, item in mag_mapping.items()}
    try:
        mag = inv_mag_mapping[mag_type.upper()]
    except KeyError:
        warnings.warn(mag_type + ' is not convertible')
        return ''
    return mag


def readheader(sfile):
    """
    Read header information from a seisan nordic format S-file.

    :type sfile: str
    :param sfile: Path to the s-file

    :returns: :class:`~obspy.core.event.event.Event`
    """
    with open(sfile, 'r') as f:
        header = _readheader(f=f)
    return header


def _readheader(f):
    """
    Internal header reader.
    :type f: file
    :param f: File open in read-mode.

    :returns: :class:`~obspy.core.event.event.Event`
    """
    f.seek(0)
    # Base populate to allow for empty parts of file
    new_event = Event()
    topline = _get_headline(f=f)
    if not topline:
        raise NordicParsingError('No header found, or incorrect '
                                 'formatting: corrupt s-file')
    try:
        sfile_seconds = int(topline[16:18])
        if sfile_seconds == 60:
            sfile_seconds = 0
            add_seconds = 60
        else:
            add_seconds = 0
        new_event.origins.append(Origin())
        new_event.origins[0].time = UTCDateTime(int(topline[1:5]),
                                                int(topline[6:8]),
                                                int(topline[8:10]),
                                                int(topline[11:13]),
                                                int(topline[13:15]),
                                                sfile_seconds,
                                                int(topline[19:20]) *
                                                100000)\
            + add_seconds
    except Exception:
        NordicParsingError("Couldn't read a date from sfile")
    # new_event.loc_mod_ind=topline[20]
    new_event.event_descriptions.append(EventDescription())
    new_event.event_descriptions[0].text = topline[21:23]
    # new_event.ev_id=topline[22]
    try:
        new_event.origins[0].latitude = float(topline[23:30])
        new_event.origins[0].longitude = float(topline[31:38])
        new_event.origins[0].depth = float(topline[39:43]) * 1000
    except ValueError:
        # The origin 'requires' a lat & long
        new_event.origins[0].latitude = None
        new_event.origins[0].longitude = None
        new_event.origins[0].depth = None
    # new_event.depth_ind = topline[44]
    # new_event.loc_ind = topline[45]
    new_event.creation_info = CreationInfo(agency_id=topline[45:48].strip())
    ksta = Comment(text='Number of stations=' + topline[49:51].strip())
    new_event.origins[0].comments.append(ksta)
    if _float_conv(topline[51:55]) is not None:
        new_event.origins[0].quality = OriginQuality(
            standard_error=_float_conv(topline[51:55]))
    # Read in magnitudes if they are there.
    for index in [59, 67, 75]:
        if not topline[index].isspace():
            new_event.magnitudes.append(Magnitude())
            new_event.magnitudes[-1].mag = _float_conv(
                topline[index - 3:index])
            new_event.magnitudes[-1].magnitude_type = \
                _nortoevmag(topline[index])
            new_event.magnitudes[-1].creation_info = \
                CreationInfo(agency_id=topline[index + 1:index + 4].strip())
            new_event.magnitudes[-1].origin_id = new_event.origins[0].\
                resource_id
    # Set the useful things like preferred magnitude and preferred origin
    new_event.preferred_origin_id = new_event.origins[0].resource_id
    try:
        # Select moment first, then local, then
        mag_filter = ['MW', 'Mw', 'ML', 'Ml', 'MB', 'Mb',
                      'MS', 'Ms', 'MC', 'Mc']
        _magnitudes = [(m.magnitude_type, m.resource_id)
                       for m in new_event.magnitudes]
        preferred_magnitude = sorted(_magnitudes,
                                     key=lambda x: mag_filter.index(x[0]))[0]
        new_event.preferred_magnitude_id = preferred_magnitude[1]
    except (ValueError, IndexError):
        # If there is a magnitude not specified in filter
        try:
            new_event.preferred_magnitude_id = new_event.magnitudes[0].\
                resource_id
        except IndexError:
            pass
    return new_event


def read_spectral_info(sfile):
    """
    Read spectral info from an sfile.

    :type sfile: str
    :param sfile: Sfile to read from.

    :returns:
        list of dictionaries of spectral information, units as in seisan
        manual, expect for logs which have been converted to floats.
    """
    with open(sfile, 'r') as f:
        spec_inf = _read_spectral_info(f=f)
    return spec_inf


def _read_spectral_info(f):
    """
    Internal spectral reader.

    :type f: file
    :param f: File open in read mode.

    :returns:
        list of dictionaries of spectral information, units as in
        seisan manual, expect for logs which have been converted to floats.
    """
    event = _readheader(f=f)
    f.seek(0)
    origin_date = UTCDateTime(event.origins[0].time.date)
    relevant_lines = []
    for line in f:
        if line[1:5] == 'SPEC':
            relevant_lines.append(line)
    spec_inf = {}
    if not relevant_lines:
        return spec_inf
    for line in relevant_lines:
        spec_str = line.strip()
        if spec_str[5:12] in ['AVERAGE', 'STANDARD_DEVIATION']:
            info = {}
            station = spec_str[5:12]
            channel = ''
            info['moment'] = _float_conv(spec_str[16:22])
            if info['moment'] is not None:
                info['moment'] = 10 ** info['moment']
            info['stress_drop'] = _float_conv(spec_str[24:30])
            info['spectral_level'] = _float_conv(spec_str[32:38])
            if info['spectral_level'] is not None:
                info['spectral_level'] = 10 ** info['spectral_level']
            info['corner_freq'] = _float_conv(spec_str[40:46])
            info['source_radius'] = _float_conv(spec_str[47:54])
            info['decay'] = _float_conv(spec_str[56:62])
            info['window_length'] = _float_conv(spec_str[64:70])
            info['moment_mag'] = _float_conv(spec_str[72:78])
            spec_inf[(station, channel)] = info
            continue
        station = spec_str[4:9].strip()
        channel = ''.join(spec_str[9:13].split())
        try:
            info = spec_inf[(station, channel)]
        except KeyError:
            info = {}
        if spec_str[14] == 'T':
            info['starttime'] = origin_date + \
                (int(spec_str[15:17]) * 3600) +\
                (int(spec_str[17:19]) * 60) + int(spec_str[19:21])
            if info['starttime'] < event.origins[0].time:
                # Wrong day, case of origin at end of day
                info['starttime'] += 86400
            info['kappa'] = _float_conv(spec_str[23:30])
            info['distance'] = _float_conv(spec_str[32:38])
            if spec_str[38:40] == 'VS':
                info['velocity'] = _float_conv(spec_str[40:46])
                info['velocity_type'] = 'S'
            elif spec_str[38:40] == 'VP':
                info['velocity'] = _float_conv(spec_str[40:46])
                info['velocity_type'] = 'P'
            else:
                warnings.warn('Only VP and VS spectral information'
                              ' implemented')
            info['density'] = _float_conv(spec_str[48:54])
            info['Q0'] = _float_conv(spec_str[56:62])
            info['QA'] = _float_conv(spec_str[64:70])
        elif spec_str[14] == 'M':
            info['moment'] = _float_conv(spec_str[16:22])
            if info['moment']:
                info['moment'] = 10 ** info['moment']
            info['stress_drop'] = _float_conv(spec_str[24:30])
            info['spectral_level'] = _float_conv(spec_str[32:38])
            if info['spectral_level']:
                info['spectral_level'] = 10 ** info['spectral_level']
            info['corner_freq'] = _float_conv(spec_str[40:46])
            info['source_radius'] = _float_conv(spec_str[47:54])
            info['decay'] = _float_conv(spec_str[56:62])
            info['window_length'] = _float_conv(spec_str[64:70])
            info['moment_mag'] = _float_conv(spec_str[72:78])
        spec_inf[(station, channel)] = info
    return spec_inf


def read_nordic(select_file, return_wavnames=False):
    """
    Read a catalog of events from a Nordic formatted select file.

    Generates a series of temporary files for each event in the select file.

    :type select_file: str
    :param select_file: Nordic formatted select.out file to open
    :type return_wavnames: bool
    :param return_wavnames:
        If True, will return the names of the waveforms that the events
        are associated with.

    :return: catalog of events
    :rtype: :class:`~obspy.core.event.event.Event`
    """
    catalog = Catalog()
    event_str = []
    if not hasattr(select_file, "readline"):
        try:
            f = open(select_file, 'r')
        except Exception:
            try:
                f = select_file.decode()
            except Exception:
                f = str(select_file)
    else:
        f = select_file
    wav_names = []
    for line in f:
        if len(line.rstrip()) > 0:
            event_str.append(line)
        elif len(event_str) > 0:
            tmp_sfile = io.StringIO()
            for event_line in event_str:
                tmp_sfile.write(event_line)
            new_event = _readheader(f=tmp_sfile)
            if return_wavnames:
                wav_names.append(_readwavename(f=tmp_sfile))
            catalog += _read_picks(f=tmp_sfile, new_event=new_event)
            event_str = []
    f.close()
    if return_wavnames:
        return catalog, wav_names
    return catalog


def _read_picks(f, new_event):
    """
    Internal pick reader. Use read_nordic instead.

    :type f: file
    :param f: File open in read mode
    :type wav_names: list
    :param wav_names: List of waveform files in the sfile
    :type new_event: :class:`~obspy.core.event.event.Event`
    :param new_event: event to associate picks with.

    :returns: :class:`~obspy.core.event.event.Event`
    """
    f.seek(0)
    evtime = new_event.origins[0].time
    pickline = []
    # Set a default, ignored later unless overwritten
    snr = None
    for line in f:
        if line[79] == '7':
            header = line
            break
    for line in f:
        if len(line.rstrip('\n').rstrip('\r')) in [80, 79] and \
           line[79] in ' 4\n':
            pickline += [line]
    for line in pickline:
        if line[18:28].strip() == '':  # If line is empty miss it
            continue
        weight = line[14]
        if weight == '_':
            phase = line[10:17]
            weight = 0
            polarity = ''
        else:
            phase = line[10:14].strip()
            polarity = line[16]
            if weight == ' ':
                weight = 0
        polarity_maps = {"": "undecidable", "C": "positive", "D": "negative"}
        try:
            polarity = polarity_maps[polarity]
        except KeyError:
            polarity = "undecidable"
        # It is valid nordic for the origin to be hour 23 and picks to be hour
        # 00 or 24: this signifies a pick over a day boundary.
        if int(line[18:20]) == 0 and evtime.hour == 23:
            day_add = 86400
            pick_hour = 0
        elif int(line[18:20]) == 24:
            day_add = 86400
            pick_hour = 0
        else:
            day_add = 0
            pick_hour = int(line[18:20])
        try:
            time = UTCDateTime(evtime.year, evtime.month, evtime.day,
                               pick_hour, int(line[20:22]),
                               float(line[23:28])) + day_add
        except ValueError:
            time = UTCDateTime(evtime.year, evtime.month, evtime.day,
                               int(line[18:20]), pick_hour,
                               float("0." + line[23:38].split('.')[1])) +\
                60 + day_add
            # Add 60 seconds on to the time, this copes with s-file
            # preference to write seconds in 1-60 rather than 0-59 which
            # datetime objects accept
        if header[57:60] == 'AIN':
            ain = _float_conv(line[57:60])
            warnings.warn('AIN: %s in header, currently unsupported' % ain)
        elif header[57:60] == 'SNR':
            snr = _float_conv(line[57:60])
        else:
            warnings.warn('%s is not currently supported' % header[57:60])
        # finalweight = _int_conv(line[68:70])
        # Create a new obspy.event.Pick class for this pick
        _waveform_id = WaveformStreamID(station_code=line[1:6].strip(),
                                        channel_code=line[6:8].strip(),
                                        network_code='NA')
        pick = Pick(waveform_id=_waveform_id, phase_hint=phase,
                    polarity=polarity, time=time)
        try:
            pick.onset = onsets[line[9]]
        except KeyError:
            pass
        if line[15] == 'A':
            pick.evaluation_mode = 'automatic'
        else:
            pick.evaluation_mode = 'manual'
        # Note these two are not always filled - velocity conversion not yet
        # implemented, needs to be converted from km/s to s/deg
        # if not velocity == 999.0:
            # new_event.picks[pick_index].horizontal_slowness = 1.0 / velocity
        if _float_conv(line[46:51]) is not None:
            pick.backazimuth = _float_conv(line[46:51])
        # Create new obspy.event.Amplitude class which references above Pick
        # only if there is an amplitude picked.
        if _float_conv(line[33:40]) is not None:
            _amplitude = Amplitude(generic_amplitude=_float_conv(line[33:40]),
                                   period=_float_conv(line[41:45]),
                                   pick_id=pick.resource_id,
                                   waveform_id=pick.waveform_id)
            if pick.phase_hint == 'IAML':
                # Amplitude for local magnitude
                _amplitude.type = 'AML'
                # Set to be evaluating a point in the trace
                _amplitude.category = 'point'
                # Default AML unit in seisan is nm (Page 139 of seisan
                # documentation, version 10.0)
                _amplitude.generic_amplitude /= 1e9
                _amplitude.unit = 'm'
                _amplitude.magnitude_hint = 'ML'
            else:
                # Generic amplitude type
                _amplitude.type = 'A'
            if snr:
                _amplitude.snr = snr
            new_event.amplitudes.append(_amplitude)
        elif _int_conv(line[28:33]) is not None:
            # Create an amplitude instance for code duration also
            _amplitude = Amplitude(generic_amplitude=_int_conv(line[28:33]),
                                   pick_id=pick.resource_id,
                                   waveform_id=pick.waveform_id)
            # Amplitude for coda magnitude
            _amplitude.type = 'END'
            # Set to be evaluating a point in the trace
            _amplitude.category = 'duration'
            _amplitude.unit = 's'
            _amplitude.magnitude_hint = 'Mc'
            if snr is not None:
                _amplitude.snr = snr
            new_event.amplitudes.append(_amplitude)
        # Create new obspy.event.Arrival class referencing above Pick
        if _float_conv(line[33:40]) is None:
            arrival = Arrival(phase=pick.phase_hint, pick_id=pick.resource_id)
            if weight is not None:
                arrival.time_weight = weight
            if _int_conv(line[60:63]) is not None:
                arrival.backazimuth_residual = _int_conv(line[60:63])
            if _float_conv(line[63:68]) is not None:
                arrival.time_residual = _float_conv(line[63:68])
            if _float_conv(line[70:75]) is not None:
                arrival.distance = kilometers2degrees(_float_conv(line[70:75]))
            if _int_conv(line[76:79]) is not None:
                arrival.azimuth = _int_conv(line[76:79])
            new_event.origins[0].arrivals.append(arrival)
        new_event.picks.append(pick)
    return new_event


def readwavename(sfile):
    """
    Extract the waveform filename from the s-file.

    Returns a list of waveform names found in the s-file as multiples can
    be present.

    :type sfile: str
    :param sfile: Path to the sfile

    :returns: List of strings of wave paths
    :rtype: list
    """
    with open(sfile, 'r') as f:
        wavenames = _readwavename(f=f)
    return wavenames


def _readwavename(f):
    """
    Internal wave-name reader.

    :type f: file
    :param f: File open in read-mode

    :return: list of wave-file names
    """
    wavename = []
    for line in f:
        if len(line) == 81 and line[79] == '6':
            wavename.append(line[1:79].strip())
    return wavename


def blanksfile(wavefile, evtype, userid, overwrite=False, evtime=None):
    """
    Generate an empty s-file with a populated header for a given waveform.

    :type wavefile: str
    :param wavefile: Wave-file to associate with this S-file, the timing of \
        the S-file will be taken from this file if evtime is not set.
    :type evtype: str
    :param evtype: Event type letter code, e.g. L, R, D
    :type userid: str
    :param userid: 4-character SEISAN USER ID
    :type overwrite: bool
    :param overwrite: Overwrite an existing S-file, default=False
    :type evtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param evtime: If given this will set the timing of the S-file

    :returns: str, S-file name
    """
    if evtime is None:
        try:
            st = read(wavefile)
            evtime = st[0].stats.starttime
        except Exception:
            raise NordicParsingError('Wavefile: ' + wavefile +
                                     ' is invalid, try again with real data.')
    # Check that user ID is the correct length
    if len(userid) != 4:
        raise NordicParsingError('User ID must be 4 characters long')
    # Check that evtype is one of L,R,D
    if evtype not in ['L', 'R', 'D']:
        raise NordicParsingError('Event type must be either L, R or D')

    # Generate s-file name in the format dd-hhmm-ss[L,R,D].Syyyymm
    sfile = str(evtime.day).zfill(2) + '-' + str(evtime.hour).zfill(2) +\
        str(evtime.minute).zfill(2) + '-' + str(evtime.second).zfill(2) +\
        evtype + '.S' + str(evtime.year) + str(evtime.month).zfill(2)
    # Check is sfile exists
    if os.path.isfile(sfile) and not overwrite:
        warnings.warn('Desired sfile: ' + sfile + ' exists, will not ' +
                      'overwrite')
        for i in range(1, 10):
            sfile = str(evtime.day).zfill(2) + '-' +\
                str(evtime.hour).zfill(2) +\
                str(evtime.minute).zfill(2) + '-' +\
                str(evtime.second + i).zfill(2) + evtype + '.S' +\
                str(evtime.year) + str(evtime.month).zfill(2)
            if not os.path.isfile(sfile):
                break
        else:
            msg = ('Tried generated files up to 20s in advance and found ' +
                   'all exist.')
            raise NordicParsingError(msg)
    with open(sfile, 'w') as f:
        # Write line 1 of s-file
        f.write(' ' + str(evtime.year) + ' ' + str(evtime.month).rjust(2) +
                str(evtime.day).rjust(2) + ' ' +
                str(evtime.hour).rjust(2) +
                str(evtime.minute).rjust(2) + ' ' +
                str(float(evtime.second)).rjust(4) + ' ' +
                evtype + '1'.rjust(58) + '\n')
        # Write line 2 of s-file
        output_time = datetime.datetime.now()
        f.write(' ACTION:ARG ' + str(output_time.year)[2:4] +
                '-' + str(output_time.month).zfill(2) + '-' +
                str(output_time.day).zfill(2) + ' ' +
                str(output_time.hour).zfill(2) + ':' +
                str(output_time.minute).zfill(2) + ' OP:' +
                userid.ljust(4) + ' STATUS:' + 'ID:'.rjust(18) +
                str(evtime.year) + str(evtime.month).zfill(2) +
                str(evtime.day).zfill(2) + str(evtime.hour).zfill(2) +
                str(evtime.minute).zfill(2) + str(evtime.second).zfill(2) +
                'I'.rjust(6) + '\n')
        # Write line 3 of s-file
        write_wavfile = os.path.basename(wavefile)
        f.write(' ' + write_wavfile + '6'.rjust(79 - len(write_wavfile)) +
                '\n')
        # Write final line of s-file
        f.write(" STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU" +
                " VELO AIN AR TRES W  DIS CAZ7\n")
    return sfile


def write_select(catalog, filename, userid='OBSP', evtype='L',
                 wavefiles=None):
    """
    Function to write a catalog to a select file in nordic format.

    :type catalog: :class:`~obspy.core.event.event.Catalog`
    :param catalog: A catalog of obspy events
    :type filename: str
    :param filename: Path to write to
    :type userid: str
    :param userid: Up to 4 character user ID
    :type evtype: str
    :param evtype:
        Single character string to describe the event, either L, R or D.
    :type wavefiles: list
    :param wavefiles:
        Waveforms to associate the events with, must be ordered in the same
         way as the events in the catalog.
    """
    if not wavefiles:
        wavefiles = ['DUMMY' for _i in range(len(catalog))]
    with open(filename, 'w') as fout:
        for event, wavfile in zip(catalog, wavefiles):
            select = io.StringIO()
            _write_nordic(event=event, filename=None, userid=userid,
                          evtype=evtype, wavefiles=wavfile,
                          string_io=select)
            select.seek(0)
            for line in select:
                fout.write(line)
            fout.write('\n')


def _write_nordic(event, filename, userid='OBSP', evtype='L', outdir='.',
                  wavefiles='DUMMY', explosion=False,
                  overwrite=True, string_io=None):
    """
    Write an :class:`~obspy.core.event.Event` to a nordic formatted s-file.

    :type event: :class:`~obspy.core.event.event.Event`
    :param event: A single obspy event
    :type filename: str
    :param filename:
        Filename to write to, can be None, and filename will be generated from
        the origin time in nordic format.
    :type userid: str
    :param userid: Up to 4 character user ID
    :type evtype: str
    :param evtype:
        Single character string to describe the event, either L, R or D.
    :type outdir: str
    :param outdir: Path to directory to write to
    :type wavefiles: list
    :param wavefiles: Waveforms to associate the nordic file with
    :type explosion: bool
    :param explosion:
        Note if the event is an explosion, will be marked by an E.
    :type overwrite: bool
    :param overwrite: force to overwrite old files, defaults to False
    :type string_io: io.StringIO
    :param string_io:
        If given, will write to the StringIO object in memory rather than to
        the filename.

    :returns: str: name of nordic file written

    .. note::

        Seisan can find waveforms either by their relative or absolute path, or
        by looking for the file recursively in directories within the WAV
        directory in your seisan install.  Because all lines need to be less
        than 79 characters long (fortran hangover) in the s-files, you will
        need to determine whether the full-path is okay or not.
    """
    # First we need to work out what to call the s-file and open it
    # Check that user ID is the correct length
    if len(userid) != 4:
        raise NordicParsingError('%s User ID must be 4 characters long'
                                 % userid)
    # Check that outdir exists
    if not os.path.isdir(outdir):
        raise NordicParsingError('Out path does not exist, I will not '
                                 'create this: ' + outdir)
    # Check that evtype is one of L,R,D
    if evtype not in ['L', 'R', 'D']:
        raise NordicParsingError('Event type must be either L, R or D')
    if explosion:
        evtype += 'E'
    # Check that there is one event
    if isinstance(event, Catalog) and len(event) == 1:
        event = event[0]
    elif isinstance(event, Event):
        event = event
    else:
        raise NordicParsingError('Needs a single event')
    if not isinstance(wavefiles, list):
        wavefiles = [str(wavefiles)]
    # Determine name from origin time
    try:
        evtime = event.origins[0].time
    except IndexError:
        msg = ('Need at least one origin with at least an origin time')
        raise NordicParsingError(msg)
    if not evtime:
        msg = ('event has an origin, but time is not populated.  ' +
               'This is required!')
        raise NordicParsingError(msg)
    # Attempt to cope with possible pre-existing files
    if not filename:
        range_list = []
        for i in range(30):  # Look +/- 30 seconds around origin time
            range_list.append(i)
            range_list.append(-1 * i)
        range_list = range_list[1:]
        for add_secs in range_list:
            sfilename = (evtime + add_secs).datetime.strftime('%d-%H%M-%S') +\
                evtype[0] + '.S' + (evtime + add_secs).\
                datetime.strftime('%Y%m')
            if not os.path.isfile(os.path.join(outdir, sfilename)):
                sfile_path = os.path.join(outdir, sfilename)
                break
            elif overwrite:
                sfile_path = os.path.join(outdir, sfilename)
                break
        else:
            raise NordicParsingError(os.path.join(outdir, sfilename) +
                                     ' already exists, will not overwrite')
    else:
        sfile_path = os.path.join(outdir, filename)
        sfilename = filename
    # Write the header info.
    if event.origins[0].latitude is not None:
        lat = '{0:.3f}'.format(event.origins[0].latitude)
    else:
        lat = ''
    if event.origins[0].longitude is not None:
        lon = '{0:.3f}'.format(event.origins[0].longitude)
    else:
        lon = ''
    if event.origins[0].depth is not None:
        depth = '{0:.1f}'.format(event.origins[0].depth / 1000)
    else:
        depth = ''
    if event.creation_info:
        try:
            agency = event.creation_info.get('agency_id')
            # If there is creation_info this may not raise an error annoyingly
            if agency is None:
                agency = ''
        except AttributeError:
            agency = ''
    else:
        agency = ''
    if len(agency) > 3:
        agency = agency[0:3]
    # Cope with differences in event uncertainty naming
    if event.origins[0].quality and event.origins[0].quality['standard_error']:
        timerms = '{0:.1f}'.format(event.origins[0].quality['standard_error'])
    else:
        timerms = '0.0'
    conv_mags = []
    for mag_ind in range(3):
        mag_info = {}
        try:
            mag_info['mag'] = '{0:.1f}'.format(
                event.magnitudes[mag_ind].mag) or ''
            mag_info['type'] = _evmagtonor(event.magnitudes[mag_ind].
                                           magnitude_type) or ''
            if event.magnitudes[0].creation_info:
                mag_info['agency'] = event.magnitudes[mag_ind].\
                    creation_info.agency_id or ''
            else:
                mag_info['agency'] = ''
        except IndexError:
            mag_info['mag'] = ''
            mag_info['type'] = ''
            mag_info['agency'] = ''
        conv_mags.append(mag_info)
    # Work out how many stations were used
    if len(event.picks) > 0:
        stations = [pick.waveform_id.station_code for pick in event.picks]
        ksta = str(len(set(stations)))
    else:
        ksta = ''
    if not string_io:
        sfile = open(sfile_path, 'w')
    else:
        sfile = string_io
    sfile.write(' ' + str(evtime.year) + ' ' +
                str(evtime.month).rjust(2) +
                str(evtime.day).rjust(2) + ' ' +
                str(evtime.hour).rjust(2) +
                str(evtime.minute).rjust(2) + ' ' +
                str(evtime.second).rjust(2) + '.' +
                str(evtime.microsecond).ljust(1)[0:1] + ' ' +
                evtype.ljust(2) + lat.rjust(7) + ' ' + lon.rjust(7) +
                depth.rjust(5) + agency.rjust(5) + ksta.rjust(3) +
                timerms.rjust(4) +
                conv_mags[0]['mag'].rjust(4) + conv_mags[0]['type'].rjust(1) +
                conv_mags[0]['agency'][0:3].rjust(3) +
                conv_mags[1]['mag'].rjust(4) + conv_mags[1]['type'].rjust(1) +
                conv_mags[1]['agency'][0:3].rjust(3) +
                conv_mags[2]['mag'].rjust(4) + conv_mags[2]['type'].rjust(1) +
                conv_mags[2]['agency'][0:3].rjust(3) + '1' + '\n')
    # Write line 2 of s-file
    sfile.write(' ACTION:ARG ' + str(datetime.datetime.now().year)[2:4] + '-' +
                str(datetime.datetime.now().month).zfill(2) + '-' +
                str(datetime.datetime.now().day).zfill(2) + ' ' +
                str(datetime.datetime.now().hour).zfill(2) + ':' +
                str(datetime.datetime.now().minute).zfill(2) + ' OP:' +
                userid.ljust(4) + ' STATUS:' + 'ID:'.rjust(18) +
                str(evtime.year) +
                str(evtime.month).zfill(2) +
                str(evtime.day).zfill(2) +
                str(evtime.hour).zfill(2) +
                str(evtime.minute).zfill(2) +
                str(evtime.second).zfill(2) +
                'I'.rjust(6) + '\n')
    # Write line 3 of s-file
    for wavefile in wavefiles:
        sfile.write(' ' + os.path.basename(wavefile) +
                    '6'.rjust(79 - len(wavefile)) + '\n')
    # Write final line of s-file
    sfile.write(' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU' +
                ' VELO AIN AR TRES W  DIS CAZ7\n')
    # Now call the populate sfile function
    if len(event.picks) > 0:
        newpicks = '\n'.join(nordpick(event))
        sfile.write(newpicks + '\n')
        sfile.write('\n'.rjust(81))
    if not string_io:
        sfile.close()
        return str(sfilename)
    else:
        return


def nordpick(event):
    """
    Format picks in an :class:`~obspy.core.event.event.Event` to nordic.

    :type event: :class:`~obspy.core.event.event.Event`
    :param event: A single obspy event.

    :returns: List of String

    .. note::

        Currently finalweight is unsupported, nor is velocity, or
        angle of incidence.  This is because
        :class:`~obspy.core.event.event.Event` stores slowness
        in s/deg and takeoff angle, which would require computation
        from the values stored in seisan.  Multiple weights are also
        not supported.
    """

    pick_strings = []
    for pick in event.picks:
        if not pick.waveform_id:
            msg = ('No waveform id for pick at time %s, skipping' % pick.time)
            warnings.warn(msg)
            continue
        # Convert string to short sting
        if pick.onset == 'impulsive':
            impulsivity = 'I'
        elif pick.onset == 'emergent':
            impulsivity = 'E'
        else:
            impulsivity = ' '

        # Convert string to short string
        if pick.polarity == 'positive':
            polarity = 'C'
        elif pick.polarity == 'negative':
            polarity = 'D'
        else:
            polarity = ' '
        # Extract velocity: Note that horizontal slowness in quakeML is stored
        # as s/deg
        if pick.horizontal_slowness is not None:
            # velocity = 1.0 / pick.horizontal_slowness
            velocity = ' '  # Currently this conversion is unsupported.
        else:
            velocity = ' '
        # Extract azimuth
        if pick.backazimuth is not None:
            azimuth = pick.backazimuth
        else:
            azimuth = ' '
        # Extract the correct arrival info for this pick - assuming only one
        # arrival per pick...
        arrival = [arrival for arrival in event.origins[0].arrivals
                   if arrival.pick_id == pick.resource_id]
        if len(arrival) > 0:
            arrival = arrival[0]
            # Extract weight - should be stored as 0-4, or 9 for seisan.
            if arrival.time_weight is not None:
                weight = int(arrival.time_weight)
            else:
                weight = '0'
            # Extract azimuth residual
            if arrival.backazimuth_residual is not None:
                azimuthres = int(arrival.backazimuth_residual)
            else:
                azimuthres = ' '
            # Extract time residual
            if arrival.time_residual is not None:
                timeres = arrival.time_residual
            else:
                timeres = ' '
            # Extract distance
            if arrival.distance is not None:
                distance = degrees2kilometers(arrival.distance)
                if distance >= 100.0:
                    distance = str(_int_conv(distance))
                elif 10.0 < distance < 100.0:
                    distance = _str_conv(round(distance, 1), 1)
                elif distance < 10.0:
                    distance = _str_conv(round(distance, 2), 2)
                else:
                    distance = _str_conv(distance, False)
            else:
                distance = ' '
            # Extract CAZ
            if arrival.azimuth is not None:
                caz = int(arrival.azimuth)
            else:
                caz = ' '
        else:
            caz = ' '
            distance = ' '
            timeres = ' '
            azimuthres = ' '
            azimuth = ' '
            weight = 0
        if not pick.phase_hint:
            # Cope with some authorities not providing phase hints :(
            phase_hint = ' '
        else:
            phase_hint = pick.phase_hint
        # Extract amplitude: note there can be multiple amplitudes, but they
        # should be associated with different picks.
        amplitude = [amplitude for amplitude in event.amplitudes
                     if amplitude.pick_id == pick.resource_id]
        if len(amplitude) > 0:
            if len(amplitude) > 1:
                msg = 'Nordic files need one pick for each amplitude, ' + \
                      'using the first amplitude only'
                warnings.warn(msg)
            amplitude = amplitude[0]
            # Determine type of amplitude
            if amplitude.type != 'END':
                # Extract period
                if amplitude.period is not None:
                    peri = amplitude.period
                    if peri < 10.0:
                        peri_round = 2
                    elif peri >= 10.0:
                        peri_round = 1
                    else:
                        peri_round = False
                else:
                    peri = ' '
                    peri_round = False
                # Extract amplitude and convert units
                if amplitude.generic_amplitude is not None:
                    amp = amplitude.generic_amplitude
                    if amplitude.unit in ['m', 'm/s', 'm/(s*s)', 'm*s']:
                        amp *= 1e9
                    # Otherwise we will assume that the amplitude is in counts
                else:
                    amp = None
                coda = ' '
                if amplitude.magnitude_hint.upper() == 'ML':
                    phase_hint = 'IAML'
                    impulsivity = ' '
            else:
                coda = int(amplitude.generic_amplitude)
                peri = ' '
                peri_round = False
                amp = None
        else:
            peri = ' '
            peri_round = False
            amp = None
            coda = ' '
        # If the weight is 0 then we don't need to print it
        if weight == 0 or weight == '0':
            weight = None  # this will return an empty string using _str_conv
        elif weight is not None:
            weight = int(weight)
        if pick.evaluation_mode == "automatic":
            eval_mode = "A"
        elif pick.evaluation_mode == "manual":
            eval_mode = " "
        else:
            warnings.warn("Evaluation mode %s is not supported"
                          % pick.evaluation_mode)
        # Generate a print string and attach it to the list
        channel_code = pick.waveform_id.channel_code or '   '
        pick_strings.append(' ' + pick.waveform_id.station_code.ljust(5) +
                            channel_code[0] + channel_code[-1] +
                            ' ' + impulsivity + phase_hint.ljust(4) +
                            _str_conv(weight).rjust(1) + eval_mode +
                            polarity.rjust(1) + ' ' +
                            str(pick.time.hour).rjust(2) +
                            str(pick.time.minute).rjust(2) +
                            str(pick.time.second).rjust(3) + '.' +
                            str(float(pick.time.microsecond) /
                            (10 ** 4)).split('.')[0].zfill(2) +
                            _str_conv(coda).rjust(5)[0:5] +
                            _str_conv(amp, rounded=1).rjust(7)[0:7] +
                            _str_conv(peri, rounded=peri_round).rjust(5) +
                            _str_conv(azimuth).rjust(6) +
                            _str_conv(velocity).rjust(5) +
                            _str_conv(' ').rjust(4) +
                            _str_conv(azimuthres).rjust(3) +
                            _str_conv(timeres, rounded=2).rjust(5)[0:5] +
                            _str_conv(' ').rjust(2) +
                            distance.rjust(5) +
                            _str_conv(caz).rjust(4) + ' ')
        # Note that currently finalweight is unsupported, nor is velocity, or
        # angle of incidence.  This is because obspy.event stores slowness in
        # s/deg and takeoff angle, which would require computation from the
        # values stored in seisan.  Multiple weights are also not supported in
        # Obspy.event
    return pick_strings


if __name__ == "__main__":
    import doctest
    doctest.testmod()
