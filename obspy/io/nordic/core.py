# -*- coding: utf-8 -*-
"""
Nordic file format support for ObsPy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

.. note::

    Pick time-residuals are handled in event.origins[0].arrivals, with
    the arrival.pick_id linking the arrival (which contain calculated
    information) with the pick.resource_id (where the pick contains only
    physical measured information).

.. versionchanged:: 1.2.0

    The number of stations used to calculate the origin was previously
    incorrectly stored in a comment. From version 1.2.0 this is now stored
    in `origin.quality.used_station_count`
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import warnings
import datetime
import os
import io
from math import sqrt

from obspy import UTCDateTime, read
from obspy.geodetics import kilometers2degrees, degrees2kilometers
from obspy.core.event import (
    Event, Origin, Magnitude, Catalog, EventDescription, CreationInfo,
    OriginQuality, OriginUncertainty, Pick, WaveformStreamID, Arrival,
    Amplitude, FocalMechanism, MomentTensor, NodalPlane, NodalPlanes,
    QuantityError, Tensor, ResourceIdentifier)
from obspy.io.nordic import NordicParsingError
from obspy.io.nordic.utils import (
    _int_conv, _str_conv, _float_conv, _evmagtonor, _nortoevmag,
    _get_line_tags, _km_to_deg_lat, _km_to_deg_lon)
from obspy.io.nordic.ellipse import Ellipse


POLARITY_MAPPING = {"": "undecidable", "C": "positive", "D": "negative"}
INV_POLARITY_MAPPING = {item: key for key, item in POLARITY_MAPPING.items()}
ONSET_MAPPING = {'I': 'impulsive', 'E': 'emergent'}
INV_ONSET_MAPPING = {item: key for key, item in ONSET_MAPPING.items()}
EVALUATION_MAPPING = {'A': 'automatic', ' ': 'manual'}
INV_EVALUTATION_MAPPING = {
    item: key for key, item in EVALUATION_MAPPING.items()}


def _is_sfile(sfile, encoding='latin-1'):
    """
    Basic test of whether the file is nordic format or not.

    Not exhaustive, but checks some of the basics.

    :type sfile: str
    :param sfile: Path to sfile
    :type encoding: str
    :param encoding: Encoding of the file.

    :rtype: bool
    """
    if not hasattr(sfile, "readline"):
        try:
            with open(sfile, 'r', encoding=encoding) as f:
                tags = _get_line_tags(f=f, report=False)
        except Exception:
            return False
    else:
        try:
            tags = _get_line_tags(f=sfile, report=False)
        except Exception:
            return False
    if tags is not None:
        try:
            head_line = tags['1'][0][0]
        except IndexError:
            return False
        try:
            sfile_seconds = int(head_line[16:18])
        except ValueError:
            return False
        if sfile_seconds == 60:
            sfile_seconds = 0
        try:
            UTCDateTime(
                int(head_line[1:5]), int(head_line[6:8]), int(head_line[8:10]),
                int(head_line[11:13]), int(head_line[13:15]), sfile_seconds,
                int(head_line[19:20]) * 100000)
            return True
        except Exception:
            return False
    else:
        return False


def readheader(sfile, encoding='latin-1'):
    """
    Read header information from a seisan nordic format S-file.

    :type sfile: str
    :param sfile: Path to the s-file
    :type encoding: str
    :param encoding: Encoding for file, used to decode from bytes to string

    :returns: :class:`~obspy.core.event.event.Event`
    """
    with open(sfile, 'r', encoding=encoding) as f:
        tagged_lines = _get_line_tags(f)
        if len(tagged_lines['1']) == 0:
            raise NordicParsingError("No header lines found")
        header = _readheader(head_lines=tagged_lines['1'])
    return header


def _readheader(head_lines):
    """
    Internal header reader.
    :type head_lines: list
    :param head_lines:
        List of tuples of (strings, line-number) of the header lines.

    :returns: :class:`~obspy.core.event.event.Event`
    """
    # There are two possible types of origin line, one with all info, and a
    # subsequent one for additional magnitudes.
    head_lines.sort(key=lambda tup: tup[1])
    # Construct a rough catalog, then merge events together to cope with
    # multiple origins
    _cat = Catalog()
    for line in head_lines:
        _cat.append(_read_origin(line=line[0]))
    new_event = _cat.events.pop(0)
    for event in _cat:
        matched = False
        origin_times = [origin.time for origin in new_event.origins]
        if event.origins[0].time in origin_times:
            origin_index = origin_times.index(event.origins[0].time)
            agency = new_event.origins[origin_index].creation_info.agency_id
            if event.creation_info.agency_id == agency:
                event_desc = new_event.event_descriptions[origin_index].text
                if event.event_descriptions[0].text == event_desc:
                    matched = True
        new_event.magnitudes.extend(event.magnitudes)
        if not matched:
            new_event.origins.append(event.origins[0])
            new_event.event_descriptions.append(event.event_descriptions[0])
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


def _read_origin(line):
    """
    Read one origin (type 1) line.

    :param str line: Origin format (type 1) line
    :return: `~obspy.core.event.Event`
    """
    new_event = Event()
    try:
        sfile_seconds = line[16:20].strip()
        if len(sfile_seconds) == 0:
            sfile_seconds = 0.0
        else:
            sfile_seconds = float(sfile_seconds)
        new_event.origins.append(Origin())
        new_event.origins[0].time = UTCDateTime(
            int(line[1:5]), int(line[6:8]), int(line[8:10]),
            int(line[11:13]), int(line[13:15]), 0, 0) + sfile_seconds
    except Exception:
        raise NordicParsingError("Couldn't read a date from sfile")
    # new_event.loc_mod_ind=line[20]
    new_event.event_descriptions.append(EventDescription(text=line[21:23]))
    for key, _slice in [('latitude', slice(23, 30)),
                        ('longitude', slice(30, 38)),
                        ('depth', slice(38, 43))]:
        try:
            new_event.origins[0].__dict__[key] = float(line[_slice])
        except ValueError:
            new_event.origins[0].__dict__[key] = None
    if new_event.origins[0].depth:
        new_event.origins[0].depth *= 1000.
    if line[43].strip():
        warnings.warn("Depth indicator {0} has not been mapped "
                      "to the event".format(line[43]))
    if line[44].strip():
        warnings.warn("Origin location indicator {0} has not been mapped "
                      "to the event".format(line[44]))
    if line[10] == "F":
        new_event.origins[0].time_fixed = True
    new_event.creation_info = CreationInfo(agency_id=line[45:48].strip())
    new_event.origins[0].creation_info = CreationInfo(
        agency_id=line[45:48].strip())
    used_station_count = line[49:51].strip()
    if used_station_count != '':
        new_event.origins[0].quality = OriginQuality(
            used_station_count=int(used_station_count))
    timeres = _float_conv(line[51:55])
    if timeres is not None:
        if new_event.origins[0].quality is not None:
            new_event.origins[0].quality.standard_error = timeres
        else:
            new_event.origins[0].quality = OriginQuality(
                standard_error=timeres)
    # Read in magnitudes if they are there.
    magnitudes = []
    magnitudes.extend(_read_mags(line, new_event))
    new_event.magnitudes = magnitudes
    return new_event


def _read_mags(line, event):
    """
    Read the magnitude info from a Nordic header line. Convenience function
    """
    magnitudes = []
    for index in [59, 67, 75]:
        if not line[index].isspace():
            magnitudes.append(Magnitude(
                mag=_float_conv(line[index - 3:index]),
                magnitude_type=_nortoevmag(line[index]),
                creation_info=CreationInfo(
                    agency_id=line[index + 1:index + 4].strip()),
                origin_id=event.origins[0].resource_id))
    return magnitudes


def read_spectral_info(sfile, encoding='latin-1'):
    """
    Read spectral info from an sfile.

    :type sfile: str
    :param sfile: Sfile to read from.
    :type encoding: str
    :param encoding: Encoding for file, used to decode from bytes to string

    :returns:
        list of dictionaries of spectral information, units as in seisan
        manual, expect for logs which have been converted to floats.
    """
    with open(sfile, 'r', encoding=encoding) as f:
        tagged_lines = _get_line_tags(f=f)
        spec_inf = _read_spectral_info(tagged_lines=tagged_lines)
    return spec_inf


def _read_spectral_info(tagged_lines, event=None):
    """
    Internal spectral reader.

    :type tagged_lines: dict
    :param tagged_lines: dictionary of tagged lines
    :type event: :class:`~obspy.core.event.Event`
    :param event: Event to associate spectral info with

    :returns:
        list of dictionaries of spectral information, units as in
        seisan manual, expect for logs which have been converted to floats.
    """
    if '3' not in tagged_lines.keys():
        return {}
    if event is None:
        event = _readheader(head_lines=tagged_lines['1'])
    origin_date = UTCDateTime(event.origins[0].time.date)
    relevant_lines = []
    for line in tagged_lines['3']:
        if line[0][1:5] == 'SPEC':
            relevant_lines.append(line)
    relevant_lines = [line[0] for line in
                      sorted(relevant_lines, key=lambda tup: tup[1])]
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


def read_nordic(select_file, return_wavnames=False, encoding='latin-1'):
    """
    Read a catalog of events from a Nordic formatted select file.

    Generates a series of temporary files for each event in the select file.

    :type select_file: str
    :param select_file: Nordic formatted select.out file to open
    :type return_wavnames: bool
    :param return_wavnames:
        If True, will return the names of the waveforms that the events
        are associated with.
    :type encoding: str
    :param encoding: Encoding for file, used to decode from bytes to string

    :return: catalog of events
    :rtype: :class:`~obspy.core.event.event.Event`
    """
    if not hasattr(select_file, "readline"):
        try:
            f = open(select_file, 'r', encoding=encoding)
        except Exception:
            try:
                f = select_file.decode(encoding)
            except Exception:
                f = str(select_file)
    else:
        f = select_file
    wav_names = []
    event_str = []
    catalog = Catalog()
    for line in f:
        if len(line.rstrip()) > 0:
            event_str.append(line)
        elif len(event_str) > 0:
            catalog, wav_names = _extract_event(
                event_str=event_str, catalog=catalog, wav_names=wav_names,
                return_wavnames=return_wavnames)
            event_str = []
    f.close()
    if len(event_str) > 0:
        # May occur if the last line of the file is not blank as it should be
        catalog, wav_names = _extract_event(
            event_str=event_str, catalog=catalog, wav_names=wav_names,
            return_wavnames=return_wavnames)
    if return_wavnames:
        return catalog, wav_names
    for event in catalog:
        event.scope_resource_ids()
    return catalog


def _extract_event(event_str, catalog, wav_names, return_wavnames=False):
    """
    Helper to extract event info from a list of line strings.

    :param event_str: List of lines from sfile
    :type event_str: list of str
    :param catalog: Catalog to append the event to
    :type catalog: `obspy.core.event.Catalog`
    :param wav_names: List of waveform names
    :type wav_names: list
    :param return_wavnames: Whether to extract the waveform name or not.
    :type return_wavnames: bool

    :return: Adds event to catalog and returns. Works in place on catalog.
    """
    tmp_sfile = io.StringIO()
    for event_line in event_str:
        tmp_sfile.write(event_line)
    tagged_lines = _get_line_tags(f=tmp_sfile)
    new_event = _readheader(head_lines=tagged_lines['1'])
    new_event = _read_uncertainty(tagged_lines, new_event)
    new_event = _read_highaccuracy(tagged_lines, new_event)
    new_event = _read_focal_mechanisms(tagged_lines, new_event)
    new_event = _read_moment_tensors(tagged_lines, new_event)
    if return_wavnames:
        wav_names.append(_readwavename(tagged_lines=tagged_lines['6']))
    new_event = _read_picks(tagged_lines=tagged_lines, new_event=new_event)
    catalog += new_event
    return catalog, wav_names


def _read_uncertainty(tagged_lines, event):
    """
    Read hyp uncertainty line.

    :param tagged_lines: Lines keyed by line type
    :type tagged_lines: dict
    :returns: updated event
    :rtype: :class:`~obspy.core.event.event.Event`
    """
    if 'E' not in tagged_lines.keys():
        return event
    # In principle there shouldn't be more than one error line, but I think
    # there can be - need to associate the correct error
    line = tagged_lines['E'][0][0]
    # TODO: Convert this to ConfidenceEllipsoid
    errors = {'x_err': None}
    try:
        errors = {'x_err': _float_conv(line[24:30]),
                  'y_err': _float_conv(line[32:38]),
                  'z_err': _float_conv(line[38:43]),
                  'xy_cov': _float_conv(line[43:55]),
                  'xz_cov': _float_conv(line[55:67]),
                  'yz_cov': _float_conv(line[67:79])}
    except ValueError:
        pass
    orig = event.origins[0]
    if errors['x_err'] is not None:
        e = Ellipse.from_uncerts(errors['x_err'],
                                 errors['y_err'],
                                 errors['xy_cov'])
        if e:
            orig.origin_uncertainty = OriginUncertainty(
                max_horizontal_uncertainty=e.a * 1000.,
                min_horizontal_uncertainty=e.b * 1000.,
                azimuth_max_horizontal_uncertainty=e.theta,
                preferred_description="uncertainty ellipse")
            orig.latitude_errors = QuantityError(
                _km_to_deg_lat(errors['y_err']))
            orig.longitude_errors = QuantityError(
                _km_to_deg_lon(errors['x_err'], orig.latitude))
            orig.depth_errors = QuantityError(errors['z_err'] * 1000.)
    try:
        gap = int(line[5:8])
    except ValueError:
        gap = None
    try:
        orig.time_errors = QuantityError(float(line[14:20]))
    except ValueError:
        pass
    if orig.quality:
        orig.quality.azimuthal_gap = gap
    else:
        orig.quality = OriginQuality(azimuthal_gap=gap)
    return event


def _read_highaccuracy(tagged_lines, event):
    """
    Read high accuracy origin line.

    :param tagged_lines: Lines keyed by line type
    :type tagged_lines: dict
    :returns: updated event
    :rtype: :class:`~obspy.core.event.event.Event`
    """
    if 'H' not in tagged_lines.keys():
        return event
    # In principle there shouldn't be more than one high precision line
    line = tagged_lines['H'][0][0]
    try:
        dt = {'Y': _int_conv(line[1:5]),
              'MO': _int_conv(line[6:8]),
              'D': _int_conv(line[8:10]),
              'H': _int_conv(line[11:13]),
              'MI': _int_conv(line[13:15]),
              'S': _float_conv(line[16:23])}
    except ValueError:
        pass
    try:
        ev_time = UTCDateTime(dt['Y'], dt['MO'], dt['D'],
                              dt['H'], dt['MI'], 0, 0) + dt['S']
        if abs(event.origins[0].time - ev_time) < 0.1:
            event.origins[0].time = ev_time
        else:
            print('High accuracy time differs from normal time by >0.1s')
    except ValueError:
        pass
    try:
        values = {'latitude': _float_conv(line[23:32]),
                  'longitude': _float_conv(line[33:43]),
                  'depth': _float_conv(line[44:52]),
                  'rms': _float_conv(line[53:59])}
    except ValueError:
        pass
    if values['latitude'] is not None:
        event.origins[0].latitude = values['latitude']
    if values['longitude'] is not None:
        event.origins[0].longitude = values['longitude']
    if values['depth'] is not None:
        event.origins[0].depth = values['depth'] * 1000.
    if values['rms'] is not None:
        if event.origins[0].quality is not None:
            event.origins[0].quality.standard_error = values['rms']
        else:
            event.origins[0].quality = OriginQuality(
                standard_error=values['rms'])
    return event


def _read_focal_mechanisms(tagged_lines, event):
    """
    Read focal mechanism info from s-file

    :param tagged_lines: Lines keyed by line type
    :type tagged_lines: dict
    :returns: updated event
    :rtype: :class:`~obspy.core.event.event.Event`
    """
    if 'F' not in tagged_lines.keys():
        return event
    UserWarning("Found focal-mechanism info: reading amplitude-ratio fit,"
                "number of bad polarities and number of bad amplitude ratios"
                "is not implemented.")
    for line, line_num in tagged_lines['F']:
        nodal_p = NodalPlane(strike=float(line[0:10]), dip=float(line[10:20]),
                             rake=float(line[20:30]))
        try:
            # Apparently these don't have to be filled.
            nodal_p.strike_errors = QuantityError(float(line[30:35]))
            nodal_p.dip_errors = QuantityError(float(line[35:40]))
            nodal_p.rake_errors = QuantityError(float(line[40:45]))
        except ValueError:
            pass
        fm = FocalMechanism(nodal_planes=NodalPlanes(nodal_plane_1=nodal_p))
        try:
            fm.method_id = ResourceIdentifier(
                "smi:nc.anss.org/focalMehcanism/" + line[70:77].strip())
            fm.creation_info = CreationInfo(agency_id=line[66:69].strip)
            fm.misfit = float(line[45:50])
            fm.station_distribution_ratio = float(line[50:55])
        except ValueError:
            pass
        event.focal_mechanisms.append(fm)
    return event


def _read_moment_tensors(tagged_lines, event):
    """
    Read moment tensors from s-file

    :param tagged_lines: Lines keyed by line type
    :type tagged_lines: dict
    :returns: updated event
    :rtype: :class:`~obspy.core.event.event.Event`
    """
    if 'M' not in tagged_lines.keys():
        return event
    # Group moment tensor lines together
    mt_lines = sorted(tagged_lines['M'], key=lambda tup: tup[1])
    for mt_ind in range(len(mt_lines) // 2):
        mt_line_1 = mt_lines[mt_ind * 2][0]
        mt_line_2 = mt_lines[(mt_ind * 2) + 1][0]
        if not str(mt_line_2[1:3]) == str('MT'):
            raise NordicParsingError("Matching moment tensor lines not found.")
        sfile_seconds = int(mt_line_1[16:18])
        if sfile_seconds == 60:
            sfile_seconds = 0
            add_seconds = 60
        else:
            add_seconds = 0
        ori_time = UTCDateTime(
            int(mt_line_1[1:5]), int(mt_line_1[6:8]), int(mt_line_1[8:10]),
            int(mt_line_1[11:13]), int(mt_line_1[13:15]), sfile_seconds,
            int(mt_line_1[19:20]) * 100000) + add_seconds
        event.origins.append(Origin(
            time=ori_time, latitude=float(mt_line_1[23:30]),
            longitude=float(mt_line_1[30:38]),
            depth=float(mt_line_1[38:43]) * 1000,
            creation_info=CreationInfo(agency_id=mt_line_1[45:48].strip())))
        event.magnitudes.append(Magnitude(
            mag=float(mt_line_1[55:59]),
            magnitude_type=_nortoevmag(mt_line_1[59]),
            creation_info=CreationInfo(agency_id=mt_line_1[60:63].strip()),
            origin_id=event.origins[-1].resource_id))
        event.focal_mechanisms.append(FocalMechanism(
            moment_tensor=MomentTensor(
                derived_origin_id=event.origins[-1].resource_id,
                moment_magnitude_id=event.magnitudes[-1].resource_id,
                scalar_moment=float(mt_line_2[52:62]), tensor=Tensor(
                    m_rr=float(mt_line_2[3:9]), m_tt=float(mt_line_2[10:16]),
                    m_pp=float(mt_line_2[17:23]), m_rt=float(mt_line_2[24:30]),
                    m_rp=float(mt_line_2[31:37]),
                    m_tp=float(mt_line_2[38:44])),
                method_id=ResourceIdentifier(
                    "smi:nc.anss.org/momentTensor/" + mt_line_1[70:77].strip()
                ))))
    return event


def _read_picks(tagged_lines, new_event):
    """
    Internal pick reader. Use read_nordic instead.

    :type tagged_lines: dict
    :param tagged_lines: Lines keyed by line type
    :type new_event: :class:`~obspy.core.event.event.Event`
    :param new_event: event to associate picks with.

    :returns: :class:`~obspy.core.event.event.Event`
    """
    evtime = new_event.origins[0].time
    pickline = []
    # pick-lines can be tagged by either ' ' or '4'
    tags = [' ', '4']
    for tag in tags:
        try:
            pickline.extend(
                [tup[0] for tup in sorted(
                    tagged_lines[tag], key=lambda tup: tup[1])])
        except KeyError:
            pass
    header = sorted(tagged_lines['7'], key=lambda tup: tup[1])[0][0]
    for line in pickline:
        ain, snr = (None, None)
        if line[18:28].strip() == '':  # If line is empty miss it
            continue
        if len(line) < 80:
            line = line.ljust(80)  # Pick-lines without a tag may be short.
        weight = line[14]
        if weight not in ' 012349_':  # Long phase name
            weight = line[8]
            if weight == ' ':
                weight = 0
            phase = line[10:17].strip()
            polarity = ''
        elif weight == '_':
            phase = line[10:17]
            weight = 0
            polarity = ''
        else:
            phase = line[10:14].strip()
            polarity = line[16]
            if weight == ' ':
                weight = 0
        polarity = POLARITY_MAPPING.get(polarity, None)  # Empty could be None
        # or undecidable.
        # It is valid nordic for the origin to be hour 23 and picks to be hour
        # 00 or 24: this signifies a pick over a day boundary.
        pick_hour = int(line[18:20])
        pick_minute = int(line[20:22])
        pick_seconds = float(line[22:29])  # 29 should be blank, but sometimes
        # SEISAN appears to overflow here, see #2348
        if pick_hour == 0 and evtime.hour == 23:
            day_add = 86400
        elif pick_hour >= 24:  # Nordic supports up to 48 hours advanced.
            day_add = 86400
            pick_hour -= 24
        else:
            day_add = 0
        time = UTCDateTime(
            year=evtime.year, month=evtime.month, day=evtime.day,
            hour=pick_hour, minute=pick_minute) + (pick_seconds + day_add)
        if header[57:60] == 'AIN':
            ain = _float_conv(line[57:60])
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
            pick.onset = ONSET_MAPPING[line[9]]
        except KeyError:
            pass
        pick.evaluation_mode = EVALUATION_MAPPING.get(line[15], "manual")
        # Note these two are not always filled - velocity conversion not yet
        # implemented, needs to be converted from km/s to s/deg
        # if not velocity == 999.0:
        #     new_event.picks[pick_index].horizontal_slowness = 1.0 / velocity
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
        elif _int_conv(line[29:33]) is not None:
            # Create an amplitude instance for coda duration also
            _amplitude = Amplitude(generic_amplitude=_int_conv(line[29:33]),
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
            if ain is not None:
                arrival.takeoff_angle = ain
            new_event.origins[0].arrivals.append(arrival)
        new_event.picks.append(pick)
    return new_event


def readwavename(sfile, encoding='latin-1'):
    """
    Extract the waveform filename from the s-file.

    Returns a list of waveform names found in the s-file as multiples can
    be present.

    :type sfile: str
    :param sfile: Path to the sfile
    :type encoding: str
    :param encoding: Encoding for file, used to decode from bytes to string

    :returns: List of strings of wave paths
    :rtype: list
    """
    with open(sfile, 'r', encoding=encoding) as f:
        tagged_lines = _get_line_tags(f=f)
        if len(tagged_lines['6']) == 0:
            msg = ('No waveform files in sfile %s' % sfile)
            warnings.warn(msg)
        wavenames = _readwavename(tagged_lines=tagged_lines['6'])
    return wavenames


def _readwavename(tagged_lines):
    """
    Internal wave-name reader.

    :type tagged_lines: list
    :param tagged_lines:
        List of tuples of (strings, line-number) of the waveform lines.

    :return: list of wave-file names
    """
    wavename = []
    for line in tagged_lines:
        wavename.append(line[0][1:79].strip())
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
                 wavefiles=None, high_accuracy=True):
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
    :type high_accuracy: bool
    :param high_accuracy:
        Whether to output pick seconds at 6.3f (high_accuracy) or
        5.2f (standard)
    """
    if not wavefiles:
        wavefiles = ['DUMMY' for _i in range(len(catalog))]
    with open(filename, 'w') as fout:
        for event, wavfile in zip(catalog, wavefiles):
            select = io.StringIO()
            _write_nordic(event=event, filename=None, userid=userid,
                          evtype=evtype, wavefiles=wavfile,
                          string_io=select, high_accuracy=high_accuracy)
            select.seek(0)
            for line in select:
                fout.write(line)
            fout.write('\n')


def _write_nordic(event, filename, userid='OBSP', evtype='L', outdir='.',
                  wavefiles='DUMMY', explosion=False,
                  overwrite=True, string_io=None, high_accuracy=True):
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
    :type high_accuracy: bool
    :param high_accuracy:
        Whether to output pick seconds at 6.3f (high_accuracy) or
        5.2f (standard)

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
        origin = event.preferred_origin() or event.origins[0]
    except IndexError:
        msg = 'Need at least one origin with at least an origin time'
        raise NordicParsingError(msg)
    evtime = origin.time
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
    if origin.latitude is not None:
        lat = '{0:.3f}'.format(origin.latitude)
    else:
        lat = ''
    if origin.longitude is not None:
        lon = '{0:.3f}'.format(origin.longitude)
    else:
        lon = ''
    if origin.depth is not None:
        depth = '{0:.1f}'.format(origin.depth / 1000.0)
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
    if origin.quality and origin.quality['standard_error']:
        timerms = '{0:.1f}'.format(origin.quality['standard_error'])
    else:
        timerms = '0.0'
    conv_mags = []
    # Get up to six magnitudes
    mt_ids = []
    if hasattr(event, 'focal_mechanisms'):
        for fm in event.focal_mechanisms:
            if hasattr(fm, 'moment_tensor') and hasattr(
               fm.moment_tensor, 'moment_magnitude_id'):
                mt_ids.append(fm.moment_tensor.moment_magnitude_id)
    for mag_ind in range(6):
        mag_info = {}
        try:
            if event.magnitudes[mag_ind].resource_id in mt_ids:
                raise IndexError("Repeated magnitude")
                # This magnitude will get put in with the moment tensor
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
            mag_info.update({'mag': '', 'type': '', 'agency': ''})
        conv_mags.append(mag_info)
    conv_mags.sort(key=lambda d: d['mag'], reverse=True)
    if conv_mags[3]['mag'] == '':
        conv_mags = conv_mags[0:3]
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
    sfile.write(
        " {0} {1}{2} {3}{4} {5}.{6} {7}{8}{9}{10}  {11}{12}{13}{14}{15}{16}"
        "{17}{18}{19}{20}{21}{22}1\n".format(
            evtime.year, str(evtime.month).rjust(2), str(evtime.day).rjust(2),
            str(evtime.hour).rjust(2), str(evtime.minute).rjust(2),
            str(evtime.second).rjust(2), str(evtime.microsecond).ljust(1)[0:1],
            evtype.ljust(2), lat.rjust(7), lon.rjust(8), depth.rjust(5),
            agency.rjust(3)[0:3], ksta.rjust(3), timerms.rjust(4),
            conv_mags[0]['mag'].rjust(4), conv_mags[0]['type'].rjust(1),
            conv_mags[0]['agency'][0:3].rjust(3),
            conv_mags[1]['mag'].rjust(4), conv_mags[1]['type'].rjust(1),
            conv_mags[1]['agency'][0:3].rjust(3),
            conv_mags[2]['mag'].rjust(4), conv_mags[2]['type'].rjust(1),
            conv_mags[2]['agency'][0:3].rjust(3)))
    if len(conv_mags) > 3:
        sfile.write(
            " {0} {1}{2} {3}{4} {5}.{6} {7}                      {8}       "
            "{9}{10}{11}{12}{13}{14}{15}{16}{17}1\n".format(
                evtime.year, str(evtime.month).rjust(2),
                str(evtime.day).rjust(2), str(evtime.hour).rjust(2),
                str(evtime.minute).rjust(2), str(evtime.second).rjust(2),
                str(evtime.microsecond).ljust(1)[0:1], evtype.ljust(2),
                agency.rjust(3)[0:3],
                conv_mags[3]['mag'].rjust(4), conv_mags[3]['type'].rjust(1),
                conv_mags[3]['agency'][0:3].rjust(3),
                conv_mags[4]['mag'].rjust(4), conv_mags[4]['type'].rjust(1),
                conv_mags[4]['agency'][0:3].rjust(3),
                conv_mags[5]['mag'].rjust(4), conv_mags[5]['type'].rjust(1),
                conv_mags[5]['agency'][0:3].rjust(3)))
    # Write hyp error line
    try:
        sfile.write(_write_hyp_error_line(origin) + '\n')
    except (NordicParsingError, TypeError):
        pass
    # Write fault plane solution
    if hasattr(event, 'focal_mechanisms') and len(event.focal_mechanisms) > 0:
        for focal_mechanism in event.focal_mechanisms:
            try:
                sfile.write(
                    _write_focal_mechanism_line(focal_mechanism) + '\n')
            except AttributeError:
                pass
        # Write moment tensor solution
        for focal_mechanism in event.focal_mechanisms:
            try:
                sfile.write(
                    _write_moment_tensor_line(focal_mechanism) + '\n')
            except AttributeError:
                pass
    # Write line 2 (type: I) of s-file
    sfile.write(
        " Action:ARG {0} OP:{1} STATUS:               ID:{2}     I\n".format(
            datetime.datetime.now().strftime("%y-%m-%d %H:%M"),
            userid.ljust(4)[0:4], evtime.strftime("%Y%m%d%H%M%S")))
    # Write line-type 6 of s-file
    for wavefile in wavefiles:
        sfile.write(' ' + os.path.basename(wavefile) +
                    '6'.rjust(79 - len(os.path.basename(wavefile))) + '\n')
    # Write final line of s-file
    sfile.write(' STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU' +
                ' VELO AIN AR TRES W  DIS CAZ7\n')
    # Now call the populate sfile function
    if len(event.picks) > 0:
        newpicks = '\n'.join(nordpick(event, high_accuracy=high_accuracy))
        sfile.write(newpicks + '\n')
        sfile.write('\n'.rjust(81))
    if not string_io:
        sfile.close()
        return str(sfilename)
    else:
        return


def _write_moment_tensor_line(focal_mechanism):
    """
    Generate the two lines required for moment tensor solutions in Nordic.
    """
    # First line contains hypocenter info, second contains tensor info
    lines = [list(' ' * 79 + 'M'), list(' ' * 79 + 'M')]
    # Get the origin associated with the moment tensor
    origin = focal_mechanism.moment_tensor.derived_origin_id.\
        get_referred_object()
    magnitude = focal_mechanism.moment_tensor.moment_magnitude_id.\
        get_referred_object()
    # Sort out the first line
    lines[0][1:5] = str(origin.time.year).rjust(4)
    lines[0][6:8] = str(origin.time.month).rjust(2)
    lines[0][8:10] = str(origin.time.day).rjust(2)
    lines[0][11:13] = str(origin.time.hour).rjust(2)
    lines[0][13:15] = str(origin.time.minute).rjust(2)
    lines[0][16:20] = str(origin.time.second).rjust(2) + '.' +\
        str(origin.time.microsecond).ljust(1)[0:1]
    lines[0][23:30] = _str_conv(origin.latitude, 3).rjust(7)
    lines[0][30:38] = _str_conv(origin.longitude, 3).rjust(8)
    lines[0][38:43] = _str_conv(origin.depth / 1000.0, 1).rjust(5)
    if hasattr(origin, 'creation_info') and hasattr(
            origin.creation_info, 'agency_id'):
        lines[0][45:48] = origin.creation_info.agency_id.rjust(3)[0:3]
    lines[0][55:59] = _str_conv(magnitude.mag, 1).rjust(4)
    lines[0][59] = _evmagtonor(magnitude.magnitude_type)
    if hasattr(magnitude, 'creation_info') and hasattr(
            magnitude.creation_info, 'agency_id'):
        lines[0][60:63] = magnitude.creation_info.agency_id.rjust(3)[0:3]
    lines[0][70:77] = (str(
        focal_mechanism.moment_tensor.method_id).split('/')[-1]).rjust(7)
    # Sort out the second line
    lines[1][1:3] = 'MT'
    lines[1][3:9] = _str_conv(
        focal_mechanism.moment_tensor.tensor.m_rr, 3)[0:6].rjust(6)
    lines[1][10:16] = _str_conv(
        focal_mechanism.moment_tensor.tensor.m_tt, 3)[0:6].rjust(6)
    lines[1][17:23] = _str_conv(
        focal_mechanism.moment_tensor.tensor.m_pp, 3)[0:6].rjust(6)
    lines[1][24:30] = _str_conv(
        focal_mechanism.moment_tensor.tensor.m_rt, 3)[0:6].rjust(6)
    lines[1][31:37] = _str_conv(
        focal_mechanism.moment_tensor.tensor.m_rp, 3)[0:6].rjust(6)
    lines[1][38:44] = _str_conv(
        focal_mechanism.moment_tensor.tensor.m_tp, 3)[0:6].rjust(6)
    if hasattr(magnitude, 'creation_info') and hasattr(
            magnitude.creation_info, 'agency_id'):
        lines[1][45:48] = magnitude.creation_info.agency_id.rjust(3)[0:3]
    lines[1][48] = 'S'
    lines[1][52:62] = (
        "%.3e" % focal_mechanism.moment_tensor.scalar_moment).rjust(10)
    lines[1][70:77] = (
        str(focal_mechanism.moment_tensor.method_id).split('/')[-1]).rjust(7)
    return '\n'.join([''.join(line) for line in lines])


def _write_focal_mechanism_line(focal_mechanism):
    """
    Get the line for a focal-mechanism
    """
    nodal_plane = focal_mechanism.nodal_planes.nodal_plane_1
    line = list(' ' * 79 + 'F')
    line[0:10] = (_str_conv(nodal_plane.strike, 1)).rjust(10)
    line[10:20] = (_str_conv(nodal_plane.dip, 1)).rjust(10)
    line[20:30] = (_str_conv(nodal_plane.rake, 1)).rjust(10)
    try:
        line[30:35] = (_str_conv(nodal_plane.strike_errors.uncertainty,
                                 1)).rjust(5)
        line[35:40] = (_str_conv(nodal_plane.dip_errors.uncertainty,
                                 1)).rjust(5)
        line[40:45] = (_str_conv(nodal_plane.rake_errors.uncertainty,
                                 1)).rjust(5)
    except AttributeError:
        pass
    if hasattr(focal_mechanism, 'misfit'):
        line[45:50] = (_str_conv(focal_mechanism.misfit, 1)).rjust(5)
    if hasattr(focal_mechanism, 'station_distribution_ratio'):
        line[50:55] = (_str_conv(
            focal_mechanism.station_distribution_ratio, 1)).rjust(5)
    if hasattr(focal_mechanism, 'creation_info') and hasattr(
            focal_mechanism.creation_info, 'agency_id'):
        line[66:69] = (str(
            focal_mechanism.creation_info.agency_id)).rjust(3)[0:3]
    if hasattr(focal_mechanism, 'method_id'):
        line[70:77] = (str(focal_mechanism.method_id).split('/')[-1]).rjust(7)
    return ''.join(line)


def _write_hyp_error_line(origin):
    """
    Generate hypocentral error line.

    format:
     GAP=126        0.64       1.3     1.7  2.3 -0.9900E+00 -0.4052E+00  \
     0.2392E+00E
    """
    error_line = list(' ' * 79 + 'E')
    if not hasattr(origin, 'quality'):
        raise NordicParsingError("Origin has no quality associated")
    error_line[1:5] = 'GAP='
    error_line[5:8] = str(int(origin.quality['azimuthal_gap'])).ljust(3)
    error_line[14:20] = (_str_conv(
        origin.quality['standard_error'], 2)).rjust(6)
    # try:
    errors = dict()
    if hasattr(origin, 'origin_uncertainty'):
        # Following will work once Ellipsoid class added
        # if hasattr(origin.origin_uncertainty, 'confidence_ellipsoid'):
        #     cov = Ellipsoid.from_confidence_ellipsoid(
        #         origin.origin_uncertainty['confidence_ellipsoid']).to_cov()
        #     errors['x_err'] = sqrt(cov(0, 0)) / 1000.0
        #     errors['y_err'] = sqrt(cov(1, 1)) / 1000.0
        #     errors['z_err'] = sqrt(cov(2, 2)) / 1000.0
        #     # xy_, xz_, then yz_cov fields
        #     error_line[43:55] = ("%.4e" % (cov(0, 1) / 1.e06)).rjust(12)
        #     error_line[55:67] = ("%.4e" % (cov(0, 2) / 1.e06)).rjust(12)
        #     error_line[67:79] = ("%.4e" % (cov(1, 2) / 1.e06)).rjust(12)
        # else:
        cov = Ellipse.from_origin_uncertainty(origin.origin_uncertainty).\
              to_cov()
        errors['x_err'] = sqrt(cov(0, 0)) / 1000.0
        errors['y_err'] = sqrt(cov(1, 1)) / 1000.0
        errors['z_err'] = origin.depth_errors / 1000.0
        # xy covariance field
        error_line[43:55] = ("%.4e" % (cov(0, 1) / 1.e06)).rjust(12)
    else:
        try:
            errors['x_err'] = origin.longitude_errors.uncertainty / \
                              _km_to_deg_lon(1.0, origin.latitude)
            errors['y_err'] = origin.latitude_errors.uncertainty / \
                _km_to_deg_lat(1.0)
            errors['z_err'] = origin.depth_errors.uncertainty / 1000.0
        except AttributeError:
            return ''.join(error_line)

    error_line[24:30] = (_str_conv(errors['y_err'], 1)).rjust(6)
    error_line[32:38] = (_str_conv(errors['x_err'], 1)).rjust(6)
    error_line[38:43] = (_str_conv(errors['z_err'], 1)).rjust(5)
    return ''.join(error_line)


def nordpick(event, high_accuracy=True):
    """
    Format picks in an :class:`~obspy.core.event.event.Event` to nordic.

    :type event: :class:`~obspy.core.event.event.Event`
    :param event: A single obspy event.
    :type high_accuracy: bool
    :param high_accuracy:
        Whether to output pick seconds at 6.3f (high_accuracy) or
        5.2f (standard).

    :returns: List of String

    .. note::

        Currently finalweight is unsupported, nor is velocity, or
        angle of incidence.  This is because
        :class:`~obspy.core.event.event.Event` stores slowness
        in s/deg and takeoff angle, which would require computation
        from the values stored in seisan.  Multiple weights are also
        not supported.
    """
    # Nordic picks do not have a date associated with them - we need time
    # relative to some origin time.
    try:
        origin = event.preferred_origin() or event.origins[0]
        origin_date = origin.time.date
    except IndexError:
        origin = Origin()
        origin_date = min([p.time for p in event.picks]).date
    pick_strings = []
    if high_accuracy:
        pick_rounding = 3
    else:
        pick_rounding = 2
    for pick in event.picks:
        if not pick.waveform_id:
            msg = ('No waveform id for pick at time %s, skipping' % pick.time)
            warnings.warn(msg)
            continue
        impulsivity = _str_conv(INV_ONSET_MAPPING.get(pick.onset))
        polarity = _str_conv(INV_POLARITY_MAPPING.get(pick.polarity))
        # Extract velocity: Note that horizontal slowness in quakeML is stored
        # as s/deg
        # if pick.horizontal_slowness is not None:
        #     # velocity = 1.0 / pick.horizontal_slowness
        #     velocity = ' '  # Currently this conversion is unsupported.
        # else:
        #     velocity = ' '
        velocity = ' '
        azimuth = _str_conv(pick.backazimuth)
        # Extract the correct arrival info for this pick - assuming only one
        # arrival per pick...
        arrival = [arrival for arrival in origin.arrivals
                   if arrival.pick_id == pick.resource_id]
        if len(arrival) > 0:
            if len(arrival) > 1:
                warnings.warn("Multiple arrivals for pick - only writing one")
            arrival = arrival[0]
            # Extract weight - should be stored as 0-4, or 9 for seisan.
            weight = _str_conv(int(arrival.time_weight or 0))
            # Extract azimuth residual
            if arrival.backazimuth_residual is not None:
                azimuthres = _str_conv(int(arrival.backazimuth_residual))
            else:
                azimuthres = ' '
            if arrival.takeoff_angle is not None:
                ain = _str_conv(int(arrival.takeoff_angle))
            else:
                ain = ' '
            # Extract time residual
            if arrival.time_residual is not None:
                timeres = _str_conv(arrival.time_residual, rounded=2)
            else:
                timeres = ' '
            # Extract distance
            if arrival.distance is not None:
                distance = degrees2kilometers(arrival.distance)
                if distance >= 100.0:
                    distance = str(_int_conv(distance))
                elif 10.0 < distance < 100.0:
                    distance = _str_conv(round(distance, 1), rounded=1)
                elif distance < 10.0:
                    distance = _str_conv(round(distance, 2), rounded=2)
                else:
                    distance = _str_conv(distance, False)
            else:
                distance = ' '
            # Extract CAZ
            if arrival.azimuth is not None:
                caz = _str_conv(int(arrival.azimuth))
            else:
                caz = ' '
        else:
            caz, distance, timeres, azimuthres, azimuth, weight, ain = (
                ' ', ' ', ' ', ' ', ' ', 0, ' ')
        phase_hint = pick.phase_hint or ' '
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
                mag_hint = (amplitude.magnitude_hint or amplitude.type)
                if mag_hint is not None and mag_hint.upper() in ['AML', 'ML']:
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
        eval_mode = INV_EVALUTATION_MAPPING.get(pick.evaluation_mode, None)
        if eval_mode is None:
            warnings.warn("Evaluation mode {0} is not mappable".format(
                pick.evaluation_mode))
            eval_mode = " "
        # Generate a print string and attach it to the list
        channel_code = pick.waveform_id.channel_code or '   '
        pick_hour = pick.time.hour
        if pick.time.date > origin_date:
            # pick hours up to 48 are supported
            days_diff = (pick.time.date - origin_date).days
            if days_diff > 1:
                raise NordicParsingError(
                    "Pick is {0} days from the origin, must be < 48 "
                    "hours".format(days_diff))
            pick_hour += 24
        pick_seconds = pick.time.second + (pick.time.microsecond / 1e6)
        if len(phase_hint) > 4:
            # Weight goes in 9 and phase_hint runs through 11-18
            if polarity != ' ':
                UserWarning("Polarity not written due to phase hint length")
            phase_info = (
                _str_conv(weight).rjust(1) + impulsivity + phase_hint.ljust(8))
        else:
            phase_info = (
                ' ' + impulsivity + phase_hint.ljust(4) +
                _str_conv(weight).rjust(1) + eval_mode +
                polarity.rjust(1) + ' ')
        pick_string_formatter = (
            " {station:5s}{instrument:1s}{component:1s}{phase_info:10s}"
            "{hour:2d}{minute:2d}{seconds:>6s}{coda:5s}{amp:7s}{period:5s}"
            "{azimuth:6s}{velocity:5s}{ain:4s}{azimuthres:3s}{timeres:5s}  "
            "{distance:5s}{caz:4s} ")
        # Note that pick seconds rounding only works because SEISAN does not
        # enforce that seconds stay 0 <= seconds < 60, so rounding something
        # like seconds = 59.997 to 2dp gets to 60.00, which SEISAN is happy
        # with.  It appears that SEISAN is happy with large numbers of seconds
        # see #2348.
        pick_strings.append(pick_string_formatter.format(
            station=pick.waveform_id.station_code,
            instrument=channel_code[0], component=channel_code[-1],
            phase_info=phase_info, hour=pick_hour,
            minute=pick.time.minute,
            seconds=_str_conv(pick_seconds, rounded=pick_rounding),
            coda=_str_conv(coda).rjust(5)[0:5],
            amp=_str_conv(amp, rounded=1).rjust(7)[0:7],
            period=_str_conv(peri, rounded=peri_round).rjust(5)[0:5],
            azimuth=_str_conv(azimuth).rjust(6)[0:6],
            velocity=_str_conv(velocity).rjust(5)[0:5],
            ain=ain.rjust(4)[0:4],
            azimuthres=_str_conv(azimuthres).rjust(3)[0:3],
            timeres=_str_conv(timeres, rounded=2).rjust(5)[0:5],
            distance=distance.rjust(5)[0:5],
            caz=_str_conv(caz).rjust(4)[0:4]))
        # Note that currently finalweight is unsupported, nor is velocity, or
        # angle of incidence.  This is because obspy.event stores slowness in
        # s/deg and takeoff angle, which would require computation from the
        # values stored in seisan.  Multiple weights are also not supported in
        # Obspy.event
    return pick_strings


if __name__ == "__main__":
    import doctest
    doctest.testmod()
