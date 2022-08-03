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

  note::

    Station-magnitude residuals (only for New Nordic files) are handled in
    obspy.core.event.magnitude.StationMagnitude.mag_errors.uncertainty.

  note::

    When you read a Nordic file into Obspy and then write to Nordic format, the
    following information is not retained:
     - amplitude-picks have no distance or azimuth to source
     - some event (sub-)types may change if no equivalent event type exists in
       Obspy

.. versionchanged:: 1.2.3
    * The pick-weight from the Nordic file (0-4, 9) is now read into
      pick.extra.nordic_pick_weight (was arrival.time_weight) while the
      finalweight (0-100 %) is read into arrival.time_weight (or
      backazmiuth_weight, respectively).
    * Empty network codes are now read as None instead of "NA"
    * Magnitudes are no longer automatically sorted by size.

.. versionchanged:: 1.2.0

    The number of stations used to calculate the origin was previously
    incorrectly stored in a comment. From version 1.2.0 this is now stored
    in `origin.quality.used_station_count`

"""
import warnings
from pathlib import Path
import io
import os
import re
from math import sqrt
import datetime
from obspy import UTCDateTime, read
from obspy.geodetics import kilometers2degrees, degrees2kilometers
from obspy.core.event import (
    Event, Origin, Magnitude, StationMagnitude, Catalog, EventDescription,
    CreationInfo, OriginQuality, OriginUncertainty, Pick, WaveformStreamID,
    Arrival, Amplitude, FocalMechanism, MomentTensor, NodalPlane, NodalPlanes,
    QuantityError, Tensor, ResourceIdentifier, Comment)
from obspy.core.event.header import EventType, EventTypeCertainty
from obspy.core.inventory.util import _resolve_seedid, _add_resolve_seedid_doc
from obspy.io.nordic import NordicParsingError
from obspy.io.nordic.utils import (
    _int_conv, _str_conv, _float_conv, _evmagtonor, _nortoevmag,
    _get_line_tags, _km_to_deg_lat, _km_to_deg_lon, _nordic_iasp_phase_ok,
    _get_agency_id, _is_iasp_ampl_phase, EVENT_TYPE_MAPPING_FROM_SEISAN,
    EVENT_TYPE_CERTAINTY_MAPPING_FROM_SEISAN, EVENT_TYPE_MAPPING_TO_SEISAN,
    EVENT_TYPE_AND_CERTAINTY_MAPPING_TO_SEISAN)
from obspy.io.nordic.ellipse import Ellipse


POLARITY_MAPPING = {"": "undecidable", "C": "positive", "D": "negative"}
INV_POLARITY_MAPPING = {item: key for key, item in POLARITY_MAPPING.items()}
ONSET_MAPPING = {'I': 'impulsive', 'E': 'emergent'}
INV_ONSET_MAPPING = {item: key for key, item in ONSET_MAPPING.items()}
EVALUATION_MAPPING = {'A': 'automatic', ' ': 'manual'}
INV_EVALUTATION_MAPPING = {
    item: key for key, item in EVALUATION_MAPPING.items()}
OLD_PHASE_HEADER_LINE = (" STAT SP IPHASW D HRMM SECON CODA AMPLIT PERI AZIMU"
                         " VELO AIN AR TRES W  DIS CAZ7\n")
NEW_PHASE_HEADER_LINE = (" STAT COM NTLO IPHASE   W HHMM SS.SSS   PAR1  PAR2"
                         " AGA OPE  AIN  RES W  DIS CAZ7\n")


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
        header, _ = _readheader(head_lines=tagged_lines['1'])
    return header


def _readheader(head_lines):
    """
    Internal header reader.

    :type head_lines: list
    :param head_lines: List of tuples of (strings, line-number) of the header
        lines.

    :returns: :class:`~obspy.core.event.event.Event`
    """
    # There are two possible types of origin line, one with all info, and a
    # subsequent one for additional magnitudes.
    head_lines.sort(key=lambda tup: tup[1])
    # Construct a rough catalog, then merge events together to cope with
    # multiple origins
    _cat = Catalog()
    # Make a list of the line numbers that are associated with proper origins
    # (i.e., not just extended origin lines)
    origin_line_numbers = []
    for line in head_lines:
        _cat.append(_read_origin(line=line[0]))
        origin_line_numbers.append(line[1])
    new_event = _cat.events.pop(0)
    # No need to check first origin line for whether it is an extended line -
    # it has to be the main defining line.
    check_origin_line_numbers = origin_line_numbers.copy()
    check_origin_line_numbers.pop(0)

    # origin_line_numbers is a list of line numbers where a new origin
    # definitino starts in the S-file. Hence origin lines that contain extended
    # information (more magnitudes, higher accuracy) for a previous origin line
    # are not part of the list.
    # for event, orig_line_number in zip(_cat, origin_line_numbers[1:]):
    for event, orig_line_number in zip(_cat, check_origin_line_numbers):
        matched = False
        origin_times = [origin.time for origin in new_event.origins]
        origin = event.origins[0]
        otime = origin.time
        if (otime in origin_times and not origin.latitude and
                not origin.longitude and not origin.depth):
            origin_index = origin_times.index(origin.time)
            agency = new_event.origins[origin_index].creation_info.agency_id
            if event.creation_info.agency_id == agency:
                event_desc = new_event.event_descriptions[origin_index].text
                if event.event_descriptions[0].text == event_desc:
                    matched = True
                    origin_line_numbers.remove(orig_line_number)
        # Nordic format actually only requires year, month, day, and agency
        # to appear in extended hypocenter line.
        if not matched:
            if otime.hour == 0 and otime.minute == 0 and otime.second == 0:
                agency_ids = [origin.creation_info.agency_id
                              for origin in new_event.origins]
                agency_id = event.origins[0].creation_info.agency_id
                if agency_id in agency_ids:
                    origin_index = agency_ids.index(agency_id)
                    matched = True
                    origin_line_numbers.remove(orig_line_number)

        new_event.magnitudes.extend(event.magnitudes)
        if not matched:
            new_event.origins.append(event.origins[0])
            new_event.event_descriptions.append(event.event_descriptions[0])
        else:
            # If there's an extended origin line, fix the origin_ids of any
            # associated magnitudes
            for magnitude in event.magnitudes:
                if magnitude.origin_id == origin.resource_id:
                    magnitude.origin_id = new_event.origins[
                        origin_index].resource_id
    # Remove duplicated EventDescription-lines
    new_event.event_descriptions = [
        x for j, x in enumerate(new_event.event_descriptions)
        if j == new_event.event_descriptions.index(x)]
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
    return new_event, origin_line_numbers


def _read_origin(line):
    """
    Read one origin (type 1) line.

    :param str line: Origin format (type 1) line
    :return: `~obspy.core.event.event.Event`
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
    new_event.event_type = EventType(EVENT_TYPE_MAPPING_FROM_SEISAN.get(
        line[22]))
    new_event.event_type_certainty = EventTypeCertainty(
        EVENT_TYPE_CERTAINTY_MAPPING_FROM_SEISAN.get(line[22]))
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
                mag=_float_conv(line[index - 4:index]),
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
    :type event: :class:`~obspy.core.event.event.Event`
    :param event: Event to associate spectral info with

    :returns:
        list of dictionaries of spectral information, units as in
        seisan manual, expect for logs which have been converted to floats.
    """
    if '3' not in tagged_lines.keys():
        return {}
    if event is None:
        event, _ = _readheader(head_lines=tagged_lines['1'])
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


@_add_resolve_seedid_doc
def read_nordic(select_file, return_wavnames=False, encoding='latin-1',
                nordic_format='UKN', **kwargs):
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
    :type nordic_format: str
    :param nordic_format:
        'UKN', 'OLD', or 'NEW' (unknown, old, new). For 'UKN', the function
        will find out on its own

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
                return_wavnames=return_wavnames, nordic_format=nordic_format,
                **kwargs)
            event_str = []
    f.close()
    if len(event_str) > 0:
        # May occur if the last line of the file is not blank as it should be
        catalog, wav_names = _extract_event(
            event_str=event_str, catalog=catalog, wav_names=wav_names,
            return_wavnames=return_wavnames, nordic_format=nordic_format,
            **kwargs)
    if return_wavnames:
        return catalog, wav_names
    for event in catalog:
        event.scope_resource_ids()
    return catalog


def _extract_event(event_str, catalog, wav_names, return_wavnames=False,
                   nordic_format='UKN', **kwargs):
    """
    Helper to extract event info from a list of line strings.

    :param event_str: List of lines from sfile
    :type event_str: list[str]
    :param catalog: Catalog to append the event to
    :type catalog: `obspy.core.event.Catalog`
    :param wav_names: List of waveform names
    :type wav_names: list
    :param return_wavnames: Whether to extract the waveform name or not.
    :type return_wavnames: bool
    :type nordic_format: str
    :param nordic_format:
        'UKN', 'OLD', or 'NEW' (unknown, old, new). For 'UKN', the function
        will find out on its own

    :return: Adds event to catalog and returns. Works in place on catalog.
    """
    tmp_sfile = io.StringIO()
    for event_line in event_str:
        tmp_sfile.write(event_line)
    tagged_lines = _get_line_tags(f=tmp_sfile)
    new_event, origin_line_numbers = _readheader(head_lines=tagged_lines['1'])
    new_event = _read_uncertainty(tagged_lines, new_event)
    new_event = _read_highaccuracy(
        tagged_lines, new_event, origin_line_numbers)
    new_event = _read_focal_mechanisms(tagged_lines, new_event)
    new_event = _read_moment_tensors(tagged_lines, new_event)
    new_event = _read_comments(tagged_lines, new_event)
    if return_wavnames:
        wav_names.append(_readwavename(tagged_lines=tagged_lines['6']))
    new_event = _read_picks(tagged_lines=tagged_lines, new_event=new_event,
                            nordic_format=nordic_format, **kwargs)
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
        errors = {'x_err': _float_conv(line[32:38]),
                  'y_err': _float_conv(line[24:30]),
                  'z_err': _float_conv(line[38:43]),
                  'xy_cov': _float_conv(line[43:55]),
                  'xz_cov': _float_conv(line[55:67]),
                  'yz_cov': _float_conv(line[67:79])}
    except ValueError:
        pass
    orig = event.origins[0]
    # If any value is Zero or None then no Ellipse can be constructed
    if not [x for x in (errors['x_err'], errors['y_err'], errors['xy_cov'])
            if not x]:
        e = Ellipse.from_uncerts(errors['x_err'],
                                 errors['y_err'],
                                 errors['xy_cov'])
        if e:
            orig.origin_uncertainty = OriginUncertainty(
                max_horizontal_uncertainty=e.a * 1000.,
                min_horizontal_uncertainty=e.b * 1000.,
                azimuth_max_horizontal_uncertainty=e.theta,
                preferred_description="uncertainty ellipse")
    # But lat / lon / depth-errors may still be filled
    if errors['y_err'] is not None:
        orig.latitude_errors = QuantityError(_km_to_deg_lat(errors['y_err']))
    if errors['x_err'] is not None and orig.latitude:
        orig.longitude_errors = QuantityError(_km_to_deg_lon(errors['x_err'],
                                                             orig.latitude))
    if errors['z_err'] is not None:
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


def _read_highaccuracy(tagged_lines, event, origin_line_numbers):
    """
    Read high accuracy origin line.

    :param tagged_lines: Lines keyed by line type
    :type tagged_lines: dict
    :param origin_line_numbers:
        List of the line numbers of the proper origins (i.e., exluding ex-
        tended origin lines)
    :type origin_line_numbers: list of int
    :returns: updated event
    :rtype: :class:`~obspy.core.event.event.Event`

    note:
    The S-file can include a type=H line for each hypocenter (new from
    version 12). Prior to version 12.0 only one type=H line was allowed in
    SEISAN. The H line gives the same solution as the type=1 line, but with
    higher accuracy. In order to know which H-line belongs to which 1-line,
    the location program indicator and the agency must match. If only one
    H-line and no agency, it is assumed it belongs to the main hypocenter.
    This ensures backwards compatibility.
    """
    if 'H' not in tagged_lines.keys():
        return event
    agency_ids = [origin.creation_info.agency_id for origin in event.origins]

    for line, ha_line_number in tagged_lines['H']:
        agency_id = _str_conv(line[60:63])
        # Prior to version 12 there could only be one high accuracy line; if
        # there is no agency id or no origin with matching agency id, then
        # refer to the first origin.
        if (len(tagged_lines['H']) == 1 and
                (agency_id is None or agency_id not in agency_ids)):
            origin_index = 0
        else:
            # Select the proper origin header line that is written just above
            # the high accuracy line and select the relevant origin index.
            origin_index = [
                jl for jl, o_line_no in enumerate(origin_line_numbers)
                if o_line_no < ha_line_number][-1]
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
            if abs(event.origins[origin_index].time - ev_time) < 0.1:
                event.origins[origin_index].time = ev_time
            else:
                print('High accuracy time differs from normal time by >0.1s')
        except ValueError:
            pass
        values = {'latitude': None,
                  'longitude': None,
                  'depth': None,
                  'rms': None}
        try:
            values = {'latitude': _float_conv(line[23:32]),
                      'longitude': _float_conv(line[33:43]),
                      'depth': _float_conv(line[44:52]),
                      'rms': _float_conv(line[53:59])}
        except ValueError:
            pass

        if values['latitude'] is not None:
            event.origins[origin_index].latitude = values['latitude']
        if values['longitude'] is not None:
            event.origins[origin_index].longitude = values['longitude']
        if values['depth'] is not None:
            event.origins[origin_index].depth = values['depth'] * 1000.
        if values['rms'] is not None:
            if event.origins[origin_index].quality is not None:
                event.origins[origin_index].quality.standard_error = (
                    values['rms'])
            else:
                event.origins[origin_index].quality = OriginQuality(
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
            mag=_float_conv(mt_line_1[55:59]),
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


def _read_comments(tagged_lines, event):
    """
    Read comment lines from s-file

    :param tagged_lines: Lines keyed by line type
    :type tagged_lines: dict
    :returns: updated event
    :rtype: :class:`~obspy.core.event.event.Event`
    """
    if '3' not in tagged_lines.keys():
        return event
    # Get comment lines
    com_lines = tagged_lines['3']
    # can contain SPEC which is read in as spectral information -
    # should such lines not be read as comments?
    for seisan_comment, line in com_lines:
        # Remove end-of-line characters and empty text
        save_comment = re.sub('3\\n$', '', seisan_comment).strip()
        event.comments.append(Comment(text=save_comment))

    # Add waveform-file names as comment to event
    wav_lines = tagged_lines['6']
    for wav_line, line in wav_lines:
        save_comment = 'Waveform-filename: ' + re.sub(
            '6\\n', '', wav_line).strip()
        event.comments.append(Comment(text=save_comment))
    return event


def _read_picks(tagged_lines, new_event, nordic_format='UKN', **kwargs):
    """
    Internal pick reader. Use read_nordic instead.

    :type tagged_lines: dict
    :param tagged_lines: Lines keyed by line type
    :type new_event: :class:`~obspy.core.event.event.Event`
    :param new_event: event to associate picks with.
    :type nordic_format: str
    :param nordic_format:
        'UKN', 'OLD', or 'NEW' (unknown, old, new). For 'UKN', the function
        will find out on its own

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

    if nordic_format == 'UKN':
        nordic_format, phase_ok = check_nordic_format_version(pickline)

    if nordic_format == 'NEW':
        new_event = _read_picks_nordic_new(pickline, new_event, header, evtime,
                                           **kwargs)
    elif nordic_format == 'OLD':
        new_event = _read_picks_nordic_old(pickline, new_event, header, evtime,
                                           **kwargs)
    elif nordic_format == 'UKN':
        warnings.warn('Cannot check whether Nordic format is Old or New, is '
                      'this really a Nordic file?')

    return new_event


def check_nordic_format_version(pickline):
    """
    Check whether the type 4 line is for the old or the new Nordic format.

    :type pickline: dict
    :param pickline: Lines keyed by line type

    :returns: str, 'UKN', 'OLD', or 'NEW' (unknown, old, new)
    :returns: bool, whether line contains an allowed phase or not.
    """

    # The test should only need to be done on one single pick-line per event or
    # catalog; then it should be clear what is the format for the rest of the
    # file.

    # Here is the basic logic for the Format check; based on subroutine
    # `check_if_phase_line_format` in Seisan:
    #   1. Check weather New or Old format based on the seconds-string of the
    #      old format.
    #   2. Assure that it's really New Nordic format
    #     2.a  Check the full format of the pick-time
    #     2.b  Check phase-descriptor
    #   3. Assure that it's really Old Nordic format
    #     3.a  Check the full format of the pick-time
    #     3.b  Check phase-descriptor

    # Let's  assume it's a New Nordic format for now.
    is_phase = False
    nordic_format = 'UKN'

    for line in pickline:
        # if whole line is blank, it cannot be a phase line
        if line.isspace():
            return nordic_format, is_phase

        old_format_secs = line[23:28].strip(' ')
        # Check whether the old-formatted seconds are a float rather than int.
        # If they are int, then it is probably new nordic format.
        try:
            comp_str = old_format_secs.replace(' ', '').replace('A', '')
            if str(int(comp_str)) == old_format_secs:
                nordic_format = 'NEW'
        except ValueError:
            nordic_format = 'OLD'
            pass

        if nordic_format == 'NEW' or nordic_format == 'UKN':
            hr = int(line[26:28].strip() or 0)
            min = int(line[28:30].strip() or 0)
            sec = float(line[30:37].strip() or 0)
            if (line[9:10] == ' ' and
               line[14:15] == ' ' and
               line[26:27] in ' 012' and
               hr >= 0 and hr <= 26 and
               min >= 0 and min <= 60 and
               sec >= 0.0 and sec <= 200 and
               not line[28:37].isspace()):
                if (_nordic_iasp_phase_ok(line[16:25]) or
                        line[16:19] == 'END' or
                        line[16:19] == 'BAZ' or
                        line[16:18] == 'IA' or
                        line[16:18] == 'IV' or
                        line[16:18] == 'AM' or
                        line[16:18] == 'AP' or
                        (line[16:17] == ' ' and
                         line[15:16] == 'I' or line[15:16] == 'E')):
                    is_phase = True
                    nordic_format = 'NEW'
                    return nordic_format, is_phase
        else:
            hr = int(line[18:20].strip() or 0)
            min = int(line[20:22].strip() or 0)
            sec = float(line[22:28].strip() or 0)
            if (line[18:19] in ' 012' and
               hr >= 0 and hr <= 26 and
               min >= 0 and min <= 60 and
               sec >= 0.0 and sec <= 200):
                if (_nordic_iasp_phase_ok(line[10:18]) or
                        line[10:13] == 'END' or
                        line[10:13] == 'BAZ' or
                        line[10:12] == 'IA' or
                        line[10:12] == 'IV' or
                        line[10:12] == 'AM' or
                        line[10:12] == 'AP' or
                        (line[10:11] == ' ' and
                         line[9:10] == 'I' or line[9:10] == 'E')):
                    is_phase = True
                    nordic_format = 'OLD'
                    return nordic_format, is_phase
    # If neither New nor Old format, return "UKN"
    return nordic_format, is_phase


def _read_picks_nordic_old(pickline, new_event, header, evtime, **kwargs):
    """
    Reads the type 4 line of the old Nordic format.
    """
    for line in pickline:
        ain, snr = (None, None)
        if line[18:28].strip() == '':  # If line is empty miss it
            continue
        if len(line) < 80:
            line = line.ljust(80)  # Pick-lines without a tag may be short.
        weight = line[14]
        if weight not in ' 012349_':  # Long phase name
            weight = line[8]
            phase = line[10:17].strip()
            polarity = ''
        elif weight == '_':
            phase = line[10:17]
            weight = None
            polarity = ''
        else:
            phase = line[10:14].strip()
            polarity = line[16]
        if weight == ' ':
            weight = None
        polarity = POLARITY_MAPPING.get(polarity, None)  # Empty could be None
        # or undecidable.
        # It is valid nordic for the origin to be hour 23 and picks to be hour
        # 00 or 24: this signifies a pick over a day boundary.
        pick_hour = int(line[18:20].strip() or 0)
        pick_minute = int(line[20:22].strip() or 0)
        pick_seconds = float(line[22:29].strip() or 0.0)  # 29 should be blank,
        # but sometimes SEISAN appears to overflow here, see #2348
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
        finalweight = _int_conv(line[68:70])
        # Create a new obspy.event.Pick class for this pick
        widargs = _resolve_seedid(
            station=line[1:6].strip(), component=line[6:8].strip(), time=time,
            **kwargs)
        _waveform_id = WaveformStreamID(*widargs)
        pick = Pick(waveform_id=_waveform_id, phase_hint=phase,
                    polarity=polarity, time=time)
        try:
            pick.onset = ONSET_MAPPING[line[9]]
        except KeyError:
            pass
        pick.evaluation_mode = EVALUATION_MAPPING.get(line[15], "manual")
        # Pick-weight from Seisan is not covered by Obspy/Quakeml standard
        if weight is not None:
            pick.extra = {
                'nordic_pick_weight': {
                    'value': weight,
                    'namespace':
                        'https://seis.geus.net/software/seisan/node239.html'}}
        # Note BAZ and slowness are not always filled.
        if _float_conv(line[46:51]) is not None:
            pick.backazimuth = _float_conv(line[46:51])
        app_velocity = _float_conv(line[51:56])
        if (app_velocity is not None and app_velocity != 999.0 and
                app_velocity != 0):
            pick.horizontal_slowness = 1 / kilometers2degrees(app_velocity)
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
            if finalweight is not None:
                arrival.time_weight = finalweight / 10
            new_event.origins[0].arrivals.append(arrival)
        new_event.picks.append(pick)

    return new_event


def _read_picks_nordic_new(pickline, new_event, header, evtime, **kwargs):
    """
    Reads the type 4 line of the old Nordic format.
    """
    for line in pickline:
        ain, snr = (None, None)
        if line[14:37].strip() == '':  # If line is empty miss it
            continue
        if len(line) < 80:
            line = line.ljust(80)  # Pick-lines without a tag may be short.
        weight = line[24]
        if weight == ' ':
            weight = None
        phase = line[16:24].strip()
        if _nordic_iasp_phase_ok(phase) and phase not in ['END', 'BAZ']:
            polarity = line[43]
        else:
            polarity = 'undecidable'
        polarity = POLARITY_MAPPING.get(polarity, None)  # Empty could be None
        # or undecidable.
        # It is valid nordic for the origin to be hour 23 and picks to be hour
        # 00 or 24: this signifies a pick over a day boundary. Seisan also
        # allows empty hour/min/sec, which is equal to 00.
        pick_hour = int(line[26:28].strip() or 0)
        pick_minute = int(line[28:30].strip() or 0)
        pick_seconds = float(line[31:37].strip() or 0)
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
        if header[60:63] == 'AIN':
            ain = _float_conv(line[58:63])
        elif header[60:63] == 'SNR':
            snr = _float_conv(line[58:63])
        else:
            warnings.warn('%s is not currently supported' % header[60:63])
        finalweight = _int_conv(line[68:70])
        # Create a new obspy.event.Pick class for this pick
        sta, cha = line[1:6].strip(), line[6:9].strip()
        net, loc = line[10:12].strip(), line[12:14].strip()
        if net == '' and loc == '':
            widargs = _resolve_seedid(station=sta, component=cha, time=time,
                                      **kwargs)
        else:
            widargs = net, sta, loc, cha
        _waveform_id = WaveformStreamID(*widargs)
        pick = Pick(waveform_id=_waveform_id, phase_hint=phase,
                    polarity=polarity, time=time)
        # agency and operator / author / analyst
        if line[51:54] != '   ' or line[55:58] != '   ':
            pick.creation_info = CreationInfo(agency_id=line[51:54].strip(),
                                              author=line[55:58].strip())
        try:
            pick.onset = ONSET_MAPPING[line[15]]
        except KeyError:
            pass
        pick.evaluation_mode = EVALUATION_MAPPING.get(line[25], "manual")
        # Pick-weight from Seisan is not covered by Obspy/Quakeml standard
        if weight is not None:
            pick.extra = {
                'nordic_pick_weight': {
                    'value': weight,
                    'namespace':
                        'https://seis.geus.net/software/seisan/node239.html'}}
        # Note that BAZ and apparent velocity are not always filled
        found_baz_associated_pick = False
        if 'BAZ' in phase and _float_conv(line[37:44]) is not None:
            # Seisan phase name can be e.g. "BAZ-P"
            baz_phase_type = phase.strip('BAZ-')
            # Check if there is matching pick for the BAZ. THIS MEANS THAT
            # THE BAZ-line in Nordic file has to follow the actual time pick!
            for existing_pick in new_event.picks:
                if (existing_pick.waveform_id == pick.waveform_id and
                        existing_pick.phase_hint == baz_phase_type and
                        existing_pick.time == pick.time):
                    pick = existing_pick
                    found_baz_associated_pick = True
                    break
            # BAZ-phase name doesn't have to specify the associated phase
            # name - as a secondary resort, compare pick times only.
            if not found_baz_associated_pick:
                for existing_pick in new_event.picks:
                    if (existing_pick.waveform_id == pick.waveform_id and
                            existing_pick.time == pick.time and
                            not _is_iasp_ampl_phase(existing_pick.phase_hint)):
                        pick = existing_pick
                        found_baz_associated_pick = True
                        break
            pick.backazimuth = _float_conv(line[37:44])
            app_velocity = _float_conv(line[44:50])
            if (app_velocity is not None and app_velocity != 0 and
                    app_velocity != 999.0):
                pick.horizontal_slowness = 1 / kilometers2degrees(app_velocity)
        # Create new obspy.event.Amplitude class which references above Pick
        # only if there is an amplitude picked.
        is_coda_ref_pick = False
        if (_is_iasp_ampl_phase(phase) and _float_conv(line[37:44]) is not None
                and not found_baz_associated_pick):
            _amplitude = Amplitude(generic_amplitude=_float_conv(line[37:44]),
                                   period=_float_conv(line[44:50]),
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
            # Magnitude for single trace / station computed from amplitude
            # if line[63:68].strip() != '':
            mag_residual = _float_conv(line[63:68])
            if mag_residual is None:
                mag_residual = 0
            # assoc_mag = new_event.magnitudes[0]
            assoc_mag = None
            for mag in new_event.magnitudes:
                try:
                    if (mag.magnitude_type == _amplitude.magnitude_hint
                            and mag.creation_info.agency_id
                            == pick.creation_info.agency_id):
                        assoc_mag = mag
                except AttributeError:
                    pass
            if assoc_mag is not None:
                _trace_mag = StationMagnitude(
                    origin_id=assoc_mag.origin_id,
                    mag=assoc_mag.mag - mag_residual,
                    mag_errors=mag_residual,
                    station_magnitude_type=assoc_mag.magnitude_type,
                    amplitude_id=_amplitude.resource_id,
                    method_id=assoc_mag.method_id,
                    waveform_id=pick.waveform_id,
                    creation_info=pick.creation_info)
                new_event.station_magnitudes.append(_trace_mag)
        elif phase == 'END' and _int_conv(line[37:44]) is not None:
            # Coda duration amplitude - should be generally in reference to the
            # previous pick, then it can be added simply as an amplitude
            # referencing the pick; or it could become its own pick.
            if (len(new_event.picks) > 0
                    and new_event.picks[-1].waveform_id.station_code ==
                    pick.waveform_id.station_code):
                pick = new_event.picks[-1]
                is_coda_ref_pick = True
            # Create an amplitude instance for coda duration also
            _amplitude = Amplitude(generic_amplitude=_int_conv(line[37:44]),
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
        if not _is_iasp_ampl_phase(phase) and not phase == 'END':
            # If there is a pick associated with a BAZ, then there's also an
            # arrival for that pick already
            if found_baz_associated_pick:
                for existing_arrival in new_event.origins[0].arrivals:
                    if existing_arrival.pick_id == pick.resource_id:
                        arrival = existing_arrival
                        break
            else:
                arrival = Arrival(phase=pick.phase_hint,
                                  pick_id=pick.resource_id)
            if _float_conv(line[63:68]) is not None:
                if 'BAZ' in phase:
                    arrival.backazimuth_residual = _float_conv(line[63:68])
                else:
                    arrival.time_residual = _float_conv(line[63:68])
            # Add other information and append arrival itself only if it's new.
            if not found_baz_associated_pick:
                if _float_conv(line[70:75]) is not None:
                    arrival.distance = kilometers2degrees(_float_conv(
                        line[70:75]))
                if _int_conv(line[76:79]) is not None:
                    arrival.azimuth = _int_conv(line[76:79])
                if ain is not None:
                    arrival.takeoff_angle = ain
                if finalweight is not None:
                    arrival.time_weight = finalweight / 10
                new_event.origins[0].arrivals.append(arrival)
        # Add the pick, but do not add it as new if the pick was only updated
        # with new information (BAZ, slowness etc.)
        if not (found_baz_associated_pick or is_coda_ref_pick):
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


def blanksfile(wavefile, evtype, userid, overwrite=False, evtime=None,
               nordic_format='OLD'):
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
    :type nordic_format: str
    :param nordic_format:
        Version of Nordic format to be used for output, either OLD or NEW.

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
    if Path(sfile).is_file() and not overwrite:
        warnings.warn('Desired sfile: ' + sfile + ' exists, will not ' +
                      'overwrite')
        for i in range(1, 10):
            sfile = str(evtime.day).zfill(2) + '-' +\
                str(evtime.hour).zfill(2) +\
                str(evtime.minute).zfill(2) + '-' +\
                str(evtime.second + i).zfill(2) + evtype + '.S' +\
                str(evtime.year) + str(evtime.month).zfill(2)
            if not Path(sfile).is_file():
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
        write_wavfile = str(Path(wavefile).parent)
        f.write(' ' + write_wavfile + '6'.rjust(79 - len(write_wavfile)) +
                '\n')
        # Write final line of s-file
        if nordic_format == 'OLD':
            f.write(OLD_PHASE_HEADER_LINE)
        elif nordic_format == 'NEW':
            f.write(NEW_PHASE_HEADER_LINE)
    return sfile


def write_select(catalog, filename, userid='OBSP', evtype='L',
                 wavefiles=None, high_accuracy=True, nordic_format='OLD'):
    """
    Function to write a catalog to a select file in nordic format.

    :type catalog: :class:`~obspy.core.event.catalog.Catalog`
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
    :type nordic_format: str
    :param nordic_format:
        Version of Nordic format to be used for output, either OLD or NEW.
    """
    if nordic_format not in ['OLD', 'NEW']:
        raise ValueError('Nordic format can be ''OLD'' or ''NEW'', not '
                         + nordic_format)
    if not wavefiles:
        wavefiles = ['' for _i in range(len(catalog))]
    with open(filename, 'w') as fout:
        for event, wavfile in zip(catalog, wavefiles):
            select = io.StringIO()
            _write_nordic(
                event=event, filename=None, userid=userid, evtype=evtype,
                wavefiles=wavfile, nordic_format=nordic_format,
                string_io=select, high_accuracy=high_accuracy)
            select.seek(0)
            for line in select:
                fout.write(line)
            fout.write('\n')


def _write_nordic(event, filename, userid='OBSP', evtype='L', outdir='.',
                  wavefiles=None, explosion=False, nordic_format='OLD',
                  overwrite=True, string_io=None, high_accuracy=True):
    """
    Write an :class:`~obspy.core.event.event.Event` to a nordic formatted
    s-file.

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
    :type nordic_format: str
    :param nordic_format:
        nordic_format of Nordic format to be used for output, either OLD or
        NEW.
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
    if len(userid) > 4:
        raise NordicParsingError('%s User ID must be at most 4 characters long'
                                 % userid)
    userid = userid.ljust(4)
    # Check that outdir exists
    if not Path(outdir).is_dir():
        raise NordicParsingError('Out path does not exist, I will not '
                                 'create this: ' + outdir)
    # Check that evtype is one of L,R,D
    if evtype not in ['L', 'R', 'D']:
        raise NordicParsingError('Event type must be either L, R or D')
    if explosion:
        evtype += 'E'
    elif event.event_type is not None:
        try:
            evtype += EVENT_TYPE_AND_CERTAINTY_MAPPING_TO_SEISAN.get(
                event.event_type_certainty + ' ' + event.event_type)
        except TypeError:
            try:
                evtype += EVENT_TYPE_MAPPING_TO_SEISAN.get(event.event_type)
            except TypeError:
                pass
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
            if not (Path(outdir) / sfilename).is_file():
                sfile_path = Path(outdir) / sfilename
                break
            elif overwrite:
                sfile_path = Path(outdir) / sfilename
                break
        else:
            raise NordicParsingError(str(Path(outdir) / sfilename) +
                                     ' already exists, will not overwrite')
    else:
        sfile_path = Path(outdir) / filename
        sfilename = filename
    if not string_io:
        sfile = open(sfile_path, 'w')
    else:
        sfile = string_io
    # Write header line(s)
    sfile.write(_write_header_line(
        event, origin, evtype, is_preferred_origin=True))
    if high_accuracy:
        sfile.write(_write_high_accuracy_origin(origin))
    # Write hyp error line
    try:
        sfile.write(_write_hyp_error_line(origin) + '\n')
    except (NordicParsingError, TypeError):
        pass
    # Write origin lines for additional origins
    for add_origin in event.origins:
        if not add_origin == origin:
            sfile.write(_write_header_line(event, add_origin, evtype,
                                           is_preferred_origin=False))
            if high_accuracy:
                sfile.write(_write_high_accuracy_origin(add_origin))
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
        # Do not write names that do not actually link to a waveform file
        if wavefile == '' or wavefile == 'None' or wavefile is None:
            continue
        sfile.write(' ' + os.path.basename(wavefile) +
                    '6'.rjust(79 - len(os.path.basename(wavefile))) + '\n')
    for comment in event.comments:
        nordic_comment = _write_comment(comment)
        if nordic_comment is None:
            continue
        sfile.write(nordic_comment + '\n')
    # Write final line of s-file
    if nordic_format == 'OLD':
        sfile.write(OLD_PHASE_HEADER_LINE)
    elif nordic_format == 'NEW':
        sfile.write(NEW_PHASE_HEADER_LINE)
    # Now call the populate sfile function
    if len(event.picks) > 0:
        newpicks = '\n'.join(nordpick(event, high_accuracy=high_accuracy,
                                      nordic_format=nordic_format))
        sfile.write(newpicks + '\n')
        sfile.write('\n'.rjust(81))
    if not string_io:
        sfile.close()
        return str(sfilename)
    else:
        return


def _write_header_line(event, origin, evtype, is_preferred_origin=True):
    """
    Write one Seisan header line for origin. Needs to treat the preferred
    origin a bit differently than additional origins.
    """
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
    conv_mags = []
    if is_preferred_origin:
        agency = _get_agency_id(event)
        other_orig_ids = [orig.resource_id for orig in event.origins
                          if orig != origin]
    else:  # Addition (not preferred) origin - don't need to add magnitudes
        agency = _get_agency_id(origin)
        other_orig_ids = [mag.origin_id for mag in event.magnitudes]
    # Get up to six magnitudes
    mt_ids = []
    if hasattr(event, 'focal_mechanisms'):
        for fm in event.focal_mechanisms:
            if hasattr(fm, 'moment_tensor') and hasattr(
                    fm.moment_tensor, 'moment_magnitude_id'):
                mt_ids.append(fm.moment_tensor.moment_magnitude_id)
    n_mags = 0
    mag_ind = 0
    while n_mags < 6:
        mag_info = {}
        try:
            if event.magnitudes[mag_ind].resource_id in mt_ids:
                mag_ind += 1
                continue
                # raise IndexError("Repeated magnitude")
                # This magnitude will get put in with the moment tensor
            if (event.magnitudes[mag_ind].origin_id == origin.resource_id or
                    event.magnitudes[mag_ind].origin_id not in other_orig_ids):
                mag_info['mag'] = '{0:.1f}'.format(
                    event.magnitudes[mag_ind].mag) or ''
                mag_info['type'] = _evmagtonor(event.magnitudes[mag_ind].
                                               magnitude_type) or ''
                mag_info['agency'] = _get_agency_id(event.magnitudes[mag_ind])
                conv_mags.append(mag_info)
            else:
                mag_ind += 1
                continue
        except IndexError:
            mag_info.update({'mag': '', 'type': '', 'agency': ''})
            conv_mags.append(mag_info)
        n_mags += 1
        mag_ind += 1
    # Better not to sort magnitudes by their value, instead stick with
    # their order:
    # conv_mags.sort(key=lambda d: d['mag'], reverse=True)
    if conv_mags[3]['mag'] == '':
        conv_mags = conv_mags[0:3]
    # Cope with differences in event uncertainty naming
    if origin.quality and origin.quality['standard_error']:
        timerms = '{0:.1f}'.format(origin.quality['standard_error'])
    else:
        timerms = '0.0'
    # Work out how many stations were used
    stations = []
    # First try to count number of arrival-stations
    if len(origin.arrivals) > 0:
        try:
            stations = [
                arrival.pick_id.get_referred_object().waveform_id.station_code
                for arrival in origin.arrivals]
            ksta = str(len(set(stations)))
        except AttributeError:
            pass
    # If not successfull, count the number of pick-stations
    if not stations:
        if len(event.picks) > 0:
            stations = [pick.waveform_id.station_code for pick in event.picks]
            ksta = str(len(set(stations)))
        else:
            ksta = ''
    # Nordic format supports only 3-letter number of stations
    if len(ksta) > 3 and len(set(stations)) > 999:
        ksta = '999'
    evtime = origin.time
    if not evtime:
        return

    lines = []
    lines.append(
        " {0} {1}{2} {3}{4} {5}.{6} {7}{8}{9}{10}  {11}{12}{13}{14}{15}{16}"
        "{17}{18}{19}{20}{21}{22}1\n".format(
            evtime.year, str(evtime.month).rjust(2), str(evtime.day).rjust(2),
            str(evtime.hour).rjust(2), str(evtime.minute).rjust(2),
            str(evtime.second).rjust(2), str(evtime.microsecond).ljust(1)[0:1],
            evtype.ljust(2), lat.rjust(7), lon.rjust(8), depth.rjust(5),
            agency, ksta.rjust(3), timerms.rjust(4),
            conv_mags[0]['mag'].rjust(4), conv_mags[0]['type'].rjust(1),
            conv_mags[0]['agency'][0:3].rjust(3),
            conv_mags[1]['mag'].rjust(4), conv_mags[1]['type'].rjust(1),
            conv_mags[1]['agency'][0:3].rjust(3),
            conv_mags[2]['mag'].rjust(4), conv_mags[2]['type'].rjust(1),
            conv_mags[2]['agency'][0:3].rjust(3)))
    if len(conv_mags) > 3:
        lines.append(
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
    return ''.join([''.join(line) for line in lines])


def _write_high_accuracy_origin(origin):
    """
    Write high accuracy hypocenter line
    E.g.:
    1996  6 3 2006 35.511  46.78711  153.72245   33.011  1.923
    """
    # Write the header info.
    if origin.latitude is not None:
        lat = '{0:.5f}'.format(origin.latitude)
    else:
        lat = ''
    if origin.longitude is not None:
        lon = '{0:.5f}'.format(origin.longitude)
    else:
        lon = ''
    if origin.depth is not None:
        depth = '{0:.3f}'.format(origin.depth / 1000.0)
    else:
        depth = ''
    evtime = origin.time
    if not evtime:
        return

    # Cope with differences in event uncertainty naming
    if origin.quality and origin.quality['standard_error']:
        timerms = '{0:.3f}'.format(origin.quality['standard_error'])
    else:
        timerms = '0.000'

    ha_origin_line = (
        " {0} {1}{2} {3}{4} {5}.{6} {7} {8} {9} {10} {11}H\n".format(
            evtime.year, str(evtime.month).rjust(2), str(evtime.day).rjust(2),
            str(evtime.hour).rjust(2), str(evtime.minute).rjust(2),
            str(evtime.second).rjust(2), str(evtime.microsecond).ljust(3)[0:3],
            lat.rjust(9), lon.rjust(10), depth.rjust(8), timerms.rjust(6),
            " " * 19))

    return ha_origin_line


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
    lines[0][45:48] = _get_agency_id(origin)
    lines[0][55:59] = _str_conv(magnitude.mag, 1).rjust(4)
    lines[0][59] = _evmagtonor(magnitude.magnitude_type)
    lines[0][60:63] = _get_agency_id(magnitude)
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
    lines[1][45:48] = _get_agency_id(magnitude)
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
    line[66:69] = _get_agency_id(focal_mechanism)
    if hasattr(focal_mechanism, 'method_id'):
        line[70:77] = (str(focal_mechanism.method_id).split('/')[-1]
                       ).rjust(7)[0:7]
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
    if origin.quality['azimuthal_gap']:
        error_line[5:8] = str(int(origin.quality['azimuthal_gap'])).ljust(3)
    error_line[11:14] = _get_agency_id(origin)
    if origin.time_errors.uncertainty:
        error_line[14:20] = _str_conv(
            origin.time_errors.uncertainty, 2).rjust(6)
    # try:
    errors = dict()
    add_simplified_uncertainty = False
    add_uncertainty = False
    if hasattr(origin, 'origin_uncertainty'):
        orig_unc = origin.origin_uncertainty
        if (hasattr(orig_unc, 'min_horizontal_uncertainty')
                and hasattr(orig_unc, 'max_horizontal_uncertainty')):
            # Even though uncertainty should not be Zero, such files exist.
            if (orig_unc.min_horizontal_uncertainty == 0.0 or
                    orig_unc.max_horizontal_uncertainty == 0.0):
                add_simplified_uncertainty = True
            elif origin.origin_uncertainty is not None:
                add_uncertainty = True
            # Following will work once Ellipsoid class added
            # if hasattr(origin.origin_uncertainty, 'confidence_ellipsoid'):
            #     cov = Ellipsoid.from_confidence_ellipsoid(
            #       origin.origin_uncertainty['confidence_ellipsoid']).to_cov()
            #     errors['x_err'] = sqrt(cov(0, 0)) / 1000.0
            #     errors['y_err'] = sqrt(cov(1, 1)) / 1000.0
            #     errors['z_err'] = sqrt(cov(2, 2)) / 1000.0
            #     # xy_, xz_, then yz_cov fields
            #     error_line[43:55] = ("%.4e" % (cov(0, 1) / 1.e06)).rjust(12)
            #     error_line[55:67] = ("%.4e" % (cov(0, 2) / 1.e06)).rjust(12)
            #     error_line[67:79] = ("%.4e" % (cov(1, 2) / 1.e06)).rjust(12)
            # else:

    if origin.depth_errors and origin.depth_errors.uncertainty:
        errors['z_err'] = origin.depth_errors.uncertainty / 1000.0
    else:
        errors['z_err'] = None
    if add_uncertainty:
        cov = Ellipse.from_origin_uncertainty(origin.origin_uncertainty).\
              to_cov()
        errors['x_err'] = sqrt(cov[0][0][0]) / 1000.0
        errors['y_err'] = sqrt(cov[0][1][1]) / 1000.0
        # xy covariance field
        error_line[43:55] = ("%.4e" % (cov[0][0][1] / 1.e06)).rjust(12)
    elif add_simplified_uncertainty:  # Deal with Zero uncertainty
        errors['x_err'] = 0.0
        errors['y_err'] = 0.0
    else:
        # Only return without writing error-line when no errors available
        if not (origin.longitude_errors.uncertainty and
                origin.latitude_errors.uncertainty and
                origin.depth_errors.uncertainty):
            return ''.join(error_line)
        try:
            errors['x_err'] = (origin.longitude_errors.uncertainty /
                               _km_to_deg_lon(1.0, origin.latitude))
        except AttributeError:
            pass
        try:
            errors['y_err'] = (origin.latitude_errors.uncertainty /
                               _km_to_deg_lat(1.0))
        except AttributeError:
            pass

    error_line[24:30] = (_str_conv(errors['y_err'], 1)).rjust(6)[0:6]
    error_line[32:38] = (_str_conv(errors['x_err'], 1)).rjust(6)[0:6]
    error_line[38:43] = (_str_conv(errors['z_err'], 1)).rjust(5)[0:5]
    return ''.join(error_line)


def _write_comment(comment):
    """
    Write comment to s-file

    :param comment: comment
    :type comment: `~obspy.core.event.Comment`
    :returns: List of String

    """
    comment_line = list(' ' * 79 + '3')
    if comment.text is None:
        return None
    comment_str = comment.text
    # Check if it's a comment line containing a Seisan-waveform
    if comment_str.startswith('Waveform-filename: '):
        comment_str = re.sub('^Waveform-filename: ', '', comment_str)
        comment_line[-1] = '6'

    # Check if it's a type-I line comment:
    if "ACTION" in comment_str.upper() and comment_str[-2:] == ' I':
        comment_line[-1] = 'I'

    n_comment_chars = len(comment_str)
    if n_comment_chars > 78:
        UserWarning('Writing of comment-lines to S-file does not currently'
                    'support lines longer than 78 characters. Will cut line'
                    'for printing to file.')
        n_comment_chars = 78
    comment_line[1:n_comment_chars+1] = comment_str[:n_comment_chars]
    comment_line = ''.join(comment_line)
    return comment_line


def nordpick(event, high_accuracy=True, nordic_format='OLD'):
    """
    Format picks in an :class:`~obspy.core.event.event.Event` to nordic.

    :type event: :class:`~obspy.core.event.event.Event`
    :param event: A single obspy event.
    :type high_accuracy: bool
    :param high_accuracy:
        Whether to output pick seconds at 6.3f (high_accuracy) or
        5.2f (standard).
    :type nordic_format: str
    :param nordic_format:
        Version of Nordic format to be used for output, either OLD or NEW.

    :returns: List of String

    .. note::

        Nordic files contain an angle of incidence ("AIN") that is actually the
        takeoff angle from the source, and hence now properly supported as
        arrival.takeoff_angle.
        Multiple weights are not supported.
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
        # Extract weight - should be stored as 0-4, or 9 for seisan.
        try:
            weight = pick.extra.get('nordic_pick_weight')['value']
        except AttributeError:
            weight = ' '
        # Extract velocity: Note that horizontal slowness in quakeML is stored
        # as s/deg and Seisan stores apparent velocity in km/s
        if pick.horizontal_slowness:
            velocity = degrees2kilometers(1.0 / pick.horizontal_slowness)
        else:
            velocity = ' '
        backazimuth = _str_conv(pick.backazimuth)
        # Extract the correct arrival info for this pick - assuming only one
        # arrival per pick...
        arrival = [arrival for arrival in origin.arrivals
                   if arrival.pick_id == pick.resource_id]
        if len(arrival) > 0:
            if len(arrival) > 1:
                warnings.warn("Multiple arrivals for pick - only writing one")
            arrival = arrival[0]
            # Extract azimuth residual
            if arrival.backazimuth_residual is not None:
                backazimuthres = _str_conv(int(arrival.backazimuth_residual))
            else:
                backazimuthres = ' '
            if arrival.takeoff_angle is not None:
                ain = _str_conv(arrival.takeoff_angle, rounded=1)
            else:
                ain = ' '
            # Extract time residual
            timeres = ' '
            if hasattr(arrival, 'time_residual'):
                if arrival.time_residual is not None:
                    timeres = _str_conv(arrival.time_residual, rounded=2)
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
            # Extract finalweight
            finalweight = '  '
            if arrival.time_weight is not None:
                finalweight = _str_conv(int(round(
                    arrival.time_weight * 10))).rjust(2)[0:2]
            if backazimuth != ' ':
                if arrival.backazimuth_weight is not None:
                    finalweight = _str_conv(int(round(
                        arrival.backazimuth_weight * 10))).rjust(2)[0:2]
        else:
            (caz, distance, timeres, backazimuthres, finalweight, ain) = (
                ' ', ' ', ' ', ' ', '  ', ' ')
        phase_hint = pick.phase_hint or ' '
        # Extract amplitude: note there can be multiple amplitudes, but they
        # should be associated with different picks.
        amplitudes = [amplitude for amplitude in event.amplitudes
                      if amplitude.pick_id == pick.resource_id]
        amp_list = []
        if len(amplitudes) > 0:
            if len(amplitudes) > 1 and nordic_format == 'OLD':
                msg = 'Nordic files need one pick for each amplitude, ' + \
                    'using the first amplitude only'
                warnings.warn(msg)
            # amplitude = amplitude[0]
            for amplitude in amplitudes:
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
                        # Otherwise we assume that the amplitude is in counts
                    else:
                        amp = None
                    coda = ' '
                    mag_hint = (amplitude.magnitude_hint or amplitude.type)
                    if (mag_hint is not None and
                            mag_hint.upper() in ['AML', 'ML']):
                        phase_hint = 'IAML'
                        impulsivity = ' '
                else:
                    coda = str(int(amplitude.generic_amplitude))
                    peri = ' '
                    peri_round = False
                    amp = None
                    coda_eval_mode = INV_EVALUTATION_MAPPING.get(
                        amplitude.evaluation_mode, ' ')
                if nordic_format == 'OLD':  # only use 1st amplitude
                    break
                amp_list.append(amp)
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
        if len(channel_code) == 1:
            channel_code = '  ' + channel_code[-1]
        if len(channel_code) == 2:
            channel_code = channel_code[0] + ' ' + channel_code[-1]
        network_code = pick.waveform_id.network_code or '  '
        location_code = pick.waveform_id.location_code or '  '
        # Seisan doesn't accept questions marks, need to replace with space
        network_code = network_code.replace('?', ' ')
        location_code = location_code.replace('?', ' ')
        channel_code = channel_code.replace('?', ' ')
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

        # Differentiate based on Nordic format versions
        if nordic_format == 'OLD':
            if len(phase_hint) > 4:
                # Weight goes in 9 and phase_hint runs through 11-18
                if polarity != ' ':
                    UserWarning(
                        "Polarity not written due to phase hint length")
                phase_info = (_str_conv(weight).rjust(1) + impulsivity
                              + phase_hint.ljust(8))
            else:
                phase_info = (
                    ' ' + impulsivity + phase_hint.ljust(4) +
                    _str_conv(weight).rjust(1) + eval_mode +
                    polarity.rjust(1) + ' ')
            pick_string_formatter = (
                " {station:5s}{instrument:1s}{component:1s}{phase_info:10s}"
                "{hour:2d}{minute:2d}{seconds:>6s}{coda:5s}{amp:7s}{period:5s}"
                "{backazimuth:6s}{velocity:5s}{ain:4s}{backazimuthres:3s}"
                "{timeres:5s}{finalweight:2s}{distance:5s}{caz:4s} ")
            # Note that pick seconds rounding only works because SEISAN does
            # not enforce that seconds stay 0 <= seconds < 60, so rounding
            # something like seconds = 59.997 to 2dp gets to 60.00, which
            # SEISAN is happy with. It appears that SEISAN is happy with large
            # numbers of seconds see #2348.
            pick_strings.append(pick_string_formatter.format(
                station=pick.waveform_id.station_code,
                instrument=channel_code[0], component=channel_code[-1],
                phase_info=phase_info, hour=pick_hour,
                minute=pick.time.minute,
                seconds=_str_conv(pick_seconds, rounded=pick_rounding),
                coda=_str_conv(coda).rjust(5)[0:5],
                amp=_str_conv(amp, rounded=1).rjust(7)[0:7],
                period=_str_conv(peri, rounded=peri_round).rjust(5)[0:5],
                backazimuth=_str_conv(backazimuth).rjust(6)[0:6],
                velocity=_str_conv(velocity).rjust(5)[0:5],
                ain=ain[:-2].rjust(4)[0:4],
                backazimuthres=_str_conv(backazimuthres).rjust(3)[0:3],
                timeres=_str_conv(timeres, rounded=2).rjust(5)[0:5],
                finalweight=finalweight, distance=distance.rjust(5)[0:5],
                caz=_str_conv(caz).rjust(4)[0:4]))
            # Nordic files contain an angle of incidence ("AIN") that is
            # actually the takeoff angle from the source, and hence now
            # properly supported as arrival.takeoff_angle.
        elif nordic_format == 'NEW':
            # Define par1, par2, & residual depending on type of observation:
            # Coda, backzimuth (add extra line), amplitude, or other phase pick
            add_amp_line = False
            add_coda_line = False
            is_amp_pick = False
            add_baz_line = False
            par1 = '       '
            par2 = '      '
            residual = '     '
            # Phase pick
            if polarity.strip() != '':
                par1 = '      ' + polarity
            if hasattr(arrival, 'time_residual'):
                if arrival.time_residual is not None:
                    residual = _str_conv(arrival.time_residual, rounded=2
                                         ).rjust(5)[0:5]
            # Coda
            if coda.strip() != '':
                add_coda_line = True
                coda_phase_hint = 'END'
                coda_par1 = _str_conv(coda).rjust(7)[0:7]  # coda duration
                coda_par2 = '      '
                # TODO: magnitude residual for coda
                coda_residual = '     '
                # TODO: weight for coda
            # Back Azimuth
            if backazimuth.strip() != '':  # back-azimuth
                add_baz_line = True
                # If the BAZ-measurement is an extra pick in addition to the
                # actual phase, then don't duplicate the BAZ-line. Instead,
                # write the BAZ-pick into a single line.
                if pick.phase_hint and pick.phase_hint.startswith('BAZ-'):
                    add_baz_line = False
                if len(phase_hint) <= 4:  # max total phase name length is 8
                    baz_phase_hint = 'BAZ-' + phase_hint
                else:
                    baz_phase_hint = phase_hint
                baz_par1 = _str_conv(backazimuth, rounded=1).rjust(7)[0:7]
                baz_par2 = _str_conv(velocity, rounded=2).rjust(6)[0:6]
                baz_residual = '     '
                baz_finalweight = '  '
                if arrival:
                    if arrival.backazimuth_residual is not None:
                        baz_residual = _str_conv(arrival.backazimuth_residual,
                                                 rounded=1).rjust(5)[0:5]
                    if arrival.backazimuth_weight is not None:
                        baz_finalweight = _str_conv(
                            arrival.backazimuth_weight*10,
                            rounded=0).rjust[2][0:2]
                if not add_baz_line:
                    par1 = baz_par1
                    par2 = baz_par2
                    residual = baz_residual
                    finalweight = baz_finalweight
            # Amplitude
            if amp is not None:
                add_amp_line = True
                # In New Nordic format, multiple amplitudes can now be associ-
                # ated with one pick (e.g., measured at different periods)
                amp_phase_hints, amp_eval_modes, amp_finalweights = [], [], []
                amp_par1s, amp_par2s = [], []
                amp_residuals, mag_residuals = [], []
                for j, amp in enumerate(amp_list):
                    # check if the amplitude and pick reference the same phase
                    # - then the amplitude in the line below the pick.
                    if (pick.phase_hint and
                            (pick.phase_hint == amplitudes[j].type or
                             pick.phase_hint[1:] == amplitudes[j].type)
                            and _is_iasp_ampl_phase(pick.phase_hint)):
                        is_amp_pick = True
                    mag_hint = (
                        amplitudes[j].magnitude_hint or amplitudes[j].type)
                    if (mag_hint is not None and
                            mag_hint.upper() in ['AML', 'ML']):
                        amp_phase_hints.append('IAML')
                    else:
                        if amplitudes[j].type is not None:
                            amp_phase_hints.append(amplitudes[j].type)
                        else:  # Generic amplitude
                            amp_phase_hints.append('A')
                    amp_eval_modes.append(' ' or INV_EVALUTATION_MAPPING.get(
                        amplitude.evaluation_mode, None))
                    amp_finalweights.append('  ')
                    amp_par1s.append(_str_conv(amp, rounded=1).rjust(7)[0:7])
                    amp_par2s.append(
                        _str_conv(peri, rounded=peri_round).rjust(6)[0:6])
                    # Get StationMagnitude that corresponds to the amplitude to
                    # print magnitude residual
                    tr_mag = [
                        sta_mag for sta_mag in event.station_magnitudes
                        if (sta_mag.amplitude_id == amplitudes[j].resource_id
                            and (not sta_mag.creation_info or
                                 sta_mag.creation_info.agency_id
                                 == pick.creation_info.agency_id)
                            and sta_mag.station_magnitude_type
                            == amplitudes[j].magnitude_hint)]
                    amp_residuals.append('     ')
                    if len(tr_mag) > 0:
                        if len(tr_mag) > 1:
                            msg = ('Nordic files need one trace-amplitude for '
                                   + 'each trace / station-magnitude only.')
                            warnings.warn(msg)
                        mag_residuals.append(tr_mag[0].mag_errors.uncertainty)
                        amp_residuals[j] = _str_conv(
                            mag_residuals[j], rounded=2).rjust(5)[0:5]

            agency = '   '
            author = '   '
            if pick.creation_info is not None:
                agency = _get_agency_id(pick)
                if pick.creation_info.author is not None:
                    author = (pick.creation_info.author.ljust(3)[0:3] or '   ')

            pick_string_formatter = (
                " {station:5s}{channel:3s} {network:2s}{location:2s} "
                "{impulsivity:1s}{phase_hint:8s}{weight:1s}{eval_mode:1s}"
                "{hour:2d}{minute:2d} {seconds:6s}"
                "{par1:7s}{par2:6s} {agency:3s} {author:3s}"
                "{ain:5s}{residual:5s}{finalweight:2s}"
                "{distance:5s} {caz:3s} ")
            if not is_amp_pick:
                pick_strings.append(pick_string_formatter.format(
                    station=pick.waveform_id.station_code,
                    channel=channel_code, network=network_code,
                    location=location_code, impulsivity=impulsivity,
                    phase_hint=phase_hint, weight=_str_conv(weight).rjust(1),
                    eval_mode=eval_mode, hour=pick_hour,
                    minute=pick.time.minute,
                    seconds=_str_conv(pick_seconds, rounded=3).rjust(6),
                    par1=par1, par2=par2, agency=agency, author=author,
                    ain=ain.rjust(5)[0:5], residual=residual,
                    finalweight=finalweight,
                    distance=distance.rjust(5)[0:5],
                    caz=_str_conv(caz).rjust(3)[0:3]))
            if add_coda_line:
                pick_strings.append(pick_string_formatter.format(
                    station=pick.waveform_id.station_code,
                    channel=channel_code, network=network_code,
                    location=location_code, impulsivity=' ',
                    phase_hint=coda_phase_hint.ljust(8)[0:8],
                    weight=' ', eval_mode=coda_eval_mode,
                    hour=pick_hour, minute=pick.time.minute,
                    seconds=_str_conv(pick_seconds, rounded=3).rjust(6),
                    par1=coda_par1, par2=coda_par2, agency=agency,
                    author=author, ain='     ', residual=coda_residual,
                    finalweight=' ', distance=distance.rjust(5)[0:5],
                    caz=_str_conv(caz).rjust(3)[0:3]))
            if add_amp_line:
                for j, amp in enumerate(amp_list):
                    pick_strings.append(pick_string_formatter.format(
                        station=pick.waveform_id.station_code,
                        channel=channel_code, network=network_code,
                        location=location_code, impulsivity=' ',
                        phase_hint=amp_phase_hints[j].ljust(8)[0:8],
                        weight=' ', eval_mode=amp_eval_modes[j],
                        hour=pick_hour, minute=pick.time.minute,
                        seconds=_str_conv(pick_seconds, rounded=3).rjust(6),
                        par1=amp_par1s[j], par2=amp_par2s[j], agency=agency,
                        author=author, ain='     ', residual=amp_residuals[j],
                        finalweight=amp_finalweights[j],
                        distance=distance.rjust(5)[0:5],
                        caz=_str_conv(caz).rjust(3)[0:3]))
            if add_baz_line:
                pick_strings.append(pick_string_formatter.format(
                    station=pick.waveform_id.station_code,
                    channel=channel_code, network=network_code,
                    location=location_code, impulsivity=' ',
                    phase_hint=baz_phase_hint,
                    weight=_str_conv(weight).rjust(1), eval_mode=eval_mode,
                    hour=pick_hour, minute=pick.time.minute,
                    seconds=_str_conv(pick_seconds, rounded=3).rjust(6),
                    par1=baz_par1, par2=baz_par2, agency=agency, author=author,
                    ain='     ', residual=baz_residual,
                    finalweight=baz_finalweight,
                    distance=distance.rjust(5)[0:5],
                    caz=_str_conv(caz).rjust(3)[0:3]))
        else:
            raise ValueError('Nordic format can be ''OLD'' or ''NEW'', not '
                             + nordic_format)

    return pick_strings


if __name__ == "__main__":
    import doctest
    doctest.testmod()
