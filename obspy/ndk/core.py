#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NDK file support for ObsPy

The format is an ASCII format but will internally handled by unicode routines.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

from future import standard_library
with standard_library.hooks():
    import itertools

import math
import re
import traceback
import warnings
import uuid

from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Origin, CreationInfo, Magnitude, \
    EventDescription, Comment, FocalMechanism, MomentTensor, NodalPlanes, \
    PrincipalAxes, Axis, NodalPlane, Tensor, DataUsed, SourceTimeFunction
from obspy.core.util.geodetics import FlinnEngdahl


class ObsPyNDKException(Exception):
    """
    Base Exception class for this module.
    """
    pass


class ObsPyNDKWarning(UserWarning):
    """
    Base warning for this module.
    """
    pass


def _get_resource_id(cmtname, res_type, tag=None):
    """
    Helper function to create consistent resource ids.
    """
    res_id = "smi:local/ndk/%s/%s" % (cmtname, res_type)
    if tag is not None:
        res_id += "#" + tag
    return res_id


def _parse_date_time(date, time):
    """
    Function taking a tuple of date and time string from an NDK file and
    converting it to an UTCDateTime object.

    In particular it is able to deal with a time string specifying 60
    seconds which is not a valid ISO time string but occurs a lot in NDK
    files.
    """
    add_minute = False
    if ":60.0" in time:
        time = time.replace(":60.0", ":0.0")
        add_minute = True
    try:
        dt = UTCDateTime(date.replace("/", "-") + "T" + time)
    except (TypeError, ValueError):
        msg = ("Could not parse date/time string '%s' and '%s' to a valid "
               "time" % (date, time))
        raise ObsPyNDKException(msg)

    if add_minute:
        dt += 60.0
    return dt


def is_ndk(filename):
    """
    Checks that a file is actually an NDK file.

    It will read the first line and check to see if the date, time, and the
    location are valid. Then it assumes the file is an NDK file.
    """
    # Get the first line.
    # Not a file-like object.
    if not hasattr(filename, "readline"):
        # Check if it exists, otherwise assume its a string.
        try:
            with open(filename, "rt") as fh:
                first_line = fh.readline()
        except:
            try:
                filename = filename.decode()
            except:
                filename = str(filename)
            filename = filename.strip()
            line_ending = filename.find("\n")
            if line_ending == -1:
                return False
            first_line = filename[:line_ending]
    # File like object.
    else:
        first_line = filename.readline()
        if hasattr(first_line, "decode"):
            first_line = first_line.decode()

    # A certain minimum length is required to extract all the following
    # parameters.
    if len(first_line) < 46:
        return False

    date = first_line[5:15].strip()
    time = first_line[16:26]

    # Assemble the time.
    try:
        _parse_date_time(date, time)
    except ObsPyNDKException:
        return False

    try:
        latitude = float(first_line[27:33])
        longitude = float(first_line[34:41])
        depth = float(first_line[42:47])
    except ValueError:
        return False

    if (-90.0 <= latitude <= 90.0) and \
            (-180.0 <= longitude <= 180.0) and \
            (0 <= depth <= 800000):
        return True
    return False


def read_ndk(filename, *args, **kwargs):  # @UnusedVariable
    """
    Reads an NDK file to a :class:`~obspy.core.event.Catalog` object.

    :param filename: File or file-like object in text mode.
    """
    # Read the whole file at once. While an iterator would be more efficient
    # the largest NDK file out in the wild is 13.7 MB so it does not matter
    # much.
    if not hasattr(filename, "read"):
        # Check if it exists, otherwise assume its a string.
        try:
            with open(filename, "rt") as fh:
                data = fh.read()
        except:
            try:
                data = filename.decode()
            except:
                data = str(filename)
            data = data.strip()
    else:
        data = filename.read()
        if hasattr(data, "decode"):
            data = data.decode()

    # Create iterator that yields lines.
    def lines_iter():
        prev_line = -1
        while True:
            next_line = data.find("\n", prev_line + 1)
            if next_line < 0:
                break
            yield data[prev_line + 1: next_line]
            prev_line = next_line
        if len(data) > prev_line + 1:
            yield data[prev_line + 1:]
        raise StopIteration

    # Use one Flinn Engdahl object for all region determinations.
    fe = FlinnEngdahl()
    cat = Catalog(resource_id=_get_resource_id("catalog", str(uuid.uuid4())))

    # Loop over 5 lines at once.
    for _i, lines in enumerate(itertools.zip_longest(*[lines_iter()] * 5)):
        if None in lines:
            msg = "Skipped last %i lines. Not a multiple of 5 lines." % (
                lines.count(None))
            warnings.warn(msg, ObsPyNDKWarning)
            continue

        # Parse the lines to a human readable dictionary.
        try:
            record = _read_lines(*lines)
        except (ValueError, ObsPyNDKException):
            exc = traceback.format_exc()
            msg = (
                "Could not parse event %i (faulty file?). Will be "
                "skipped. Lines of the event:\n"
                "\t%s\n"
                "%s") % (_i + 1, "\n\t".join(lines), exc)
            warnings.warn(msg, ObsPyNDKWarning)
            continue

        # Use one creation info for essentially every item.
        creation_info = CreationInfo(
            agency_id="GCMT",
            version=record["version_code"]
        )

        # Use the ObsPy Flinn Engdahl region determiner as the region in the
        # NDK files is oftentimes trimmed.
        region = fe.get_region(record["centroid_longitude"],
                               record["centroid_latitude"])

        # Create an event object.
        event = Event(
            force_resource_id=False,
            event_type="earthquake",
            event_type_certainty="known",
            event_descriptions=[
                EventDescription(text=region, type="Flinn-Engdahl region"),
                EventDescription(text=record["cmt_event_name"],
                                 type="earthquake name")
            ]
        )

        # Assemble the time for the reference origin.
        try:
            time = _parse_date_time(record["date"], record["time"])
        except ObsPyNDKException:
            msg = ("Invalid time in event %i. '%s' and '%s' cannot be "
                   "assembled to a valid time. Event will be skipped.") % \
                  (_i + 1, record["date"], record["time"])
            warnings.warn(msg, ObsPyNDKWarning)
            continue

        # Create two origins, one with the reference latitude/longitude and
        # one with the centroidal values.
        ref_origin = Origin(
            force_resource_id=False,
            time=time,
            longitude=record["hypo_lng"],
            latitude=record["hypo_lat"],
            # Convert to m.
            depth=record["hypo_depth_in_km"] * 1000.0,
            origin_type="hypocenter",
            comments=[Comment(text="Hypocenter catalog: %s" %
                              record["hypocenter_reference_catalog"],
                              force_resource_id=False)]
        )
        ref_origin.comments[0].resource_id = _get_resource_id(
            record["cmt_event_name"], "comment", tag="ref_origin")
        ref_origin.resource_id = _get_resource_id(record["cmt_event_name"],
                                                  "origin", tag="reforigin")

        cmt_origin = Origin(
            force_resource_id=False,
            longitude=record["centroid_longitude"],
            longitude_errors={
                "uncertainty": record["centroid_longitude_error"]},
            latitude=record["centroid_latitude"],
            latitude_errors={
                "uncertainty": record["centroid_latitude_error"]},
            # Convert to m.
            depth=record["centroid_depth_in_km"] * 1000.0,
            depth_errors={
                "uncertainty": record["centroid_depth_in_km_error"] * 1000},
            time=ref_origin["time"] + record["centroid_time"],
            time_errors={"uncertainty": record["centroid_time_error"]},
            depth_type=record["type_of_centroid_depth"],
            origin_type="centroid",
            time_fixed=False,
            epicenter_fixed=False,
            creation_info=creation_info.copy()
        )
        cmt_origin.resource_id = _get_resource_id(record["cmt_event_name"],
                                                  "origin",
                                                  tag="cmtorigin")
        event.origins = [ref_origin, cmt_origin]
        event.preferred_origin_id = cmt_origin.resource_id.id

        # Create the magnitude object.
        mag = Magnitude(
            force_resource_id=False,
            mag=round(record["Mw"], 2),
            magnitude_type="Mwc",
            origin_id=cmt_origin.resource_id,
            creation_info=creation_info.copy()
        )
        mag.resource_id = _get_resource_id(record["cmt_event_name"],
                                           "magnitude", tag="moment_mag")
        event.magnitudes = [mag]
        event.preferred_magnitude_id = mag.resource_id.id

        # Add the reported mb, MS magnitudes as additional magnitude objects.
        event.magnitudes.append(Magnitude(
            force_resource_id=False,
            mag=record["mb"],
            magnitude_type="mb",
            comments=[Comment(
                force_resource_id=False,
                text="Reported magnitude in NDK file. Most likely 'mb'."
            )]
        ))
        event.magnitudes[-1].comments[-1].resource_id = _get_resource_id(
            record["cmt_event_name"], "comment", tag="mb_magnitude")
        event.magnitudes[-1].resource_id = _get_resource_id(
            record["cmt_event_name"], "magnitude", tag="mb")

        event.magnitudes.append(Magnitude(
            force_resource_id=False,
            mag=record["MS"],
            magnitude_type="MS",
            comments=[Comment(
                force_resource_id=False,
                text="Reported magnitude in NDK file. Most likely 'MS'."
            )]
        ))
        event.magnitudes[-1].comments[-1].resource_id = _get_resource_id(
            record["cmt_event_name"], "comment", tag="MS_magnitude")
        event.magnitudes[-1].resource_id = _get_resource_id(
            record["cmt_event_name"], "magnitude", tag="MS")

        # Take care of the moment tensor.
        tensor = Tensor(
            m_rr=record["m_rr"],
            m_rr_errors={"uncertainty": record["m_rr_error"]},
            m_pp=record["m_pp"],
            m_pp_errors={"uncertainty": record["m_pp_error"]},
            m_tt=record["m_tt"],
            m_tt_errors={"uncertainty": record["m_tt_error"]},
            m_rt=record["m_rt"],
            m_rt_errors={"uncertainty": record["m_rt_error"]},
            m_rp=record["m_rp"],
            m_rp_errors={"uncertainty": record["m_rp_error"]},
            m_tp=record["m_tp"],
            m_tp_errors={"uncertainty": record["m_tp_error"]},
            creation_info=creation_info.copy()
        )
        mt = MomentTensor(
            force_resource_id=False,
            scalar_moment=record["scalar_moment"],
            tensor=tensor,
            data_used=[DataUsed(**i) for i in record["data_used"]],
            inversion_type=record["source_type"],
            source_time_function=SourceTimeFunction(
                type=record["moment_rate_type"],
                duration=record["moment_rate_duration"]
            ),
            derived_origin_id=cmt_origin.resource_id,
            creation_info=creation_info.copy()
        )
        mt.resource_id = _get_resource_id(record["cmt_event_name"],
                                          "momenttensor")
        axis = [Axis(**i) for i in record["principal_axis"]]
        focmec = FocalMechanism(
            force_resource_id=False,
            moment_tensor=mt,
            principal_axes=PrincipalAxes(
                # The ordering is the same as for the IRIS SPUD service and
                # from a website of the Saint Louis University Earthquake
                # center so it should be correct.
                t_axis=axis[0],
                p_axis=axis[2],
                n_axis=axis[1]
            ),
            nodal_planes=NodalPlanes(
                nodal_plane_1=NodalPlane(**record["nodal_plane_1"]),
                nodal_plane_2=NodalPlane(**record["nodal_plane_2"])
            ),
            comments=[
                Comment(force_resource_id=False,
                        text="CMT Analysis Type: %s" %
                             record["cmt_type"].capitalize()),
                Comment(force_resource_id=False,
                        text="CMT Timestamp: %s" %
                             record["cmt_timestamp"])],
            creation_info=creation_info.copy()
        )
        focmec.comments[0].resource_id = _get_resource_id(
            record["cmt_event_name"], "comment", tag="cmt_type")
        focmec.comments[1].resource_id = _get_resource_id(
            record["cmt_event_name"], "comment", tag="cmt_timestamp")
        focmec.resource_id = _get_resource_id(record["cmt_event_name"],
                                              "focal_mechanism")
        event.focal_mechanisms = [focmec]
        event.preferred_focal_mechanism_id = focmec.resource_id.id

        # Set at end to avoid duplicate resource id warning.
        event.resource_id = _get_resource_id(record["cmt_event_name"],
                                             "event")

        cat.append(event)

    if len(cat) == 0:
        msg = "No valid events found in NDK file."
        raise ObsPyNDKException(msg)

    return cat


def _read_lines(line1, line2, line3, line4, line5):
    # First line: Hypocenter line
    # [1-4]   Hypocenter reference catalog (e.g., PDE for USGS location,
    #         ISC for #ISC catalog, SWE for surface-wave location,
    #         [Ekstrom, BSSA, 2006])
    # [6-15]  Date of reference event
    # [17-26] Time of reference event
    # [28-33] Latitude
    # [35-41] Longitude
    # [43-47] Depth
    # [49-55] Reported magnitudes, usually mb and MS
    # [57-80] Geographical location (24 characters)
    rec = {}
    rec["hypocenter_reference_catalog"] = line1[:4].strip()
    rec["date"] = line1[5:15].strip()
    rec["time"] = line1[16:26]
    rec["hypo_lat"] = float(line1[27:33])
    rec["hypo_lng"] = float(line1[34:41])
    rec["hypo_depth_in_km"] = float(line1[42:47])
    rec["mb"], rec["MS"] = map(float, line1[48:55].split())
    rec["location"] = line1[56:80].strip()

    # Second line: CMT info (1)
    # [1-16]  CMT event name. This string is a unique CMT-event identifier.
    #         Older events have 8-character names, current ones have
    #         14-character names.  See note (1) below for the naming
    #         conventions used.
    # [18-61] Data used in the CMT inversion. Three data types may be used:
    #         Long-period body waves (B), Intermediate-period surface waves
    #         (S), and long-period mantle waves (M). For each data type,
    #         three values are given: the number of stations used, the
    #         number  of components used, and the shortest period used.
    # [63-68] Type of source inverted for:
    #         "CMT: 0" - general moment tensor;
    #         "CMT: 1" - moment tensor with constraint of zero trace
    #             (standard);
    #         "CMT: 2" - double-couple source.
    # [70-80] Type and duration of moment-rate function assumed in the
    #         inversion.  "TRIHD" indicates a triangular moment-rate
    #         function, "BOXHD" indicates a boxcar moment-rate function.
    #         The value given is half the duration of the moment-rate
    #         function. This value is assumed in the inversion, following a
    #         standard scaling relationship (see note (2) below), and is
    #         not derived from the analysis.
    rec["cmt_event_name"] = line2[:16].strip()

    data_used = line2[17:61].strip()
    # Use regex to get the data used in case the data types are in a
    # different order.
    data_used = re.findall(r"[A-Z]:\s*\d+\s+\d+\s+\d+", data_used)
    rec["data_used"] = []
    for data in data_used:
        data_type, count = data.split(":")
        if data_type == "B":
            data_type = "body waves"
        elif data_type == "S":
            data_type = "surface waves"
        elif data_type == "M":
            data_type = "mantle waves"
        else:
            msg = "Unknown data type '%s'." % data_type
            raise ObsPyNDKException(msg)

        sta, comp, period = count.strip().split()

        rec["data_used"].append({
            "wave_type": data_type,
            "station_count": int(sta),
            "component_count": int(comp),
            "shortest_period": float(period)
        })

    source_type = line2[62:68].strip().upper().replace(" ", "")
    if source_type == "CMT:0":
        rec["source_type"] = "general"
    elif source_type == "CMT:1":
        rec["source_type"] = "zero trace"
    elif source_type == "CMT:2":
        rec["source_type"] = "double couple"
    else:
        msg = "Unknown source type."
        raise ObsPyNDKException(msg)

    mr_type, mr_duration = [i.strip() for i in line2[69:].split(":")]
    mr_type = mr_type.strip().upper()
    if mr_type == "TRIHD":
        rec["moment_rate_type"] = "triangle"
    elif mr_type == "BOXHD":
        rec["moment_rate_type"] = "box car"
    else:
        msg = "Moment rate function '%s' unknown." % mr_type
        raise ObsPyNDKException(msg)

    # Specified as half the duration in the file.
    rec["moment_rate_duration"] = float(mr_duration) * 2.0

    # Third line: CMT info (2)
    # [1-58]  Centroid parameters determined in the inversion. Centroid
    #         time, given with respect to the reference time, centroid
    #         latitude, centroid longitude, and centroid depth. The value
    #         of each variable is followed by its estimated standard error.
    #         See note (3) below for cases in which the hypocentral
    #         coordinates are held fixed.
    # [60-63] Type of depth. "FREE" indicates that the depth was a result
    #         of the inversion; "FIX " that the depth was fixed and not
    #         inverted for; "BDY " that the depth was fixed based on
    #         modeling of broad-band P waveforms.
    # [65-80] Timestamp. This 16-character string identifies the type of
    #         analysis that led to the given CMT results and, for recent
    #         events, the date and time of the analysis. This is useful to
    #         distinguish Quick CMTs ("Q-"), calculated within hours of an
    #         event, from Standard CMTs ("S-"), which are calculated later.
    rec["centroid_time"], rec["centroid_time_error"], \
        rec["centroid_latitude"], rec["centroid_latitude_error"], \
        rec["centroid_longitude"], rec["centroid_longitude_error"], \
        rec["centroid_depth_in_km"], rec["centroid_depth_in_km_error"] = \
        map(float, line3[:58].split()[1:])
    type_of_depth = line3[59:63].strip().upper()

    if type_of_depth == "FREE":
        rec["type_of_centroid_depth"] = "from moment tensor inversion"
    elif type_of_depth == "FIX":
        rec["type_of_centroid_depth"] = "from location"
    elif type_of_depth == "BDY":
        rec["type_of_centroid_depth"] = "from modeling of broad-band P " \
                                        "waveforms"
    else:
        msg = "Unknown type of depth '%s'." % type_of_depth
        raise ObsPyNDKException(msg)

    timestamp = line3[64:].strip().upper()
    rec["cmt_timestamp"] = timestamp
    if timestamp.startswith("Q-"):
        rec["cmt_type"] = "quick"
    elif timestamp.startswith("S-"):
        rec["cmt_type"] = "standard"
    # This is invalid but occurs a lot so we include it here.
    elif timestamp.startswith("O-"):
        rec["cmt_type"] = "unknown"
    else:
        msg = "Invalid CMT timestamp '%s' for event %s." % (
            timestamp, rec["cmt_event_name"])
        raise ObsPyNDKException(msg)

    # Fourth line: CMT info (3)
    # [1-2]   The exponent for all following moment values. For example, if
    #         the exponent is given as 24, the moment values that follow,
    #         expressed in dyne-cm, should be multiplied by 10**24.
    # [3-80]  The six moment-tensor elements: Mrr, Mtt, Mpp, Mrt, Mrp, Mtp,
    #         where r is up, t is south, and p is east. See Aki and
    #         Richards for conversions to other coordinate systems. The
    #         value of each moment-tensor element is followed by its
    #         estimated standard error. See note (4) below for cases in
    #         which some elements are constrained in the inversion.
    # Exponent converts to dyne*cm. To convert to N*m it has to be decreased
    # seven orders of magnitude.
    exponent = int(line4[:2]) - 7
    # Directly set the exponent instead of calculating it to enhance
    # precision.
    rec["m_rr"], rec["m_rr_error"], rec["m_tt"], rec["m_tt_error"], \
        rec["m_pp"], rec["m_pp_error"], rec["m_rt"], rec["m_rt_error"], \
        rec["m_rp"], rec["m_rp_error"], rec["m_tp"], rec["m_tp_error"] = \
        map(lambda x: float("%sE%i" % (x, exponent)), line4[2:].split())

    # Fifth line: CMT info (4)
    # [1-3]   Version code. This three-character string is used to track
    #         the version of the program that generates the "ndk" file.
    # [4-48]  Moment tensor expressed in its principal-axis system:
    #         eigenvalue, plunge, and azimuth of the three eigenvectors.
    #         The eigenvalue should be multiplied by 10**(exponent) as
    #         given on line four.
    # [50-56] Scalar moment, to be multiplied by 10**(exponent) as given on
    #         line four.
    # [58-80] Strike, dip, and rake for first nodal plane of the
    #         best-double-couple mechanism, repeated for the second nodal
    #         plane.  The angles are defined as in Aki and Richards. The
    #         format for this string should not be considered fixed.
    rec["version_code"] = line5[:3].strip()
    rec["scalar_moment"] = float(line5[49:56]) * (10 ** exponent)
    # Calculate the moment magnitude.
    rec["Mw"] = 2.0 / 3.0 * (math.log10(rec["scalar_moment"]) - 9.1)

    principal_axis = line5[3:48].split()
    rec["principal_axis"] = []
    for axis in zip(*[iter(principal_axis)] * 3):
        rec["principal_axis"].append({
            # Again set the exponent directly to avoid even more rounding
            # errors.
            "length": "%sE%i" % (axis[0], exponent),
            "plunge": float(axis[1]),
            "azimuth": float(axis[2])
        })

    nodal_planes = map(float, line5[57:].strip().split())
    rec["nodal_plane_1"] = {
        "strike": next(nodal_planes),
        "dip": next(nodal_planes),
        "rake": next(nodal_planes)
    }
    rec["nodal_plane_2"] = {
        "strike": next(nodal_planes),
        "dip": next(nodal_planes),
        "rake": next(nodal_planes)
    }

    return rec


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
