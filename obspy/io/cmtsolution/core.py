# -*- coding: utf-8 -*-
"""
CMTSOLUTION file format support for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import math
import uuid
import warnings

from obspy import UTCDateTime
from obspy.core.event import (Catalog, Comment, Event, EventDescription,
                              Origin, Magnitude, FocalMechanism, MomentTensor,
                              Tensor, SourceTimeFunction)
from obspy.geodetics import FlinnEngdahl


_fe = FlinnEngdahl()


def _get_resource_id(cmtname, res_type, tag=None):
    """
    Helper function to create consistent resource ids.
    """
    res_id = "smi:local/cmtsolution/%s/%s" % (cmtname, res_type)
    if tag is not None:
        res_id += "#" + tag
    return res_id


def _buffer_proxy(filename_or_buf, function, reset_fp=True,
                  file_mode="rb", *args, **kwargs):
    """
    Calls a function with an open file or file-like object as the first
    argument. If the file originally was a filename, the file will be
    opened, otherwise it will just be passed to the underlying function.

    :param filename_or_buf: File to pass.
    :type filename_or_buf: str, open file, or file-like object.
    :param function: The function to call.
    :param reset_fp: If True, the file pointer will be set to the initial
        position after the function has been called.
    :type reset_fp: bool
    :param file_mode: Mode to open file in if necessary.
    """
    try:
        position = filename_or_buf.tell()
        is_buffer = True
    except AttributeError:
        is_buffer = False

    if is_buffer is True:
        ret_val = function(filename_or_buf, *args, **kwargs)
        if reset_fp:
            filename_or_buf.seek(position, 0)
        return ret_val
    else:
        with open(filename_or_buf, file_mode) as fh:
            return function(fh, *args, **kwargs)


def _is_cmtsolution(filename_or_buf):
    """
    Checks if the file is a CMTSOLUTION file.

    :param filename_or_buf: File to test.
    :type filename_or_buf: str or file-like object.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _buffer_proxy(filename_or_buf, _internal_is_cmtsolution,
                                 reset_fp=True)
    # Happens for example when passing the data as a string which would be
    # interpreted as a filename.
    except OSError:
        return False


def _internal_is_cmtsolution(buf):
    """
    Checks if the file is a CMTSOLUTION file.

    :param buf: File to check.
    :type buf: Open file or open file like object.
    """
    # The file format is so simple. Just attempt to read the first event. If
    # it passes it will be read again but that has really no
    # significant performance impact.
    try:
        _internal_read_single_cmtsolution(buf)
        return True
    except Exception:
        return False


def _read_cmtsolution(filename_or_buf, **kwargs):
    """
    Reads a CMTSOLUTION file to a :class:`~obspy.core.event.Catalog` object.

    :param filename_or_buf: File to read.
    :type filename_or_buf: str or file-like object.
    """
    return _buffer_proxy(filename_or_buf, _internal_read_cmtsolution, **kwargs)


def _internal_read_cmtsolution(buf, **kwargs):
    """
    Reads a CMTSOLUTION file to a :class:`~obspy.core.event.Catalog` object.

    :param buf: File to read.
    :type buf: Open file or open file like object.
    """
    events = []
    cur_pos = buf.tell()

    # This also works with BytesIO and what not.
    buf.seek(0, 2)
    size = buf.tell()
    buf.seek(cur_pos, 0)

    # This is pretty inefficient due to all the file pointer jumping but
    # performance is really the least of our concerns. Also most performance
    # is still lost initializing the large ObsPy event objects.
    while True:
        if buf.tell() >= size:
            break
        line = buf.readline().strip()

        # If there is something, jump back to the beginning of the line and
        # read the next event.
        if line:
            buf.seek(cur_pos, 0)
            events.append(_internal_read_single_cmtsolution(buf))
        cur_pos = buf.tell()

    return Catalog(resource_id=_get_resource_id("catalog", str(uuid.uuid4())),
                   events=events)


def _internal_read_single_cmtsolution(buf):
    """
    Reads a single CMTSOLUTION file to a :class:`~obspy.core.event.Catalog`
    object.

    :param buf: File to read.
    :type buf: Open file or open file like object.
    """
    # The first line encodes the preliminary epicenter.
    line = buf.readline()

    hypocenter_catalog = line[:5].strip().decode()

    origin_time = line[5:].strip().split()[:6]
    values = list(map(int, origin_time[:-1])) + \
        [float(origin_time[-1])]
    try:
        origin_time = UTCDateTime(*values)
    except (TypeError, ValueError):
        warnings.warn("Could not determine origin time from line: %s. Will "
                      "be set to zero." % line)
        origin_time = UTCDateTime(0)
    line = line[28:].split()
    latitude, longitude, depth, body_wave_mag, surface_wave_mag = \
        map(float, line[:5])

    # The rest encodes the centroid solution.
    event_name = buf.readline().strip().split()[-1].decode()

    preliminary_origin = Origin(
        resource_id=_get_resource_id(event_name, "origin", tag="prelim"),
        time=origin_time,
        longitude=longitude,
        latitude=latitude,
        # Depth is in meters.
        depth=depth * 1000.0,
        origin_type="hypocenter",
        region=_fe.get_region(longitude=longitude, latitude=latitude),
        evaluation_status="preliminary"
    )

    preliminary_bw_magnitude = Magnitude(
        resource_id=_get_resource_id(event_name, "magnitude", tag="prelim_bw"),
        mag=body_wave_mag, magnitude_type="Mb",
        evaluation_status="preliminary",
        origin_id=preliminary_origin.resource_id)

    preliminary_sw_magnitude = Magnitude(
        resource_id=_get_resource_id(event_name, "magnitude", tag="prelim_sw"),
        mag=surface_wave_mag, magnitude_type="MS",
        evaluation_status="preliminary",
        origin_id=preliminary_origin.resource_id)

    values = ["time_shift", "half_duration", "latitude", "longitude",
              "depth", "m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
    cmt_values = {_i: float(buf.readline().strip().split()[-1])
                  for _i in values}

    # Moment magnitude calculation in dyne * cm.
    m_0 = 1.0 / math.sqrt(2.0) * math.sqrt(
        cmt_values["m_rr"] ** 2 +
        cmt_values["m_tt"] ** 2 +
        cmt_values["m_pp"] ** 2 +
        2.0 * cmt_values["m_rt"] ** 2 +
        2.0 * cmt_values["m_rp"] ** 2 +
        2.0 * cmt_values["m_tp"] ** 2)
    m_w = 2.0 / 3.0 * (math.log10(m_0) - 16.1)

    # Convert to meters.
    cmt_values["depth"] *= 1000.0
    # Convert to Newton meter.
    values = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
    for value in values:
        cmt_values[value] /= 1E7

    cmt_origin = Origin(
        resource_id=_get_resource_id(event_name, "origin", tag="cmt"),
        time=origin_time + cmt_values["time_shift"],
        longitude=cmt_values["longitude"],
        latitude=cmt_values["latitude"],
        depth=cmt_values["depth"],
        origin_type="centroid",
        # Could rarely be different than the epicentral region.
        region=_fe.get_region(longitude=cmt_values["longitude"],
                              latitude=cmt_values["latitude"])
        # No evaluation status as it could be any of several and the file
        # format does not provide that information.
    )

    cmt_mag = Magnitude(
        resource_id=_get_resource_id(event_name, "magnitude", tag="mw"),
        # Round to 2 digits.
        mag=round(m_w, 2),
        magnitude_type="mw",
        origin_id=cmt_origin.resource_id
    )

    foc_mec = FocalMechanism(
        resource_id=_get_resource_id(event_name, "focal_mechanism"),
        # The preliminary origin most likely triggered the focal mechanism
        # determination.
        triggering_origin_id=preliminary_origin.resource_id
    )

    tensor = Tensor(
        m_rr=cmt_values["m_rr"],
        m_pp=cmt_values["m_pp"],
        m_tt=cmt_values["m_tt"],
        m_rt=cmt_values["m_rt"],
        m_rp=cmt_values["m_rp"],
        m_tp=cmt_values["m_tp"]
    )

    # Source time function is a triangle, according to the SPECFEM manual.
    stf = SourceTimeFunction(
        type="triangle",
        # The duration is twice the half duration.
        duration=2.0 * cmt_values["half_duration"]
    )

    mt = MomentTensor(
        resource_id=_get_resource_id(event_name, "moment_tensor"),
        derived_origin_id=cmt_origin.resource_id,
        moment_magnitude_id=cmt_mag.resource_id,
        # Convert to Nm.
        scalar_moment=m_0 / 1E7,
        tensor=tensor,
        source_time_function=stf
    )

    # Assemble everything.
    foc_mec.moment_tensor = mt

    ev = Event(resource_id=_get_resource_id(event_name, "event"),
               event_type="earthquake")
    ev.event_descriptions.append(EventDescription(text=event_name,
                                                  type="earthquake name"))
    ev.comments.append(Comment(
        text="Hypocenter catalog: %s" % hypocenter_catalog,
        force_resource_id=False))

    ev.origins.append(cmt_origin)
    ev.origins.append(preliminary_origin)
    ev.magnitudes.append(cmt_mag)
    ev.magnitudes.append(preliminary_bw_magnitude)
    ev.magnitudes.append(preliminary_sw_magnitude)
    ev.focal_mechanisms.append(foc_mec)

    # Set the preferred items.
    ev.preferred_origin_id = cmt_origin.resource_id.id
    ev.preferred_magnitude_id = cmt_mag.resource_id.id
    ev.preferred_focal_mechanism_id = foc_mec.resource_id.id

    ev.scope_resource_ids()

    return ev


def _write_cmtsolution(catalog, filename_or_buf, **kwargs):
    """
    Write an event to a file.

    :param catalog: The catalog to write. Can only contain one event.
    :type catalog: :class:`~obspy.core.event.Catalog`
    :param filename_or_buf: Filename or file-like object to write to.
    :type filename_or_buf: str, open file, or file-like object.
    """
    return _buffer_proxy(filename_or_buf, _internal_write_cmtsolution,
                         file_mode="wb", catalog=catalog, **kwargs)


def _internal_write_cmtsolution(buf, catalog, **kwargs):
    """
    Write events to a file.

    :param buf: File to write to.
    :type buf: Open file or file-like object.
    :param catalog: The catalog to write.
    :type catalog: :class:`~obspy.core.event.Catalog`
    """
    # Some sanity checks.
    if len(catalog) < 1:
        raise ValueError("Catalog must contain at least one event")
    for event in catalog:
        _internal_write_single_cmtsolution(buf, event)
        # Add an empty line between events.
        if len(catalog) > 1:
            buf.write(b"\n")


def _internal_write_single_cmtsolution(buf, event, **kwargs):
    """
    Write an event to a file.

    :param buf: File to write to.
    :type buf: Open file or file-like object.
    :param event: The event to write.
    :type event: :class:`~obspy.core.event.Event`
    """
    if not event.focal_mechanisms:
        raise ValueError("Event must contain a focal mechanism.")
    foc_mec = event.preferred_focal_mechanism() or event.focal_mechanisms[0]
    if not foc_mec.moment_tensor:
        raise ValueError("The preferred or first focal mechanism must "
                         "contain a moment tensor.")
    mt = foc_mec.moment_tensor
    if not mt.tensor:
        raise ValueError("The preferred or first focal mechanism must "
                         "contain a moment tensor element with an actual "
                         "tensor.")
    if not event.origins:
        raise ValueError("Event must have at least one origin.")
    if not event.magnitudes:
        raise ValueError("Event must have at least one magnitude.")

    # Attempt to get the body and surface wave magnitudes.
    mb_candidates = \
        [_i for _i in event.magnitudes if _i.magnitude_type == "Mb"]
    ms_candidates = \
        [_i for _i in event.magnitudes if _i.magnitude_type == "MS"]

    if not mb_candidates:
        warnings.warn("No body wave magnitude found. Will be replaced by the "
                      "first magnitude in the event object.")
        mb_mag = event.magnitudes[0]
    else:
        mb_mag = mb_candidates[0]
    if not ms_candidates:
        warnings.warn("No surface wave magnitude found. Will be replaced by "
                      "the first magnitude in the event object.")
        ms_mag = event.magnitudes[0]
    else:
        ms_mag = ms_candidates[0]

    # Now find the cmt origin. First attempt to get the derived origin of
    # the moment tensor,
    if mt.derived_origin_id:
        cmt_origin = mt.derived_origin_id.get_referred_object()
    # Otherwise try to find the first one that is CMT
    else:
        candidates = [_i for _i in event.origins
                      if _i.origin_type == "centroid"]
        if candidates:
            warnings.warn("No derived origin attached to the moment tensor. "
                          "Will instead use another centroid origin to be "
                          "written to the file.")
            cmt_origin = candidates[0]
        # Otherwise just take the preferred or first one.
        else:
            warnings.warn("Could not find a centroid origin. Will instead "
                          "assume that the preferred or first origin is the "
                          "centroid origin.")
            cmt_origin = event.preferred_origin() or event.origins[0]

    # Next step is to find a hypocentral origin.
    candidates = [_i for _i in event.origins
                  if _i.origin_type == "hypocenter"]
    if candidates:
        hypo_origin = candidates[0]
    # Otherwise get the first one that is not equal to the CMT origin.
    else:
        if len(event.origins) == 1:
            warnings.warn("Hypocentral origin will be identical to the "
                          "centroid one.")
            hypo_origin = event.origins[0]
        else:
            warnings.warn("No hypocentral origin could be found. Will choose "
                          "the first one that is not identical to the "
                          "centroid origin.")
            hypo_origin = [_i for _i in event.origins if _i != cmt_origin][0]

    # Try to find the half duration.
    if mt.source_time_function:
        if mt.source_time_function.duration:
            half_duration = mt.source_time_function.duration / 2.0
        else:
            warnings.warn("Source time function has no duration. The half "
                          "duration will be set to 1.0.")
            half_duration = 1.0
    else:
        warnings.warn("Moment tensor has no source time function. Half "
                      "duration will be set to 1.0.")
        half_duration = 1.0

    # Now attempt to retrieve the event name. Otherwise just get a random one.
    event_name = None
    if event.event_descriptions:
        candidates = [_i for _i in event.event_descriptions
                      if _i.type == "earthquake name"]
        if candidates:
            event_name = candidates[0].text
    if event_name is None:
        event_name = str(uuid.uuid4())[:6].upper()

    # Also attempt to retrieve the determination type.
    hypocenter_catalog = "PDE"
    if event.comments:
        candidates = [
            _i for _i in event.comments
            if _i.text.lower().strip().startswith("hypocenter catalog:")]
        if candidates:
            hypocenter_catalog = \
                candidates[0].text.strip().split(":")[-1].upper()

    template = (
        "{hypocenter_catalog:>4} {year:4d} {month:02d} {day:02d} {hour:02d} "
        "{minute:02d} {second:05.2f} "
        "{latitude:9.4f} {longitude:9.4f} {depth:5.1f} {mb:.1f} {ms:.1f} "
        "{region}\n"
        "event name:{event_name:>17}\n"
        "time shift:{time_shift:17.4f}\n"
        "half duration:{half_duration:14.4f}\n"
        "latitude:{cmt_latitude:19.4f}\n"
        "longitude:{cmt_longitude:18.4f}\n"
        "depth:{cmt_depth:22.4f}\n"
        "Mrr:{m_rr:24.6E}\n"
        "Mtt:{m_tt:24.6E}\n"
        "Mpp:{m_pp:24.6E}\n"
        "Mrt:{m_rt:24.6E}\n"
        "Mrp:{m_rp:24.6E}\n"
        "Mtp:{m_tp:24.6E}\n"
    )

    template = template.format(
        hypocenter_catalog=hypocenter_catalog,
        year=hypo_origin.time.year,
        month=hypo_origin.time.month,
        day=hypo_origin.time.day,
        hour=hypo_origin.time.hour,
        minute=hypo_origin.time.minute,
        second=float(hypo_origin.time.second) +
        hypo_origin.time.microsecond / 1E6,
        latitude=hypo_origin.latitude,
        longitude=hypo_origin.longitude,
        depth=hypo_origin.depth / 1000.0,
        mb=mb_mag.mag,
        ms=ms_mag.mag,
        region=_fe.get_region(longitude=hypo_origin.longitude,
                              latitude=hypo_origin.latitude),
        event_name=event_name,
        time_shift=cmt_origin.time - hypo_origin.time,
        half_duration=half_duration,
        cmt_latitude=cmt_origin.latitude,
        cmt_longitude=cmt_origin.longitude,
        cmt_depth=cmt_origin.depth / 1000.0,
        # Convert to dyne * cm.
        m_rr=mt.tensor.m_rr * 1E7,
        m_tt=mt.tensor.m_tt * 1E7,
        m_pp=mt.tensor.m_pp * 1E7,
        m_rt=mt.tensor.m_rt * 1E7,
        m_rp=mt.tensor.m_rp * 1E7,
        m_tp=mt.tensor.m_tp * 1E7
    )

    # Write to a buffer/file opened in binary mode.
    buf.write(template.encode())
