# -*- coding: utf-8 -*-
"""
SCARDEC file format support for ObsPy.
These files contain source solutions calculated at IPGP using the
SCARDEC method [Vallee2011]_. These solutions contain a focal mechanism
and a source time function in form of a regularly sampled moment rate.
An equivalent moment tensor is automatically calculated. Writing of SCARDEC
files is only possible, if the event contains a moment rate function.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import uuid
import warnings

import numpy as np

from obspy import UTCDateTime
from obspy.core.event import (Catalog, Comment, Event, EventDescription,
                              Origin, Magnitude, FocalMechanism, MomentTensor,
                              Tensor, SourceTimeFunction, NodalPlane,
                              NodalPlanes)
from obspy.geodetics import FlinnEngdahl


_fe = FlinnEngdahl()


def _get_resource_id(scardecname, res_type, tag=None):
    """
    Helper function to create consistent resource ids.
    """
    res_id = "smi:local/scardec/%s/%s" % (scardecname, res_type)
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


def _is_scardec(filename_or_buf):
    """
    Checks if the file is a SCARDEC file.

    :param filename_or_buf: File to test.
    :type filename_or_buf: str or file-like object.
    """
    try:
        return _buffer_proxy(filename_or_buf, _internal_is_scardec,
                             reset_fp=True)
    # Happens for example when passing the data as a string which would be
    # interpreted as a filename.
    except OSError:
        return False


def _internal_is_scardec(buf):
    """
    Checks if the file is a SCARDEC file.

    :param buf: File to check.
    :type buf: Open file or open file like object.
    """
    # The file format is so simple. Just attempt to read the first event. If
    # it passes it will be read again but that has really no
    # significant performance impact.
    try:
        _internal_read_single_scardec(buf)
        return True
    except Exception:
        return False


def _read_scardec(filename_or_buf, **kwargs):
    """
    Reads a SCARDEC file to a :class:`~obspy.core.event.Catalog` object.

    :param filename_or_buf: File to read.
    :type filename_or_buf: str or file-like object.
    """
    return _buffer_proxy(filename_or_buf, _internal_read_scardec, **kwargs)


def _internal_read_scardec(buf, **kwargs):  # @UnusedVariable
    """
    Reads a SCARDEC file to a :class:`~obspy.core.event.Catalog` object.

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
            events.append(_internal_read_single_scardec(buf))
        cur_pos = buf.tell()

    return Catalog(resource_id=_get_resource_id("catalog", str(uuid.uuid4())),
                   events=events)


def _internal_read_single_scardec(buf):
    """
    Reads a single SCARDEC file to a :class:`~obspy.core.event.Catalog`
    object.

    :param buf: File to read.
    :type buf: Open file or open file like object.
    """
    # The first line encodes the origin time and epicenter
    line = buf.readline()

    origin_time = line.strip().split()[:6]
    values = list(map(int, origin_time[:-1])) + \
        [float(origin_time[-1])]
    try:
        origin_time = UTCDateTime(*values)
    except (TypeError, ValueError):
        warnings.warn("Could not determine origin time from line: %s. Will "
                      "be set to zero." % line)
        origin_time = UTCDateTime(0)
    line = line.split()[6:]
    latitude, longitude = map(float, line[:2])

    # The second line encodes depth and the two focal mechanisms
    line = buf.readline()
    line = line.split()

    # First three values are depth, scalar moment (in Nm) and moment magnitude
    depth, scalar_moment, moment_mag = map(float, line[0:3])

    # depth is in km in SCARDEC files
    depth *= 1e3

    # Next six values are strike, dip, rake for both planes
    strike1, dip1, rake1 = map(float, line[3:6])
    strike2, dip2, rake2 = map(float, line[6:9])

    # The rest of the file is the moment rate function
    # In each line: time (sec), moment rate (Nm/sec)
    stf_time = []
    stf_mr = []
    for line in buf:
        stf_time.append(float(line.split()[0]))
        stf_mr.append(float(line.split()[1]))

    # Normalize the source time function
    stf_mr = np.array(stf_mr)
    stf_mr /= scalar_moment

    # Calculate the time step
    dt = np.mean(np.diff(stf_time))

    # Calculate the stf offset (time of first sample wrt to origin time)
    offset = stf_time[0]

    # event name is set to generic value for now
    event_name = 'SCARDEC_event'

    cmt_origin = Origin(
        resource_id=_get_resource_id(event_name, "origin", tag="cmt"),
        time=origin_time,
        longitude=longitude,
        latitude=latitude,
        depth=depth,
        origin_type="centroid",
        region=_fe.get_region(longitude=longitude,
                              latitude=latitude)
    )

    cmt_mag = Magnitude(
        resource_id=_get_resource_id(event_name, "magnitude", tag="mw"),
        mag=moment_mag,
        magnitude_type="mw",
        origin_id=cmt_origin.resource_id
    )

    nod1 = NodalPlane(strike=strike1, dip=dip1, rake=rake1)
    nod2 = NodalPlane(strike=strike2, dip=dip2, rake=rake2)
    nod = NodalPlanes(nodal_plane_1=nod1, nodal_plane_2=nod2)

    foc_mec = FocalMechanism(
        resource_id=_get_resource_id(event_name, "focal_mechanism"),
        nodal_planes=nod
    )

    dip1 *= np.pi / 180.
    rake1 *= np.pi / 180.
    strike1 *= np.pi / 180.

    mxx = - scalar_moment * ((np.sin(dip1) * np.cos(rake1) *
                              np.sin(2 * strike1)) +
                             (np.sin(2 * dip1) * np.sin(rake1) *
                              np.sin(2 * strike1)))
    mxy = scalar_moment * ((np.sin(dip1) * np.cos(rake1) *
                            np.cos(2 * strike1)) +
                           (np.sin(2 * dip1) * np.sin(rake1) *
                            np.sin(2 * strike1) * 0.5))
    myy = scalar_moment * ((np.sin(dip1) * np.cos(rake1) *
                            np.sin(2 * strike1)) -
                           (np.sin(2 * dip1) * np.sin(rake1) *
                            np.cos(2 * strike1)))
    mxz = - scalar_moment * ((np.cos(dip1) * np.cos(rake1) *
                              np.cos(strike1)) +
                             (np.cos(2 * dip1) * np.sin(rake1) *
                              np.sin(strike1)))
    myz = - scalar_moment * ((np.cos(dip1) * np.cos(rake1) *
                             np.sin(strike1)) -
                             (np.cos(2 * dip1) * np.sin(rake1) *
                              np.cos(strike1)))
    mzz = scalar_moment * (np.sin(2 * dip1) * np.sin(rake1))

    tensor = Tensor(m_rr=mxx, m_tt=myy, m_pp=mzz, m_rt=mxy, m_rp=mxz, m_tp=myz)

    cm = [Comment(text="Basis system: North,East,Down \
                        (Jost and Herrmann 1989)")]
    cm[0].resource_id = _get_resource_id(event_name, 'comment', 'mt')
    cm.append(Comment(text="MT derived from focal mechanism, therefore \
                            constrained to pure double couple.",
                      force_resource_id=False))

    # Write moment rate function
    extra = {'moment_rate': {'value': stf_mr,
                             'namespace': r"http://test.org/xmlns/0.1"},
             'dt': {'value': dt,
                    'namespace': r"http://test.org/xmlns/0.1"},
             'offset': {'value': offset,
                        'namespace': r"http://test.org/xmlns/0.1"}
             }

    # Source time function
    stf = SourceTimeFunction(type="unknown")
    stf.extra = extra

    mt = MomentTensor(
        resource_id=_get_resource_id(event_name, "moment_tensor"),
        derived_origin_id=cmt_origin.resource_id,
        moment_magnitude_id=cmt_mag.resource_id,
        scalar_moment=scalar_moment,
        tensor=tensor,
        source_time_function=stf,
        comments=cm
    )

    # Assemble everything.
    foc_mec.moment_tensor = mt

    ev = Event(resource_id=_get_resource_id(event_name, "event"),
               event_type="earthquake")
    ev.event_descriptions.append(EventDescription(text=event_name,
                                                  type="earthquake name"))
    ev.comments.append(Comment(
        text="Hypocenter catalog: SCARDEC",
        force_resource_id=False))

    ev.origins.append(cmt_origin)
    ev.magnitudes.append(cmt_mag)
    ev.focal_mechanisms.append(foc_mec)

    # Set the preferred items.
    ev.preferred_origin_id = cmt_origin.resource_id.id
    ev.preferred_magnitude_id = cmt_mag.resource_id.id
    ev.preferred_focal_mechanism_id = foc_mec.resource_id.id

    ev.scope_resource_ids()

    return ev


def _write_scardec(catalog, filename_or_buf, **kwargs):
    """
    Write an event to a file.

    :param catalog: The catalog to write. Can only contain one event.
    :type catalog: :class:`~obspy.core.event.Catalog`
    :param filename_or_buf: Filename or file-like object to write to.
    :type filename_or_buf: str, open file, or file-like object.
    """
    return _buffer_proxy(filename_or_buf, _internal_write_scardec,
                         file_mode="wb", catalog=catalog, **kwargs)


def _internal_write_scardec(buf, catalog, **kwargs):  # @UnusedVariable
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
        _internal_write_single_scardec(buf, event)
        # Add an empty line between events.
        if len(catalog) > 1:
            buf.write(b"\n")


def _internal_write_single_scardec(buf, event, **kwargs):  # @UnusedVariable
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
    if not event.origins:
        raise ValueError("Event must have at least one origin.")
    if not event.magnitudes:
        raise ValueError("Event must have at least one magnitude.")

    # Attempt to get a moment magnitude
    mw_candidates = \
        [_i for _i in event.magnitudes if _i.magnitude_type == "Mw"]

    if not mw_candidates:
        warnings.warn("No moment wave magnitude found. Will be replaced by the"
                      " first magnitude in the event object.")
        mw_mag = event.magnitudes[0]
    else:
        mw_mag = mw_candidates[0]

    # Now find the cmt origin. Try to find the first one that is CMT
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

    if "extra" not in foc_mec.moment_tensor.source_time_function:
        raise ValueError('Event moment tensor must contain a source time \
                          function to be written in SCARDEC format')

    # Now attempt to retrieve the event name. Otherwise just get a random one.
    event_name = None
    if event.event_descriptions:
        candidates = [_i for _i in event.event_descriptions
                      if _i.type == "earthquake name"]
        if candidates:
            event_name = candidates[0].text
    if event_name is None:
        event_name = str(uuid.uuid4())[:6].upper()

    template = (
        "{year:4d} {month:02d} {day:02d} {hour:02d} "
        "{minute:02d} {second:04.1f} "
        "{latitude:9.4f} {longitude:9.4f}\n"
        "{depth:5.1f} {scalmom:9.3E} {mw:5.3f}"
        "{strike1:4d} {dip1:4d} {rake1:4d}"
        "{strike2:4d} {dip2:4d} {rake2:4d}\n"
    )

    np1 = foc_mec.nodal_planes.nodal_plane_1
    np2 = foc_mec.nodal_planes.nodal_plane_2

    template = template.format(
        year=cmt_origin.time.year,
        month=cmt_origin.time.month,
        day=cmt_origin.time.day,
        hour=cmt_origin.time.hour,
        minute=cmt_origin.time.minute,
        second=float(cmt_origin.time.second) +
        cmt_origin.time.microsecond / 1E6,
        latitude=cmt_origin.latitude,
        longitude=cmt_origin.longitude,
        depth=cmt_origin.depth / 1000.0,
        scalmom=foc_mec.moment_tensor.scalar_moment,
        mw=mw_mag.mag,
        strike1=int(np1.strike),
        dip1=int(np1.dip),
        rake1=int(np1.rake),
        strike2=int(np2.strike),
        dip2=int(np2.dip),
        rake2=int(np2.rake)
    )

    buf.write(template.encode('ascii', 'strict'))

    # Write to a buffer/file opened in binary mode.

    stf = foc_mec.moment_tensor.source_time_function.extra
    t_offset = stf['offset']['value']
    scalmom = foc_mec.moment_tensor.scalar_moment

    nsamples = len(stf['moment_rate']['value'])

    times = np.arange(0, nsamples) * stf['dt']['value'] + t_offset
    samples = stf['moment_rate']['value'] * scalmom

    np.savetxt(buf, np.asarray([times, samples]).T,
               fmt=' %16.9E %16.9E'.encode('ascii', 'strict'))
