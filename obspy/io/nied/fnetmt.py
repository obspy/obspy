# -*- coding: utf-8 -*-
"""
F-net moment tensor file format support for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org) and Yannik Behr
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import re
import uuid

from obspy import UTCDateTime
from obspy.core.event import (Catalog, Comment, Event,
                              Origin, Magnitude, FocalMechanism, MomentTensor,
                              Tensor, NodalPlane, NodalPlanes)
from . import util


class FNETMTException(Exception):
    pass


def _get_resource_id(name, res_type, tag=None):
    """
    Helper function to create consistent resource ids.
    """
    res_id = "smi:local/fnetmt/%s/%s" % (name, res_type)
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


def _is_fnetmt_catalog(filename_or_buf):
    """
    Checks if the file is an F-net moment tensor file.

    :param filename_or_buf: File to test.
    :type filename_or_buf: str or file-like object.
    """
    try:
        return _buffer_proxy(filename_or_buf, __is_fnetmt_catalog,
                             reset_fp=True)
    # Happens for example when passing the data as a string which would be
    # interpreted as a filename.
    except (OSError):
        return False


def __is_fnetmt_catalog(buf):
    """
    Test whether file is an F-net moment tensor catalog file by reading the
    header and the first data line. Reads at most 40 lines.

    :param buf: File to read.
    :type buf: Open file or open file like object.
    """
    cnt = 0
    try:
        while True:
            line = buf.readline()
            if not line:
                return False
            line = line.decode()
            # read at most 40 lines
            if cnt > 40:
                return False
            if line.find('Total Number') != -1:
                match = re.search(r'Total Number:\s+(\d+)\s+', line)
                if match:
                    nevents = int(match.group(1))
            if line.startswith('Origin Time'):
                if nevents > 0:
                    data = buf.readline()
                    a = data.split()
                    if len(a) != 21:
                        return False
                    return True
            cnt += 1
    except:
        return False
    else:
        return True


def _read_fnetmt_catalog(filename_or_buf, **kwargs):
    """
    Reads an F-net moment tensor catalog file to a
    :class:`~obspy.core.event.Catalog` object.

    :param filename_or_buf: File to read.
    :type filename_or_buf: str or file-like object.
    """
    return _buffer_proxy(filename_or_buf, __read_fnetmt_catalog, **kwargs)


def __read_fnetmt_catalog(buf, **kwargs):
    """
    Reads an F-net moment tensor catalog file to a
    :class:`~obspy.core.event.Catalog` object.

    :param buf: File to read.
    :type buf: Open file or open file like object.
    """
    events = []
    cur_pos = buf.tell()

    # This also works with BytesIO and what not.
    buf.seek(0, 2)
    size = buf.tell()
    buf.seek(cur_pos, 0)

    # First read the headerlines containing the data request parameters
    headerlines = []
    while buf.tell() < size:
        line = buf.readline().decode()
        if line.find('Total Number') != -1:
            match = re.search(r'Total Number:\s+(\d+)\s+', line)
            if match:
                nevents = int(match.group(1))
        if line.startswith('Origin Time'):
            break
        headerlines.append(line)
    cur_pos = buf.tell()

    # Now read the catalog
    while buf.tell() < size:
        line = buf.readline().strip()
        # If there is something, jump back to the beginning of the line and
        # read the next event.
        if line:
            events.append(__read_single_fnetmt_entry(line.decode()))

    # Consistency check
    if len(events) != nevents:
        raise FNETMTException('Parsing failed! Expected %d events but read %d.'
                              % (nevents, len(events)))

    return Catalog(resource_id=_get_resource_id("catalog", str(uuid.uuid4())),
                   events=events, description=headerlines[:-1])


def __read_single_fnetmt_entry(line, **kwargs):
    """
    Reads a single F-net moment tensor solution to a
    :class:`~obspy.core.event.Event` object.

    :param line: String containing moment tensor information.
    :type line: str.
    """

    a = line.split()
    try:
        ot = UTCDateTime().strptime(a[0], '%Y/%m/%d,%H:%M:%S.%f')
    except ValueError:
        ot = UTCDateTime().strptime(a[0], '%Y/%m/%d,%H:%M:%S')
    lat, lon, depjma, magjma = map(float, a[1:5])
    depjma *= 1000
    region = a[5]
    strike = tuple(map(int, a[6].split(';')))
    dip = tuple(map(int, a[7].split(';')))
    rake = tuple(map(int, a[8].split(';')))
    mo = float(a[9])
    depmt = float(a[10]) * 1000
    magmt = float(a[11])
    var_red = float(a[12])
    mxx, mxy, mxz, myy, myz, mzz, unit = map(float, a[13:20])

    event_name = util.gen_sc3_id(ot)
    e = Event(event_type="earthquake")
    e.resource_id = _get_resource_id(event_name, 'event')

    # Standard JMA solution
    o_jma = Origin(time=ot, latitude=lat, longitude=lon,
                   depth=depjma, depth_type="from location",
                   region=region)
    o_jma.resource_id = _get_resource_id(event_name,
                                         'origin', 'JMA')
    m_jma = Magnitude(mag=magjma, magnitude_type='ML',
                      origin_id=o_jma.resource_id)
    m_jma.resource_id = _get_resource_id(event_name,
                                         'magnitude', 'JMA')
    # MT solution
    o_mt = Origin(time=ot, latitude=lat, longitude=lon,
                  depth=depmt, region=region,
                  depth_type="from moment tensor inversion")
    o_mt.resource_id = _get_resource_id(event_name,
                                        'origin', 'MT')
    m_mt = Magnitude(mag=magmt, magnitude_type='Mw',
                     origin_id=o_mt.resource_id)
    m_mt.resource_id = _get_resource_id(event_name,
                                        'magnitude', 'MT')
    foc_mec = FocalMechanism(triggering_origin_id=o_jma.resource_id)
    foc_mec.resource_id = _get_resource_id(event_name,
                                           "focal_mechanism")
    nod1 = NodalPlane(strike=strike[0], dip=dip[0], rake=rake[0])
    nod2 = NodalPlane(strike=strike[1], dip=dip[1], rake=rake[1])
    nod = NodalPlanes(nodal_plane_1=nod1, nodal_plane_2=nod2)
    foc_mec.nodal_planes = nod

    tensor = Tensor(m_rr=mxx, m_tt=myy, m_pp=mzz, m_rt=mxy, m_rp=mxz, m_tp=myz)
    cm = Comment(text="Basis system: North,East,Down (Jost and \
    Herrmann 1989")
    cm.resource_id = _get_resource_id(event_name, 'comment', 'mt')
    mt = MomentTensor(derived_origin_id=o_mt.resource_id,
                      moment_magnitude_id=m_mt.resource_id,
                      scalar_moment=mo, comments=[cm],
                      tensor=tensor, variance_reduction=var_red)
    mt.resource_id = _get_resource_id(event_name,
                                      'moment_tensor')
    foc_mec.moment_tensor = mt
    e.origins = [o_jma, o_mt]
    e.magnitudes = [m_jma, m_mt]
    e.focal_mechanisms = [foc_mec]
    return e
