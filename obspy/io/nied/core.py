#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NIED moment tensor file format support for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import warnings
from obspy import UTCDateTime
import re
import uuid
from obspy.core.event import (Catalog, Comment, Event, EventDescription,
                              Origin, Magnitude, FocalMechanism, MomentTensor,
                              Tensor, SourceTimeFunction, NodalPlane,
                              NodalPlanes)

class NIEDException(Exception):
    pass


class GenSC3ID:
    """
    Generate an event ID following the SeisComP3 convention.
    """

    def __init__(self, numenc=6, sym="abcdefghijklmnopqrstuvwxyz"):
        self.sym = sym
        self.numsym = len(sym)
        self.numenc = numenc

    def get_id(self, dt):
        """
        >>> pid = GenSC3ID()
        >>> print pid.get_id(UTCDateTime(2015, 8, 18, 10, 55, 51, 367580))
        2015qffasl
        """
        x = (((((dt.julday - 1) * 24) + dt.hour) * 60 + dt.minute) * 60 + dt.second) * 1000 \
        + dt.microsecond / 1000
        dx = (((370 * 24) * 60) * 60) * 1000
        rng = 1
        tmp = rng
        for i in xrange(self.numenc):
            rng *= self.numsym
        w = int(dx / rng)
        if w == 0:
            w = 1

        if  dx >= rng:
            x = int(x / w)
        else:
            x = x * int(rng / dx)
        enc = ''
        for i in xrange(self.numenc):
            d = int(x / self.numsym)
            r = x % self.numsym
            enc += self.sym[r]
            x = d
        return '%d%s' % (dt.year, enc[::-1])


def _get_resource_id(cmtname, res_type, tag=None):
    """
    Helper function to create consistent resource ids.
    """
    res_id = "smi:local/nied/%s/%s" % (cmtname, res_type)
    if tag is not None:
        res_id += "#" + tag
    return res_id


def _is_nied_catalog(filename):
    """
    Test whether file is an NIED moment tensor catalog file by reading the
    header and the first data line. Reads at most 40 lines.
    """
    fh = open(filename)
    cnt = 0
    try:
        while True:
            line = fh.readline()
            if not line:
                return False
            if cnt >= 40:
                return False
            if line.find('Total Number') != -1:
                match = re.search(r'Total Number:\s+(\d+)\s+', line)
                if match:
                    nevents = int(match.group(1))
            if line.startswith('Origin Time'):
                if nevents > 0:
                    data = fh.readline()
                    a = data.split()
                    if len(a) != 21:
                        return False
                    return True
    except:
        return False
    else:
        return True


def _read_nied_catalog(filename):
    fh = open(filename)
    dataon = False
    headerlines = []
    pid = GenSC3ID()
    while True:
        line = fh.readline()
        if not dataon:
            headerlines.append(line)
        if not line:
            break
        if line.find('Total Number') != -1:
            match = re.search(r'Total Number:\s+(\d+)\s+', line)
            if match:
                nevents = int(match.group(1))
        if line.startswith('Origin Time'):
            dataon = True
            cat = Catalog(resource_id=_get_resource_id("catalog",
                                                       str(uuid.uuid4())))
            cat.description = headerlines[:-1]
            line = fh.readline()
        if dataon:
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
            nstat = int(a[20])

            event_name = pid.get_id(ot)
            e = Event(event_type="earthquake")
            e.resource_id = _get_resource_id(event_name, 'event')

            # Standard JMA solution
            o_jma = Origin(time=ot, latitude=lat, longitude=lon,
                           depth=depjma, depth_type="from location",
                           region=region)
            o_jma.resource_id = _get_resource_id(event_name,
                                                      'orgin', 'JMA')
            m_jma = Magnitude(mag=magjma, magnitude_type='ML',
                              origin_id=o_jma.resource_id)
            m_jma.resource_id = _get_resource_id(event_name,
                                                      'magnitude', 'JMA')
            # MT solution
            o_mt = Origin(time=ot, latitude=lat, longitude=lon,
                           depth=depmt, region=region,
                           depth_type="from moment tensor inversion")
            o_mt.resource_id = _get_resource_id(event_name,
                                                      'orgin', 'MT')
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

            tensor = Tensor(
                            m_rr=mxx,
                            m_tt=myy,
                            m_pp=mzz,
                            m_rt=mxy,
                            m_rp=mxz,
                            m_tp=myz)
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
            cat.append(e)
    # Consistency check
    if len(cat) != nevents:
        raise NIEDException('Parsing failed! Expected %d events but read %d.' \
                            % (nevents, len(cat)))
    return cat
