#!/usr/bin/env python
"""
Functions for array analysis.

These routines are largely still around to ensure compatibility with
existing codes. Please try to use the new class based
:class:`~obspy.signals.array_analysis.seismic_array.SeismicArray` interface.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import math

import numpy as np

from obspy.core import Stream
from obspy.core.inventory import Inventory, Network, Station
from obspy.signal.headers import clibsignal
from obspy.signal.invsim import cosine_taper
from obspy.signal.util import next_pow_2, util_geo_km


def get_geometry(stream, coordsys='lonlat', return_center=False,
                 verbose=False):
    """
    Method to calculate the array geometry and the center coordinates in km

    :param stream: Stream object, the trace.stats dict like class must
        contain an :class:`~obspy.core.util.attribdict.AttribDict` with
        'latitude', 'longitude' (in degrees) and 'elevation' (in km), or 'x',
        'y', 'elevation' (in km) items/attributes. See param ``coordsys``
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :param return_center: Returns the center coordinates as extra tuple
    :return: Returns the geometry of the stations as 2d :class:`numpy.ndarray`
            The first dimension are the station indexes with the same order
            as the traces in the stream object. The second index are the
            values of [lat, lon, elev] in km
            last index contains center [lat, lon, elev] in degrees and km if
            return_center is true
    """
    nstat = len(stream)
    center_lat = 0.
    center_lon = 0.
    center_h = 0.
    geometry = np.empty((nstat, 3))

    if isinstance(stream, Stream):
        for i, tr in enumerate(stream):
            if coordsys == 'lonlat':
                geometry[i, 0] = tr.stats.coordinates.longitude
                geometry[i, 1] = tr.stats.coordinates.latitude
                geometry[i, 2] = tr.stats.coordinates.elevation
            elif coordsys == 'xy':
                geometry[i, 0] = tr.stats.coordinates.x
                geometry[i, 1] = tr.stats.coordinates.y
                geometry[i, 2] = tr.stats.coordinates.elevation
    elif isinstance(stream, np.ndarray):
        geometry = stream.copy()
    else:
        raise TypeError('only Stream or numpy.ndarray allowed')

    if verbose:
        print("coordsys = " + coordsys)

    if coordsys == 'lonlat':
        center_lon = geometry[:, 0].mean()
        center_lat = geometry[:, 1].mean()
        center_h = geometry[:, 2].mean()
        for i in np.arange(nstat):
            x, y = util_geo_km(center_lon, center_lat, geometry[i, 0],
                               geometry[i, 1])
            geometry[i, 0] = x
            geometry[i, 1] = y
            geometry[i, 2] -= center_h
    elif coordsys == 'xy':
        geometry[:, 0] -= geometry[:, 0].mean()
        geometry[:, 1] -= geometry[:, 1].mean()
        geometry[:, 2] -= geometry[:, 2].mean()
    else:
        raise ValueError("Coordsys must be one of 'lonlat', 'xy'")

    if return_center:
        return np.c_[geometry.T,
                     np.array((center_lon, center_lat, center_h))].T
    else:
        return geometry


def __geometry_to_inventory(geometry):
    """
    Internal helper routine to convert a local xyz geometry to an inventory
    object.
    """
    # Import here to avoid circular imports.
    from obspy.signal.array_analysis import SeismicArray

    # A bit of an ugly hack to reduce the code duplication and use the
    # routines in the new array class.
    stations = [
        Station(code="%i" % _i,
                # Manually inverted for the best constant that work with
                # ObsPy's internal conversions.
                latitude=geom[1] / 110.5748180,
                longitude=geom[0] / 111.319941,
                elevation=0) for _i, geom in enumerate(geometry)]
    network = Network(code="XX", stations=stations)
    inv = Inventory(networks=[network], source="me")
    return SeismicArray(name="", inventory=inv)


def get_timeshift(geometry, sll_x, sll_y, sl_s, grdpts_x, grdpts_y):
    """
    Returns timeshift table for given array geometry

    .. note::
        Legacy routine - please slowly transition towards using the
        :class:`~obspy.signal.array_analysis.seismic_array.SeismicArray`
        class.

    :param geometry: Nested list containing the arrays geometry, as
        returned by get_group_geometry
    :param sll_x: slowness x min (lower)
    :param sll_y: slowness y min (lower)
    :param sl_s: slowness step
    :param grdpts_x: number of grid points in x direction
    :param grdpts_x: number of grid points in y direction
    """
    sa = __geometry_to_inventory(geometry)
    return sa._get_timeshift(sllx=sll_x, slly=sll_y, sls=sl_s,
                             grdpts_x=grdpts_x, grdpts_y=grdpts_y,
                             latitude=0.0, longitude=0.0, absolute_height=0,
                             static3d=False)


def array_transff_wavenumber(coords, klim, kstep, coordsys='lonlat'):
    """
    Returns array transfer function as a function of wavenumber difference

    .. note::
        Legacy routine - please slowly transition towards using the
        :class:`~obspy.signal.array_analysis.seismic_array.SeismicArray`
        class.

    :type coords: numpy.ndarray
    :param coords: coordinates of stations in longitude and latitude in degrees
        elevation in km, or x, y, z in km
    :type coordsys: str
    :param coordsys: valid values: 'lonlat' and 'xy', choose which coordinates
        to use
    :param klim: either a float to use symmetric limits for wavenumber
        differences or the tuple (kxmin, kxmax, kymin, kymax)
    """
    geometry = get_geometry(coords, coordsys)
    sa = __geometry_to_inventory(geometry)
    return sa.array_transfer_function_wavenumber(klim=klim, kstep=kstep)


def array_transff_freqslowness(coords, slim, sstep, fmin, fmax, fstep,
                               coordsys='lonlat'):
    """
    Returns array transfer function as a function of slowness difference and
    frequency.

    .. note::
        Legacy routine - please slowly transition towards using the
        :class:`~obspy.signal.array_analysis.seismic_array.SeismicArray`
        class.

    :type coords: numpy.ndarray
    :param coords: coordinates of stations in longitude and latitude in degrees
        elevation in km, or x, y, z in km
    :type coordsys: str
    :param coordsys: valid values: 'lonlat' and 'xy', choose which coordinates
        to use
    :param slim: either a float to use symmetric limits for slowness
        differences or the tupel (sxmin, sxmax, symin, symax)
    :type fmin: float
    :param fmin: minimum frequency in signal
    :type fmax: float
    :param fmin: maximum frequency in signal
    :type fstep: float
    :param fmin: frequency sample distance
    """
    geometry = get_geometry(coords, coordsys)
    sa = __geometry_to_inventory(geometry)
    return sa.array_transfer_function_freqslowness(
        slim=slim, sstep=sstep, fmin=fmin, fmax=fmax, fstep=fstep)


def dump(pow_map, apow_map, i):
    """
    Example function to use with `store` kwarg in
    :func:`~obspy.signal.array_analysis.array_processing`.
    """
    np.savez('pow_map_%d.npz' % i, pow_map)
    np.savez('apow_map_%d.npz' % i, apow_map)


def array_processing(stream, win_len, win_frac, sll_x, slm_x, sll_y, slm_y,
                     sl_s, semb_thres, vel_thres, frqlow, frqhigh, stime,
                     etime, prewhiten, verbose=False, coordsys='lonlat',
                     timestamp='mlabday', method=0, store=None):
    """
    Method for Seismic-Array-Beamforming/FK-Analysis/Capon

    :param stream: Stream object, the trace.stats dict like class must
        contain an :class:`~obspy.core.util.attribdict.AttribDict` with
        'latitude', 'longitude' (in degrees) and 'elevation' (in km), or 'x',
        'y', 'elevation' (in km) items/attributes. See param ``coordsys``.
    :type win_len: float
    :param win_len: Sliding window length in seconds
    :type win_frac: float
    :param win_frac: Fraction of sliding window to use for step
    :type sll_x: float
    :param sll_x: slowness x min (lower)
    :type slm_x: float
    :param slm_x: slowness x max
    :type sll_y: float
    :param sll_y: slowness y min (lower)
    :type slm_y: float
    :param slm_y: slowness y max
    :type sl_s: float
    :param sl_s: slowness step
    :type semb_thres: float
    :param semb_thres: Threshold for semblance
    :type vel_thres: float
    :param vel_thres: Threshold for velocity
    :type frqlow: float
    :param frqlow: lower frequency for fk/capon
    :type frqhigh: float
    :param frqhigh: higher frequency for fk/capon
    :type stime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param stime: Start time of interest
    :type etime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param etime: End time of interest
    :type prewhiten: int
    :param prewhiten: Do prewhitening, values: 1 or 0
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :type timestamp: str
    :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
        the timestamp in seconds since 1970-01-01T00:00:00, 'mlabday'
        returns the timestamp in days (decimals represent hours, minutes
        and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
        date plotting (see e.g. matplotlib's num2date)
    :type method: int
    :param method: the method to use 0 == bf, 1 == capon
    :type store: function
    :param store: A custom function which gets called on each iteration. It is
        called with the relative power map and the time offset as first and
        second arguments and the iteration number as third argument. Useful for
        storing or plotting the map for each iteration. For this purpose the
        dump function of this module can be used.
    :return: :class:`numpy.ndarray` of timestamp, relative relpow, absolute
        relpow, backazimuth, slowness
    """
    # Import here to avoid circular imports.
    from obspy.signal.array_analysis import seismic_array

    res = []
    eotr = True

    # check that sampling rates do not vary
    fs = stream[0].stats.sampling_rate
    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = 'in sonic sampling rates of traces in stream are not equal'
        raise ValueError(msg)

    grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
    grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

    geometry = get_geometry(stream, coordsys=coordsys, verbose=verbose)

    if verbose:
        print("geometry:")
        print(geometry)
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    time_shift_table = get_timeshift(geometry, sll_x, sll_y,
                                     sl_s, grdpts_x, grdpts_y)
    # offset of arrays
    spoint, _epoint = seismic_array._get_stream_offsets(stream, stime, etime)
    #
    # loop with a sliding window over the dat trace array and apply bbfk
    #
    nstat = len(stream)
    fs = stream[0].stats.sampling_rate
    nsamp = int(win_len * fs)
    nstep = int(nsamp * win_frac)

    # generate plan for rfftr
    nfft = next_pow_2(nsamp)
    deltaf = fs / float(nfft)
    nlow = int(frqlow / float(deltaf) + 0.5)
    nhigh = int(frqhigh / float(deltaf) + 0.5)
    nlow = max(1, nlow)  # avoid using the offset
    nhigh = min(nfft // 2 - 1, nhigh)  # avoid using nyquist
    nf = nhigh - nlow + 1  # include upper and lower frequency
    # to speed up the routine a bit we estimate all steering vectors in advance
    steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype=np.complex128)
    clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow,
                         deltaf, time_shift_table, steer)
    _r = np.empty((nf, nstat, nstat), dtype=np.complex128)
    ft = np.empty((nstat, nf), dtype=np.complex128)
    newstart = stime
    # 0.22 matches 0.2 of historical C bbfk.c
    tap = cosine_taper(nsamp, p=0.22)
    offset = 0
    relpow_map = np.empty((grdpts_x, grdpts_y), dtype=np.float64)
    abspow_map = np.empty((grdpts_x, grdpts_y), dtype=np.float64)
    while eotr:
        try:
            for i, tr in enumerate(stream):
                dat = tr.data[spoint[i] + offset:
                              spoint[i] + offset + nsamp]
                dat = (dat - dat.mean()) * tap
                ft[i, :] = np.fft.rfft(dat, nfft)[nlow:nlow + nf]
        except IndexError:
            break
        ft = np.ascontiguousarray(ft, np.complex128)
        relpow_map.fill(0.)
        abspow_map.fill(0.)
        # computing the covariances of the signal at different receivers
        dpow = 0.
        for i in range(nstat):
            for j in range(i, nstat):
                _r[:, i, j] = ft[i, :] * ft[j, :].conj()
                if method == 1:
                    _r[:, i, j] /= np.abs(_r[:, i, j].sum())
                if i != j:
                    _r[:, j, i] = _r[:, i, j].conjugate()
                else:
                    dpow += np.abs(_r[:, i, j].sum())
        dpow *= nstat
        if method == 1:
            # P(f) = 1/(e.H R(f)^-1 e)
            for n in range(nf):
                _r[n, :, :] = np.linalg.pinv(_r[n, :, :], rcond=1e-6)

        errcode = clibsignal.generalizedBeamformer(
            relpow_map, abspow_map, steer, _r, nstat, prewhiten,
            grdpts_x, grdpts_y, nf, dpow, method)
        if errcode != 0:
            msg = 'generalizedBeamforming exited with error %d'
            raise Exception(msg % errcode)
        ix, iy = np.unravel_index(relpow_map.argmax(), relpow_map.shape)
        relpow, abspow = relpow_map[ix, iy], abspow_map[ix, iy]
        if store is not None:
            store(relpow_map, abspow_map, offset)
        # here we compute baz, slow
        slow_x = sll_x + ix * sl_s
        slow_y = sll_y + iy * sl_s

        slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
        if slow < 1e-8:
            slow = 1e-8
        azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
        baz = azimut % -360 + 180
        if relpow > semb_thres and 1. / slow > vel_thres:
            res.append(np.array([newstart.timestamp, relpow, abspow, baz,
                                 slow]))
            if verbose:
                print(newstart, (newstart + (nsamp / fs)), res[-1][1:])
        if (newstart + (nsamp + nstep) / fs) > etime:
            eotr = False
        offset += nstep

        newstart += nstep / fs
    res = np.array(res)
    if timestamp == 'julsec':
        pass
    elif timestamp == 'mlabday':
        # 719163 == days between 1970 and 0001 + 1
        res[:, 0] = res[:, 0] / (24. * 3600) + 719163
    else:
        msg = "Option timestamp must be one of 'julsec', or 'mlabday'"
        raise ValueError(msg)
    return np.array(res)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
