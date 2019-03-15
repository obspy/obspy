# -*- coding: utf-8 -*-
"""
Python interface to the Seismic Analysis Code (SAC) file format.

:copyright:
    The Los Alamos National Security, LLC, Yannik Behr, C. J. Ammon,
    C. Satriano, L. Krischer, and J. MacCarthy
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)


The SACTrace object maintains consistency between SAC headers and manages
header values in a user-friendly way. This includes some value-checking, native
Python logicals (True, False) and nulls (None) instead of SAC's 0, 1, or
-12345...

SAC headers are implemented as properties, with appropriate getters and
setters.


Features
--------

1. **Read and write SAC binary or ASCII**

   -  autodetect or specify expected byteorder
   -  optional file size checking and/or header consistency checks
   -  header-only reading and writing
   -  "overwrite OK" checking ('lovrok' header)

2. **Convenient access and manipulation of relative and absolute time
   headers**
3. **User-friendly header printing/viewing**
4. **Fast access to header values from attributes**

   -  With type checking, null handling, and enumerated value checking

5. **Convert to/from ObsPy Traces**

   -  Conversion from ObsPy Trace to SAC trace retains detected previous
      SAC header values.
   -  Conversion to ObsPy Trace retains the *complete* SAC header.


Usage examples
--------------

Read/write SAC files
~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # read from a binary file
    sac = SACTrace.read(filename)

    # read header only
    sac = SACTrace.read(filename, headonly=True)

    # write header-only, file must exist
    sac.write(filename, headonly=True)

    # read from an ASCII file
    sac = SACTrace.read(filename, ascii=True)

    # write a binary SAC file for a Sun machine
    sac.write(filename, byteorder='big')

Build a SACTrace from a header dictionary and data array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Example

>>> header = {'kstnm': 'ANMO', 'kcmpnm': 'BHZ', 'stla': 40.5, 'stlo': -108.23,
...           'evla': -15.123, 'evlo': 123, 'evdp': 50, 'nzyear': 2012,
...           'nzjday': 123, 'nzhour': 13, 'nzmin': 43, 'nzsec': 17,
...           'nzmsec': 100, 'delta': 1.0/40}
>>> sac = SACTrace(data=np.random.random(100), **header)
>>> print(sac)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
Reference Time = 05/02/2012 (123) 13:43:17.100000
   iztype IB: begin time
b          = 0.0
delta      = 0.0250000...
e          = 2.4750000...
evdp       = 50.0
evla       = -15.123000...
evlo       = 123.0
iftype     = itime
internal0  = 2.0
iztype     = ib
kcmpnm     = BHZ
kstnm      = ANMO
lcalda     = False
leven      = True
lovrok     = True
lpspol     = True
npts       = 100
nvhdr      = 6
nzhour     = 13
nzjday     = 123
nzmin      = 43
nzmsec     = 100
nzsec      = 17
nzyear     = 2012
stla       = 40.5
stlo       = -108.23000...

Reference-time and relative time headers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Example

>>> sac = SACTrace(nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0,
...                nzmsec=0, t1=23.5, data=np.arange(100))
>>> print(sac.reftime)
2000-01-01T00:00:00.000000Z
>>> sac.b, sac.e, sac.t1
(0.0, 99.0, 23.5)

Move reference time by relative seconds, relative time headers are
preserved.

.. rubric:: Example

>>> sac = SACTrace(nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0,
...                nzmsec=0, t1=23.5, data=np.arange(100))
>>> sac.reftime -= 2.5
>>> sac.b, sac.e, sac.t1
(2.5, 101.5, 26.0)

Set reference time to new absolute time, relative time headers are
preserved.

.. rubric:: Example

>>> sac = SACTrace(nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0,
...                nzmsec=0, t1=23.5, data=np.arange(100))
>>> # set the reftime two minutes later
>>> sac.reftime = UTCDateTime(2000, 1, 1, 0, 2, 0, 0)
>>> sac.b, sac.e, sac.t1
(-120.0, -21.0, -96.5)

Quick header viewing
~~~~~~~~~~~~~~~~~~~~

Print non-null header values.

.. rubric:: Example

>>> sac = SACTrace()
>>> print(sac)  # doctest: +NORMALIZE_WHITESPACE
Reference Time = 01/01/1970 (001) 00:00:00.000000
   iztype IB: begin time
b          = 0.0
delta      = 1.0
e          = 0.0
iftype     = itime
internal0  = 2.0
iztype     = ib
lcalda     = False
leven      = True
lovrok     = True
lpspol     = True
npts       = 0
nvhdr      = 6
nzhour     = 0
nzjday     = 1
nzmin      = 0
nzmsec     = 0
nzsec      = 0
nzyear     = 1970

Print relative time header values.

.. rubric:: Example

>>> sac = SACTrace()
>>> sac.lh('picks')  # doctest: +NORMALIZE_WHITESPACE
Reference Time = 01/01/1970 (001) 00:00:00.000000
   iztype IB: begin time
a          = None
b          = 0.0
e          = 0.0
f          = None
o          = None
t0         = None
t1         = None
t2         = None
t3         = None
t4         = None
t5         = None
t6         = None
t7         = None
t8         = None
t9         = None

Header values as attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Great for interactive use, with (ipython) tab-completion...

.. code:: python

    sac.<tab>

::

    sac.a                 sac.kevnm             sac.nzsec
    sac.az                sac.kf                sac.nzyear
    sac.b                 sac.khole             sac.o
    sac.baz               sac.kinst             sac.odelta
    sac.byteorder         sac.knetwk            sac.read
    sac.cmpaz             sac.ko                sac.reftime
    sac.cmpinc            sac.kstnm             sac.scale
    sac.copy              sac.kt0               sac.stdp
    sac.data              sac.kt1               sac.stel
    sac.delta             sac.kt2               sac.stla
    sac.depmax            sac.kt3               sac.stlo
    sac.depmen            sac.kt4               sac.t0
    sac.depmin            sac.kt5               sac.t1
    sac.dist              sac.kt6               sac.t2
    sac.e                 sac.kt7               sac.t3
    sac.evdp              sac.kt8               sac.t4
    sac.evla              sac.kt9               sac.t5
    sac.evlo              sac.kuser0            sac.t6
    sac.f                 sac.kuser1            sac.t7
    sac.from_obspy_trace  sac.kuser2            sac.t8
    sac.gcarc             sac.lcalda            sac.t9
    sac.idep              sac.leven             sac.to_obspy_trace
    sac.ievreg            sac.lh                sac.unused23
    sac.ievtyp            sac.listhdr           sac.user0
    sac.iftype            sac.lovrok            sac.user1
    sac.iinst             sac.lpspol            sac.user2
    sac.imagsrc           sac.mag               sac.user3
    sac.imagtyp           sac.nevid             sac.user4
    sac.internal0         sac.norid             sac.user5
    sac.iqual             sac.npts              sac.user6
    sac.istreg            sac.nvhdr             sac.user7
    sac.isynth            sac.nwfid             sac.user8
    sac.iztype            sac.nzhour            sac.user9
    sac.ka                sac.nzjday            sac.validate
    sac.kcmpnm            sac.nzmin             sac.write
    sac.kdatrd            sac.nzmsec

...and documentation (in IPython)!

.. code:: python

    sac.iztype?

::

    Type:        property
    String form: <property object at 0x106404940>
    Docstring:
    I    Reference time equivalence:

    * IUNKN (5): Unknown
    * IB (9): Begin time
    * IDAY (10): Midnight of reference GMT day
    * IO (11): Event origin time
    * IA (12): First arrival time
    * ITn (13-22): User defined time pick n, n=0,9

Convert to/from ObsPy Traces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Example

>>> from obspy import read
>>> tr = read()[0]
>>> print(tr.stats)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
         network: BW
         station: RJOB
        location:
         channel: EHZ
       starttime: 2009-08-24T00:20:03.000000Z
         endtime: 2009-08-24T00:20:32.990000Z
   sampling_rate: 100.0
           delta: 0.01
            npts: 3000
           calib: 1.0
    back_azimuth: 100.0
     inclination: 30.0
        response: Channel Response
            ...


>>> sac = SACTrace.from_obspy_trace(tr)
>>> print(sac)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
Reference Time = 08/24/2009 (236) 00:20:03.000000
   iztype IB: begin time
b          = 0.0
delta      = 0.009999999...
e          = 29.989999...
iftype     = itime
iztype     = ib
kcmpnm     = EHZ
knetwk     = BW
kstnm      = RJOB
lcalda     = False
leven      = True
lovrok     = True
lpspol     = True
npts       = 3000
nvhdr      = 6
nzhour     = 0
nzjday     = 236
nzmin      = 20
nzmsec     = 0
nzsec      = 3
nzyear     = 2009
scale      = 1.0

>>> tr2 = sac.to_obspy_trace()
>>> print(tr2.stats)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
         network: BW
         station: RJOB
        location:
         channel: EHZ
       starttime: 2009-08-24T00:20:03.000000Z
         endtime: 2009-08-24T00:20:32.990000Z
   sampling_rate: 100.0
           delta: 0.01
            npts: 3000
           calib: 1.0
             sac: AttribDict(...)

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import sys
import warnings
from copy import deepcopy
from itertools import chain

import numpy as np
from obspy import Trace, UTCDateTime
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees

from . import header as HD  # noqa
from .util import SacError, SacHeaderError
from . import util as _ut
from . import arrayio as _io


# ------------- HEADER DESCRIPTORS --------------------------------------------
#
# A descriptor is a class that manages an object attribute, using the
# descriptor protocol.  A single instance of a descriptor class (FloatHeader,
# for example) will exist for both the host class and all instances of that
# host class (i.e. SACTrace and all its instances).  As a result, we must
# implement logic that can tell if the methods are being called on the host
# class or on an instance.  This looks like "if instance is None" on methods.
#
# See:
# https://docs.python.org/3.5/howto/descriptor.html
# https://nbviewer.jupyter.org/urls/gist.github.com/ChrisBeaumont/
#   5758381/raw/descriptor_writeup.ipynb

class SACHeader(object):
    def __init__(self, name):
        try:
            self.__doc__ = HD.DOC[name]
        except KeyError:
            # header doesn't have a docstring entry in HD.DOC
            pass
        self.name = name


class FloatHeader(SACHeader):
    def __get__(self, instance, instance_type):
        if instance is None:
            # a FloatHeader on the owner class was requested.
            # return the descriptor itself.
            value = self
        else:
            # a FloatHeader on an instance was requested.
            # return the descriptor value.
            value = float(instance._hf[HD.FLOATHDRS.index(self.name)])
            if value == HD.FNULL:
                value = None
        return value

    def __set__(self, instance, value):
        if value is None:
            value = HD.FNULL
        instance._hf[HD.FLOATHDRS.index(self.name)] = value


# Descriptor for setting relative time headers with either a relative
# time float or an absolute UTCDateTime
# used for: b, o, a, f, t0-t9
class RelativeTimeHeader(FloatHeader):
    def __set__(self, instance, value):
        """
        Intercept the set value to make sure it is an offset from the SAC
        reference time.

        """
        if isinstance(value, UTCDateTime):
            offset = value - instance.reftime
        else:
            offset = value
        # reuse the normal floatheader setter.
        super(RelativeTimeHeader, self).__set__(instance, offset)


# Factory function for setting geographic header values
#   (evlo, evla, stalo, stalat)
# that will check lcalda and calculate and set dist, az, baz, gcarc
class GeographicHeader(FloatHeader):
    def __set__(self, instance, value):
        super(GeographicHeader, self).__set__(instance, value)
        if instance.lcalda:
            try:
                instance._set_distances()
            except SacHeaderError:
                pass


class IntHeader(SACHeader):
    def __get__(self, instance, instance_type):
        if instance is None:
            value = self
        else:
            value = int(instance._hi[HD.INTHDRS.index(self.name)])
            if value == HD.INULL:
                value = None
        return value

    def __set__(self, instance, value):
        if value is None:
            value = HD.INULL
        if value % 1:
            warnings.warn("Non-integers may be truncated. ({}: {})".format(
                self.name, value))
        instance._hi[HD.INTHDRS.index(self.name)] = value


class BoolHeader(IntHeader):
    def __get__(self, instance, instance_type):
        # value can be an int or None
        value = super(BoolHeader, self).__get__(instance, instance_type)
        return bool(value) if value in (0, 1) else value

    def __set__(self, instance, value):
        if value not in (True, False, 1, 0):
            msg = "Logical header values must be {True, False, 1, 0}"
            raise ValueError(msg)
        # booleans are subclasses of integers.  They will be set (cast)
        # directly into an integer array as 0 or 1.
        super(BoolHeader, self).__set__(instance, value)
        if self.name == 'lcalda':
            if value:
                try:
                    instance._set_distances()
                except SacHeaderError:
                    pass


class EnumHeader(IntHeader):
    def __get__(self, instance, instance_type):
        value = super(EnumHeader, self).__get__(instance, instance_type)
        # value is int or None
        if value is None:
            name = None
        elif _ut.is_valid_enum_int(self.name, value):
            name = HD.ENUM_NAMES[value]
        else:
            msg = """Unrecognized enumerated value {} for header "{}".
                     See .header for allowed values.""".format(value,
                                                               self.name)
            warnings.warn(msg)
            name = None
        return name

    def __set__(self, instance, value):
        if value is None:
            value = HD.INULL
        elif _ut.is_valid_enum_str(self.name, value):
            if self.name == 'iztype':
                reftime = _iztype_reftime(instance, value)
                instance.reftime = reftime
                # this also shifts all non-null relative times (instance._allt)
            value = HD.ENUM_VALS[value]
        else:
            msg = 'Unrecognized enumerated value "{}" for header "{}"'
            raise ValueError(msg.format(value, self.name))
        super(EnumHeader, self).__set__(instance, value)


class StringHeader(SACHeader):
    def __get__(self, instance, instance_type):
        if instance is None:
            value = self
        else:
            value = instance._hs[HD.STRHDRS.index(self.name)]
            try:
                # value is a bytes
                value = value.decode()
            except AttributeError:
                # value is a str
                pass

            if value == HD.SNULL:
                value = None

            try:
                value = value.strip()
            except AttributeError:
                # it's None.  no .strip method
                pass
        return value

    def __set__(self, instance, value):
        if value is None:
            value = HD.SNULL
        elif len(value) > 8:
            msg = ("Alphanumeric headers longer than 8 characters are "
                   "right-truncated.")
            warnings.warn(msg)
        # values will truncate themselves, since _hs is dtype '|S8'
        try:
            instance._hs[HD.STRHDRS.index(self.name)] = value.encode('ascii',
                                                                     'strict')
        except AttributeError:
            instance._hs[HD.STRHDRS.index(self.name)] = value


# Headers for functions of .data (min, max, mean, len)
class DataHeader(SACHeader):
    def __init__(self, name, func):
        self.func = func
        super(DataHeader, self).__init__(name)

    def __get__(self, instance, instance_type):
        if instance is None:
            value = self
        else:
            try:
                value = self.func(instance.data)
                # convert to native Python types
                if self.name in HD.FLOATHDRS:
                    value = float(value)
                elif self.name in HD.INTHDRS:
                    value = int(value)
            except TypeError:
                # instance.data is None, get value from header
                if self.name in HD.FLOATHDRS:
                    value = instance._hf[HD.FLOATHDRS.index(self.name)].item()
                    value = None if value == HD.FNULL else value
                elif self.name in HD.INTHDRS:
                    value = instance._hi[HD.INTHDRS.index(self.name)].item()
                    value = None if value == HD.INULL else value
        return value

    def __set__(self, instance, value):
        msg = "{} is read-only".format(self.name)
        raise AttributeError(msg)


# OTHER GETTERS/SETTERS
def _get_e(self):
    try:
        if self.npts:
            e = self.b + (self.npts - 1) * self.delta
        else:
            e = self.b
    except TypeError:
        # b, npts, and/or delta are None/null
        # TODO: assume "b" is 0.0?
        e = None
    return e


def _iztype_reftime(sactrace, iztype):
    """
    Get the new reftime for a given iztype.

    Setting the iztype will shift the relative time headers, such that the
    header that iztype points to is (near) zero, and all others are shifted
    together by the difference.

    Affected headers: b, o, a, f, t0-t9

    :param sactrace:
    :type sactrace: SACTrace
    :param iztype: One of the following strings:
        'iunkn'
        'ib', begin time
        'iday', midnight of reference GMT day
        'io', event origin time
        'ia', first arrival time
        'it0'-'it9', user defined pick t0-t9.
    :type iztype: str
    :rtype reftime: UTCDateTime
    :return: The new SAC reference time.

    """
    # The Plan:
    # 1. find the seconds needed to shift the old reftime to the new one.
    # 2. shift reference time onto the iztype header using that shift value.
    # 3. this triggers an _allt shift of all relative times by the same amount.
    # 4. If all goes well, actually set the iztype in the header.

    # 1.
    if iztype == 'iunkn':
        # no shift
        ref_val = 0.0
    elif iztype == 'iday':
        # seconds since midnight of reference day
        reftime = sactrace.reftime
        ref_val = reftime - UTCDateTime(year=reftime.year,
                                        julday=reftime.julday)
    else:
        # a relative time header.
        # remove the 'i' (first character) in the iztype to get the header name
        ref_val = getattr(sactrace, iztype[1:])
        if ref_val is None:
            msg = "Reference header for iztype '{}' is not set".format(iztype)
            raise SacError(msg)

    # 2. set a new reference time,
    # 3. which also shifts all non-null relative times (sactrace._allt).
    #    remainder microseconds may be in the reference header value, because
    #    nzmsec can't hold them.
    new_reftime = sactrace.reftime + ref_val

    return new_reftime


# kevnm is 16 characters, split into two 8-character fields
# intercept and handle in while getting and setting
def _get_kevnm(self):
    kevnm = self._hs[HD.STRHDRS.index('kevnm')]
    kevnm2 = self._hs[HD.STRHDRS.index('kevnm2')]
    try:
        kevnm = kevnm.decode()
        kevnm2 = kevnm2.decode()
    except AttributeError:
        # kevnm is a str
        pass

    if kevnm == HD.SNULL:
        kevnm = ''
    if kevnm2 == HD.SNULL:
        kevnm2 = ''

    value = (kevnm + kevnm2).strip()

    if not value:
        value = None

    return value


def _set_kevnm(self, value):
    if value is None:
        value = HD.SNULL + HD.SNULL
    elif len(value) > 16:
        msg = "kevnm over 16 characters.  Truncated to {}.".format(value[:16])
        warnings.warn(msg)
    kevnm = '{:<8s}'.format(value[0:8])
    kevnm2 = '{:<8s}'.format(value[8:16])
    self._hs[HD.STRHDRS.index('kevnm')] = kevnm
    self._hs[HD.STRHDRS.index('kevnm2')] = kevnm2

# TODO: move get/set reftime up here, make it a property


# -------------------------- SAC OBJECT INTERFACE -----------------------------
class SACTrace(object):
    __doc__ = """
    Convenient and consistent in-memory representation of Seismic Analysis Code
    (SAC) files.

    This is the human-facing interface for making a valid instance.  For
    file-based or other constructors, see class methods .read and
    .from_obspy_trace.  SACTrace instances preserve relationships between
    header values.

    :param data: Associated time-series data vector. Optional. If omitted, None
        is set as the instance data attribute.
    :type data: :class:`numpy.ndarray` of float32

    Any valid header key/value pair is also an optional input keyword argument.
    If not provided, minimum required headers are set to valid default values.
    The default instance is an evenly-space trace, with a sample rate of 1.0,
    and len(data) or 0 npts, starting at 1970-01-01T00:00:00.000000.

    :var reftime: Read-only reference time.  Calculated from nzyear, nzjday,
        nzhour, nzmin, nzsec, nzmsec.
    :var byteorder: The byte order of the underlying header/data arrays.
        Raises :class:`SacError` if array byte orders are inconsistent, even in
        the case where '<' is your native order and byteorders look like '<',
        '=', '='.

    Any valid header name is also an attribute. See below, :mod:`header`,
    or individial attribution docstrings for more header information.

                                 THE SAC HEADER

    NOTE: All header names and string values are lowercase. Header value
    access should be through instance attributes.

    """ + HD.HEADER_DOCSTRING

    # ------------------------------- SAC HEADERS -----------------------------
    # SAC header values are defined as managed attributes, either as
    # descriptors or as properties, with getters and setters.
    #
    # Managed attributes are defined at the class leval, and therefore shared
    # across all instances, not attribute data themselves.
    #
    # This section looks ugly, but it allows for the following:
    # 1. Relationships/checks between header variables can be done in setters
    # 2. The underlying header array structure is retained, for quick writing.
    # 3. Header access looks like simple attribute-access syntax.
    #    Looks funny to read here, but natural to use.
    #
    # FLOATS
    delta = FloatHeader('delta')
    depmin = DataHeader('depmin', min)
    depmax = DataHeader('depmax', max)
    scale = FloatHeader('scale')
    odelta = FloatHeader('odelta')
    b = RelativeTimeHeader('b')
    e = property(_get_e, doc=HD.DOC['e'])
    o = RelativeTimeHeader('o')
    a = RelativeTimeHeader('a')
    internal0 = FloatHeader('internal0')
    t0 = RelativeTimeHeader('t0')
    t1 = RelativeTimeHeader('t1')
    t2 = RelativeTimeHeader('t2')
    t3 = RelativeTimeHeader('t3')
    t4 = RelativeTimeHeader('t4')
    t5 = RelativeTimeHeader('t5')
    t6 = RelativeTimeHeader('t6')
    t7 = RelativeTimeHeader('t7')
    t8 = RelativeTimeHeader('t8')
    t9 = RelativeTimeHeader('t9')
    f = RelativeTimeHeader('f')
    stla = GeographicHeader('stla')
    stlo = GeographicHeader('stlo')
    stel = FloatHeader('stel')
    stdp = FloatHeader('stdp')
    evla = GeographicHeader('evla')
    evlo = GeographicHeader('evlo')
    evdp = FloatHeader('evdp')
    mag = FloatHeader('mag')
    user0 = FloatHeader('user0')
    user1 = FloatHeader('user1')
    user2 = FloatHeader('user2')
    user3 = FloatHeader('user3')
    user4 = FloatHeader('user4')
    user5 = FloatHeader('user5')
    user6 = FloatHeader('user6')
    user7 = FloatHeader('user7')
    user8 = FloatHeader('user8')
    user9 = FloatHeader('user9')
    dist = FloatHeader('dist')
    az = FloatHeader('az')
    baz = FloatHeader('baz')
    gcarc = FloatHeader('gcarc')
    depmen = DataHeader('depmen', np.mean)
    cmpaz = FloatHeader('cmpaz')
    cmpinc = FloatHeader('cmpinc')
    #
    # INTS
    nzyear = IntHeader('nzyear')
    nzjday = IntHeader('nzjday')
    nzhour = IntHeader('nzhour')
    nzmin = IntHeader('nzmin')
    nzsec = IntHeader('nzsec')
    nzmsec = IntHeader('nzmsec')
    nvhdr = IntHeader('nvhdr')
    norid = IntHeader('norid')
    nevid = IntHeader('nevid')
    npts = DataHeader('npts', len)
    nwfid = IntHeader('nwfid')
    iftype = EnumHeader('iftype')
    idep = EnumHeader('idep')
    iztype = EnumHeader('iztype')
    iinst = IntHeader('iinst')
    istreg = IntHeader('istreg')
    ievreg = IntHeader('ievreg')
    ievtyp = EnumHeader('ievtyp')
    iqual = IntHeader('iqual')
    isynth = EnumHeader('isynth')
    imagtyp = EnumHeader('imagtyp')
    imagsrc = EnumHeader('imagsrc')
    leven = BoolHeader('leven')
    lpspol = BoolHeader('lpspol')
    lovrok = BoolHeader('lovrok')
    lcalda = BoolHeader('lcalda')
    unused23 = IntHeader('unused23')
    #
    # STRINGS
    kstnm = StringHeader('kstnm')
    kevnm = property(_get_kevnm, _set_kevnm, doc=HD.DOC['kevnm'])
    khole = StringHeader('khole')
    ko = StringHeader('ko')
    ka = StringHeader('ka')
    kt0 = StringHeader('kt0')
    kt1 = StringHeader('kt1')
    kt2 = StringHeader('kt2')
    kt3 = StringHeader('kt3')
    kt4 = StringHeader('kt4')
    kt5 = StringHeader('kt5')
    kt6 = StringHeader('kt6')
    kt7 = StringHeader('kt7')
    kt8 = StringHeader('kt8')
    kt9 = StringHeader('kt9')
    kf = StringHeader('kf')
    kuser0 = StringHeader('kuser0')
    kuser1 = StringHeader('kuser1')
    kuser2 = StringHeader('kuser2')
    kcmpnm = StringHeader('kcmpnm')
    knetwk = StringHeader('knetwk')
    kdatrd = StringHeader('kdatrd')
    kinst = StringHeader('kinst')

    def __init__(self, leven=True, delta=1.0, b=0.0, e=0.0, iztype='ib',
                 nvhdr=6, npts=0, iftype='itime', nzyear=1970, nzjday=1,
                 nzhour=0, nzmin=0, nzsec=0, nzmsec=0, lcalda=False,
                 lpspol=True, lovrok=True, internal0=2.0, data=None, **kwargs):
        """
        Initialize a SACTrace object using header key-value pairs and a
        numpy.ndarray for the data, both optional.

        ..rubric:: Example

        >>> sac = SACTrace(nzyear=1995, nzmsec=50, data=np.arange(100))
        >>> print(sac)  # doctest: +NORMALIZE_WHITESPACE
        Reference Time = 01/01/1995 (001) 00:00:00.050000
           iztype IB: begin time
        b          = 0.0
        delta      = 1.0
        e          = 99.0
        iftype     = itime
        internal0  = 2.0
        iztype     = ib
        lcalda     = False
        leven      = True
        lovrok     = True
        lpspol     = True
        npts       = 100
        nvhdr      = 6
        nzhour     = 0
        nzjday     = 1
        nzmin      = 0
        nzmsec     = 50
        nzsec      = 0
        nzyear     = 1995

        """
        # The Plan:
        # 1. Build the default header dictionary and update with provided
        #    values.
        # 2. Convert header dict to arrays (util.dict_to_header_arrays
        #    initializes the arrays and fills in without checking.
        # 3. set the _h[fis] and data arrays on self.

        # 1.
        # build the required header from provided or default values
        header = {'leven': leven, 'npts': npts, 'delta': delta, 'b': b, 'e': e,
                  'iztype': iztype, 'nvhdr': nvhdr, 'iftype': iftype,
                  'nzyear': nzyear, 'nzjday': nzjday, 'nzhour': nzhour,
                  'nzmin': nzmin, 'nzsec': nzsec, 'nzmsec': nzmsec,
                  'lcalda': lcalda, 'lpspol': lpspol, 'lovrok': lovrok,
                  'internal0': internal0}

        # combine header with remaining non-required args.
        # user can put non-SAC key:value pairs into the header, but they're
        # ignored on write.
        header.update(kwargs)

        # -------------------------- DATA ARRAY -------------------------------
        if data is None:
            # this is like "headonly=True"
            pass
        else:
            if not isinstance(data, np.ndarray):
                raise TypeError("data needs to be a numpy.ndarray")
            else:
                # Only copy the data if they are not of the required type
                # XXX: why require little endian instead of native byte order?
                # data = np.require(data, native_str('<f4'))
                pass

        # --------------------------- HEADER ARRAYS ---------------------------
        # 2.
        # TODO: this is done even when we're reading a file.
        #   if it's too much overhead, it may need to change

        # swap enum names for integer values in the header dictionary
        header = _ut.enum_string_to_int(header)

        # XXX: will these always be little endian?
        hf, hi, hs = _io.dict_to_header_arrays(header)

        # we now have data and headers, either default or provided.

        # 3.
        # this completely sidesteps any checks provided by class properties
        self._hf = hf
        self._hi = hi
        self._hs = hs
        self.data = data

        self._set_distances()

    @property
    def _header(self):
        """
        Convenient read-only dictionary of non-null header array values.

        Header value access should be through instance attributes. All header
        names and string values are lowercase. Computed every time, so use
        frugally. See class docstring for header descriptions.

        """
        out = _io.header_arrays_to_dict(self._hf, self._hi, self._hs,
                                        nulls=False)
        return out

    @property
    def byteorder(self):
        """
        The byte order of the underlying header/data arrays.

        Raises SacError if array byte orders are inconsistent, even in the
        case where '<' is your native order and byteorders look like
        '<', '=', '='.

        """
        try:
            if self.data is None:
                assert self._hf.dtype.byteorder == self._hi.dtype.byteorder
            else:
                assert self._hf.dtype.byteorder == self._hi.dtype.byteorder ==\
                    self.data.dtype.byteorder
        except AssertionError:
            msg = 'Inconsistent header/data byteorders.'
            raise SacError(msg)

        bo = self._hf.dtype.byteorder

        if bo == '=':
            byteorder = sys.byteorder
        elif bo == '<':
            byteorder = 'little'
        elif bo == '>':
            byteorder = 'big'

        return byteorder

    # TODO: make a byteorder setter?
    def _byteswap(self):
        """
        Change the underlying byte order and dtype interpretation of the float,
        int, and (if present) data arrays.

        """
        try:
            self._hf = self._hf.byteswap(True).newbyteorder('S')
            self._hi = self._hi.byteswap(True).newbyteorder('S')
            if self.data is not None:
                self.data = self.data.byteswap(True).newbyteorder('S')
        except Exception as e:
            # if this fails, roll it back?
            raise e

    @property
    def reftime(self):
        """
        Get or set the SAC header reference time as a UTCDateTime instance.

        reftime is not an attribute, but is constructed and dismantled each
        time directly to/from the SAC "nz"-time fields.

        Setting a new reftime shifts all non-null relative time headers
        accordingly.  It accepts a UTCDateTime object, from which time shifts
        are calculated.

        ..rubric:: notes

        The reftime you supply will be robbed of its remainder microseconds,
        which are then pushed into the relative time header shifts.  This means
        that the reftime you observe after you set it here may not exactly
        match the reftime you supplied; it may be `remainder microseconds`
        earlier. Nor will the iztype reference header value be exactly zero;
        it will be equal to `remainder microseconds` (as seconds).

        """
        return _ut.get_sac_reftime(self._header)

    @reftime.setter
    def reftime(self, new_reftime):
        try:
            old_reftime = self.reftime

            # snap the new reftime to the most recent milliseconds
            # (subtract the leftover microseconds)
            ns = new_reftime.ns
            utc = UTCDateTime(ns=(ns - ns % 1000000))

            self.nzyear = utc.year
            self.nzjday = utc.julday
            self.nzhour = utc.hour
            self.nzmin = utc.minute
            self.nzsec = utc.second
            self.nzmsec = utc.microsecond / 1000

            # get the float seconds between the old and new reftimes
            shift = old_reftime - utc

            # shift the relative time headers
            self._allt(np.float32(shift))

        except AttributeError:
            msg = "New reference time must be an obspy.UTCDateTime instance."
            raise TypeError(msg)

    # --------------------------- I/O METHODS ---------------------------------
    @classmethod
    def read(cls, source, headonly=False, ascii=False, byteorder=None,
             checksize=False, debug_strings=False, encoding='ASCII'):
        """
        Construct an instance from a binary or ASCII file on disk.

        :param source: Full path string for File-like object from a SAC binary
            file on disk.  If it is an open File object, open 'rb'.
        :type source: str or file
        :param headonly: If headonly is True, only read the header arrays not
            the data array.
        :type headonly: bool
        :param ascii: If True, file is a SAC ASCII/Alphanumeric file.
        :type ascii: bool
        :param byteorder: If omitted or None, automatic byte-order checking is
            done, starting with native order. If byteorder is specified and
            incorrect, a :class:`SacIOError` is raised. Only valid for binary
            files.
        :type byteorder: str {'little', 'big'}, optional
        :param checksize: If True, check that the theoretical file size from
            the header matches the size on disk. Only valid for binary files.
        :type checksize: bool
        :param debug_strings: By default, non-ASCII and null-termination
            characters are removed from character header fields, and those
            beginning with '-12345' are considered unset. If True, they
            are instead passed without modification.  Good for debugging.
        :type debug_strings: bool
        :param encoding: Encoding string that passes the user specified
        encoding scheme.
        :type encoding: str

        :raises: :class:`SacIOError` if checksize failed, byteorder was wrong,
            or header arrays are wrong size.

        .. rubric:: Example

        >>> from obspy.core.util import get_example_file
        >>> from obspy.io.sac.util import SacInvalidContentError
        >>> file_ = get_example_file("test.sac")
        >>> sac = SACTrace.read(file_, headonly=True)
        >>> sac.data is None
        True
        >>> sac = SACTrace.read(file_, headonly=False)
        >>> sac.data  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        array([ -8.74227766e-08,  -3.09016973e-01,  -5.87785363e-01,
                -8.09017122e-01,  -9.51056600e-01,  -1.00000000e+00,
                -9.51056302e-01,  -8.09016585e-01,  -5.87784529e-01,
                ...
                 8.09022486e-01,   9.51059461e-01,   1.00000000e+00,
                 9.51053500e-01,   8.09011161e-01,   5.87777138e-01,
                 3.09007347e-01], dtype=float32)

        See also: :meth:`SACTrace.validate`

        """
        if ascii:
            hf, hi, hs, data = _io.read_sac_ascii(source, headonly=headonly)
        else:
            hf, hi, hs, data = _io.read_sac(source, headonly=headonly,
                                            byteorder=byteorder,
                                            checksize=checksize)
        if not debug_strings:
            for i, val in enumerate(hs):
                val = _ut._clean_str(val.decode(encoding, 'replace'),
                                     strip_whitespace=False)
                if val.startswith(native_str('-12345')):
                    val = HD.SNULL
                hs[i] = val.encode(encoding, 'replace')

        sac = cls._from_arrays(hf, hi, hs, data)
        if sac.dist is None:
            sac._set_distances()

        return sac

    def write(self, dest, headonly=False, ascii=False, byteorder=None,
              flush_headers=True):
        """
        Write the header and (optionally) data arrays to a SAC binary file.

        :param dest: Full path or File-like object to SAC binary file on disk.
        :type dest: str or file
        :param headonly: If headonly is True, only read the header arrays not
            the data array.
        :type headonly: bool
        :param ascii: If True, file is a SAC ASCII/Alphanumeric file.
        :type ascii: bool
        :param byteorder: If omitted or None, automatic byte-order checking is
            done, starting with native order. If byteorder is specified and
            incorrect, a :class:`SacIOError` is raised. Only valid for binary
            files.
        :type byteorder: str {'little', 'big'}, optional
        :param flush_headers: If True, update data headers like 'depmin' and
            'depmax' with values from the data array.
        :type flush_headers: bool

        """
        if headonly:
            data = None
        else:
            # do a check for float32 data here instead of arrayio.write_sac?
            data = self.data
            if flush_headers:
                self._flush_headers()

        if ascii:
            _io.write_sac_ascii(dest, self._hf, self._hi, self._hs, data)
        else:
            byteorder = byteorder or self.byteorder
            _io.write_sac(dest, self._hf, self._hi, self._hs, data,
                          byteorder=byteorder)

    @classmethod
    def _from_arrays(cls, hf=None, hi=None, hs=None, data=None):
        """
        Low-level array-based constructor.

        This constructor is good for getting a "blank" SAC object, and is used
        in other, perhaps more useful, alternate constructors ("See Also").
        No value checking is done and header values are completely overwritten
        with the provided arrays, which is why this is a hidden constructor.

        :param hf: SAC float header array
        :type hf: :class:`numpy.ndarray` of floats
        :param hi: SAC int header array
        :type hi: :class:`numpy.ndarray` of ints
        :param hs: SAC string header array
        :type hs: :class:`numpy.ndarray` of str
        :param data: SAC data array, optional.

        If omitted or None, the header arrays are intialized according to
        :func:`arrayio.init_header_arrays`.  If data is omitted, it is
        simply set to None on the corresponding :class:`SACTrace`.

        .. rubric:: Example

        >>> sac = SACTrace._from_arrays()
        >>> print(sac)  # doctest: +NORMALIZE_WHITESPACE +SKIP
        Reference Time = XX/XX/XX (XXX) XX:XX:XX.XXXXXX
            iztype not set
        lcalda     = True
        leven      = False
        lovrok     = False
        lpspol     = False

        """
        # use the first byteorder we find, or system byteorder if we
        # never find any
        bo = '='
        for arr in (hf, hi, hs, data):
            try:
                bo = arr.dtype.byteorder
                break
            except AttributeError:
                # arr is None (not supplied)
                pass
        hf0, hi0, hs0 = _io.init_header_arrays(byteorder=bo)
        # TODO: hf0, hi0, hs0 = _io.init_header_array_values(hf0, hi0, hs0)

        if hf is None:
            hf = hf0
        if hi is None:
            hi = hi0
        if hs is None:
            hs = hs0

        # get the default instance, but completely replace the arrays
        # initializes arrays twice, but it beats converting empty arrays to a
        # dict and then passing it to __init__, i think...maybe...
        sac = cls()
        sac._hf = hf
        sac._hi = hi
        sac._hs = hs
        sac.data = data

        return sac

    # TO/FROM OBSPY TRACES
    @classmethod
    def from_obspy_trace(cls, trace, keep_sac_header=True):
        """
        Construct an instance from an ObsPy Trace.

        :param trace: Source Trace object
        :type trace: :class:`~obspy.core.Trace` instance
        :param keep_sac_header: If True, any old stats.sac header values are
            kept as is, and only a minimal set of values are updated from the
            stats dictionary: npts, e, and data.  If an old iztype and a valid
            reftime are present, b and e will be properly referenced to it.  If
            False, a new SAC header is constructed from only information found
            in the stats dictionary, with some other default values introduced.
        :type keep_sac_header: bool

        """
        header = _ut.obspy_to_sac_header(trace.stats, keep_sac_header)

        # handle the data headers
        data = trace.data
        try:
            if len(data) == 0:
                # data is a empty numpy.array
                data = None
        except TypeError:
            # data is None
            data = None

        try:
            byteorder = data.dtype.byteorder
        except AttributeError:
            # data is None
            byteorder = '='

        hf, hi, hs = _io.dict_to_header_arrays(header, byteorder=byteorder)
        sac = cls._from_arrays(hf, hi, hs, data)
        # sac._flush_headers()

        return sac

    def to_obspy_trace(self, debug_headers=False, encoding='ASCII'):
        """
        Return an ObsPy Trace instance.

        Required headers: nz-time fields, npts, delta, calib, kcmpnm, kstnm,
        ...?

        :param debug_headers: Include _all_ SAC headers into the
            Trace.stats.sac dictionary.
        :type debug_headers: bool
        :param encoding: Encoding string that passes the user specified
        encoding scheme.
        :type encoding: str

        .. rubric:: Example

        >>> from obspy.core.util import get_example_file
        >>> file_ = get_example_file("test.sac")
        >>> sac = SACTrace.read(file_, headonly=True)
        >>> tr = sac.to_obspy_trace()
        >>> print(tr)  # doctest: +ELLIPSIS
        .STA..Q | 1978-07-18T08:00:10.000000Z - ... | 1.0 Hz, 100 samples

        """
        # make the obspy test for tests/data/testxy.sac pass
        # ObsPy does not require a valid reftime
        # try:
        #     self.validate('reftime')
        # except SacInvalidContentError:
        #     if not self.nzyear:
        #         self.nzyear = 1970
        #     if not self.nzjday:
        #         self.nzjday = 1
        #     for hdr in ['nzhour', 'nzmin', 'nzsec', 'nzmsec']:
        #         if not getattr(self, hdr):
        #             setattr(self, hdr, 0)
        self.validate('delta')
        if self.data is None:
            # headonly is True
            # Make it something palatable to ObsPy
            data = np.array([], dtype=self._hf.dtype.byteorder + 'f4')
        else:
            data = self.data

        sachdr = _io.header_arrays_to_dict(self._hf, self._hi, self._hs,
                                           nulls=debug_headers,
                                           encoding=encoding)
        # TODO: logic to use debug_headers for real

        stats = _ut.sac_to_obspy_header(sachdr)

        return Trace(data=data, header=stats)

    # ---------------------- other properties/methods -------------------------
    def validate(self, *tests):
        """
        Check validity of loaded SAC file content, such as header/data
        consistency.

        :param tests: One or more of the following validity tests:
            'delta' : Time step "delta" is positive.
            'logicals' : Logical values are 0, 1, or null
            'data_hdrs' : Length, min, mean, max of data array match header
                values.
            'enums' : Check validity of enumerated values.
            'reftime' : Reference time values in header are all set.
            'reltime' : Relative time values in header are absolutely
                referenced.
            'all' : Do all tests.
        :type tests: str

        :raises: :class:`SacInvalidContentError` if any of the specified tests
            fail. :class:`ValueError` if 'data_hdrs' is specified and data is
            None, empty array, or no tests specified.

        .. rubric:: Example

        >>> from obspy.core.util import get_example_file
        >>> from obspy.io.sac.util import SacInvalidContentError
        >>> file_ = get_example_file("LMOW.BHE.SAC")
        >>> sac = SACTrace.read(file_)
        >>> # make the time step invalid, catch it, and fix it
        >>> sac.delta *= -1.0
        >>> try:
        ...     sac.validate('delta')
        ... except SacInvalidContentError as e:
        ...     sac.delta *= -1.0
        ...     sac.validate('delta')
        >>> # make the data and depmin/men/max not match, catch the validation
        >>> # error, then fix (flush) the headers so that they validate
        >>> sac.data += 5.0
        >>> try:
        ...     sac.validate('data_hdrs')
        ... except SacInvalidContentError:
        ...     sac._flush_headers()
        ...     sac.validate('data_hdrs')

        """
        _io.validate_sac_content(self._hf, self._hi, self._hs, self.data,
                                 *tests)

    def _format_header_str(self, hdrlist='all'):
        """
        Produce a print-friendly string of header values for __repr__ ,
        .listhdr(), and .lh()

        """
        # interpret hdrlist
        if hdrlist == 'all':
            hdrlist = sorted(self._header.keys())
        elif hdrlist == 'picks':
            hdrlist = ('a', 'b', 'e', 'f', 'o', 't0', 't1', 't2', 't3', 't4',
                       't5', 't6', 't7', 't8', 't9')
        else:
            msg = "Unrecognized hdrlist '{}'".format(hdrlist)
            raise ValueError(msg)

        # start building header string
        #
        # reference time
        header_str = []
        try:
            timefmt = "Reference Time = %m/%d/%Y (%j) %H:%M:%S.%f"
            header_str.append(self.reftime.strftime(timefmt))
        except (ValueError, SacError):
            msg = "Reference time information incomplete."
            warnings.warn(msg)
            notime_str = "Reference Time = XX/XX/XX (XXX) XX:XX:XX.XXXXXX"
            header_str.append(notime_str)
        #
        # reftime type
        # TODO: use enumerated value dict here?
        iztype = self.iztype
        if iztype is None:
            header_str.append("\tiztype not set")
        elif iztype == 'ib':
            header_str.append("\tiztype IB: begin time")
        elif iztype == 'io':
            header_str.append("\tiztype IO: origin time")
        elif iztype == 'ia':
            header_str.append("\tiztype IA: first arrival time")
        elif iztype[1] == 't':
            vals = (iztype.upper(), iztype[1:])
            izfmt = "\tiztype {}: user-defined time {}"
            header_str.append(izfmt.format(*vals))
        elif iztype == 'iunkn':
            header_str.append("\tiztype IUNKN (Unknown)")
        else:
            header_str.append("\tunrecognized iztype: {}".format(iztype))
        #
        # non-null headers
        hdrfmt = "{:10.10s} = {}"
        for hdr in hdrlist:
            # XXX: non-null header values might have no property for getattr
            try:
                header_str.append(hdrfmt.format(hdr, getattr(self, hdr)))
            except AttributeError:
                header_str.append(hdrfmt.format(hdr, self._header[hdr]))

        return '\n'.join(header_str)

    def listhdr(self, hdrlist='all'):
        """
        Print header values.

        Default is all non-null values.

        :param hdrlist: Which header fields to you want to list. Choose one of
            {'all', 'picks'} or iterable of header fields.  An iterable of
            header fields can look like 'bea' or ('b', 'e', 'a').

            'all' (default) prints all non-null values.
            'picks' prints fields which are used to define time picks.

        An iterable of header fields can look like 'bea' or ('b', 'e', 'a').

        .. rubric:: Example

        >>> from obspy.core.util import get_example_file
        >>> file_ = get_example_file("LMOW.BHE.SAC")
        >>> sac = SACTrace.read(file_)
        >>> sac.lh()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Reference Time = 04/10/2001 (100) 00:23:00.465000
           iztype IB: begin time
        a          = 0.0
        b          = 0.0
        delta      = 0.009999999...
        depmax     = 0.003305610...
        depmen     = 0.00243799...
        depmin     = 0.00148824...
        e          = 0.98999997...
        iftype     = itime
        iztype     = ib
        kcmpnm     = BHE
        kevnm      = None
        kstnm      = LMOW
        lcalda     = True
        leven      = True
        lpspol     = False
        nevid      = 0
        norid      = 0
        npts       = 100
        nvhdr      = 6
        nzhour     = 0
        nzjday     = 100
        nzmin      = 23
        nzmsec     = 465
        nzsec      = 0
        nzyear     = 2001
        stla       = -39.409999...
        stlo       = 175.75
        unused23   = 0
        """
        # https://ds.iris.edu/files/sac-manual/commands/listhdr.html
        print(self._format_header_str(hdrlist))

    def lh(self, *args, **kwargs):
        """Alias of listhdr method."""
        self.listhdr(*args, **kwargs)

    def __str__(self):
        return self._format_header_str()

    def __repr__(self):
        # XXX: run self._flush_headers first?
        # TODO: make this somehow more readable.
        h = sorted(self._header.items())
        fmt = ", {}={!r}" * len(h)
        argstr = fmt.format(*chain.from_iterable(h))[2:]
        return self.__class__.__name__ + "(" + argstr + ")"

    def copy(self):
        return deepcopy(self)

    def _flush_headers(self):
        """
        Flush to the header arrays any header property values that may not be
        reflected there, such as data min/max/mean, npts, e.

        """
        # XXX: do I really care which byte order it is?
        # self.data = np.require(self.data, native_str('<f4'))
        self._hi[HD.INTHDRS.index('npts')] = self.npts
        self._hf[HD.FLOATHDRS.index('e')] = self.e
        self._hf[HD.FLOATHDRS.index('depmin')] = self.depmin
        self._hf[HD.FLOATHDRS.index('depmax')] = self.depmax
        self._hf[HD.FLOATHDRS.index('depmen')] = self.depmen

    def _allt(self, shift):
        """
        Shift all relative time headers by some value (addition).

        Similar to SAC's "chnhdr allt".

        Note
        ----
        This method is triggered by setting an instance's iztype or changing
        its reference time, which is the most likely use case for this
        functionality.  If what you're trying to do is set an origin time and
        make a file origin-based:

        SAC> CHNHDR O GMT 1982 123 13 37 10 103
        SAC>  LISTHDR O
        O 123.103
        SAC>  CHNHDR ALLT -123.103 IZTYPE IO

        ...it is recommended to just make sure your target reference header is
        set and correct, and set the iztype:

        >>> from obspy import UTCDateTime
        >>> from obspy.core.util import get_example_file
        >>> file_ = get_example_file("test.sac")
        >>> sac = SACTrace.read(file_)
        >>> sac.o = UTCDateTime(year=1982, julday=123,
        ...                     hour=13, minute=37,
        ...                     second=10, microsecond=103)
        >>> sac.iztype = 'io'

        The iztype setter will deal with shifting the time values.

        """
        for hdr in ['b', 'o', 'a', 'f'] + ['t' + str(i) for i in range(10)]:
            val = getattr(self, hdr)
            if val is not None:
                setattr(self, hdr, val + shift)

    def _set_distances(self, force=False):
        """
        Calculate dist, az, baz, gcarc.  If force=True, ignore lcalda.
        Raises SacHeaderError if force=True and geographic headers are unset.

        """
        if self.lcalda or force:
            try:
                m, az, baz = gps2dist_azimuth(self.evla, self.evlo, self.stla,
                                              self.stlo)
                dist = m / 1000.0
                gcarc = kilometer2degrees(dist)
                self._hf[HD.FLOATHDRS.index('az')] = az
                self._hf[HD.FLOATHDRS.index('baz')] = baz
                self._hf[HD.FLOATHDRS.index('dist')] = dist
                self._hf[HD.FLOATHDRS.index('gcarc')] = gcarc
            except (ValueError, TypeError):
                # one or more of the geographic values is None
                if force:
                    msg = ("Not enough information to calculate distance, "
                           "azimuth.")
                    raise SacHeaderError(msg)
