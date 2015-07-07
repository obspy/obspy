# -*- coding: utf-8 -*-
"""
Object-oriented interface to the SAC file format.

The SACTrace object maintains consistency between SAC headers and manages
header values in a user-friendly way. This includes some value-checking, native
Python logicals (True, False) and nulls (None) instead of SAC's 0, 1, or -12345...

SAC headers are implemented as properties, with appropriate getters and setters.

"""
import sys
import warnings
from copy import deepcopy
from itertools import chain

import numpy as np
from future.utils import native_str
from obspy import Trace, UTCDateTime
from obspy.core.util.geodetics import gps2DistAzimuth, kilometer2degrees

from .. import header as HD
from .util import SacError, SacHeaderError, SacInvalidContentError
from .. import util as _ut
from .. import arrayio as _io


# ------------- HEADER GETTER/SETTERS -----------------------------------------
#
# These are functions used to set up header properties on the SACTrace class.
# Properties are accessed like class attributes, but use getter and/or setter
# functions.  They're defined here, outside the class, b/c there are lots of
# properties and we want to keep the class clean.
#
# getter/setter factories:
# Property getters/setters must be defined for _each_ header, even if they work
# the exact same way for similar headers, so we define function factories here
# for groups of headers that will be gotten and set in a similar fashion.
#
# Usage: delta_getter = _floatgetter('delta')
#        delta_setter = _floatsetter('delta')
# These factories produce functions that simply index into their array to get or set.
# Use them for header values in hf, hi, and hs that need no special handling.
# Values that depend on other values will need their own getters/setters.
# See:
# http://stackoverflow.com/questions/2123585/python-multiple-properties-one-setter-getter
#
# floats
def _floatgetter(hdr):
    def get_float(self):
        value = self._hf[HD.FLOATHDRS.index(hdr)]
        if value == HD.FNULL:
            value = None
        return value
    return get_float

def _floatsetter(hdr):
    def set_float(self, value):
        if value is None:
            value = HD.FNULL
        self._hf[HD.FLOATHDRS.index(hdr)] = value
    return set_float

# ints
def _intgetter(hdr):
    def get_int(self):
        value = self._hi[HD.INTHDRS.index(hdr)]
        if value == HD.INULL:
            value = None
        return value
    return get_int

def _intsetter(hdr):
    def set_int(self, value):
        if not isinstance(value, (np.integer, int)):
            warnings.warn("Non-integers may be truncated.")
            print(" {}: {}".format(hdr, value))
        if value is None:
            value = HD.INULL
        self._hi[HD.INTHDRS.index(hdr)] = value
    return set_int

# logicals/bools (subtype of ints)
def _boolgetter(hdr):
    def get_bool(self):
        value = self._hi[HD.INTHDRS.index(hdr)]
        return bool(value)
    return get_bool

def _boolsetter(hdr):
    def set_bool(self, value):
        if value not in (True, False, 1, 0):
            msg = "Logical header values must be {True, False, 0, 1}"
            raise ValueError(msg)
        # booleans are subclasses of integers.  They will be set (cast)
        # directly into an integer array as 0 or 1.
        self._hi[HD.INTHDRS.index(hdr)] = value
    return set_bool

# enumerated values (stored as ints, represented by strings)
def _enumgetter(hdr):
    def get_enum(self):
        value = self._hi[HD.INTHDRS.index(hdr)]
        if value == HD.INULL:
            name = None
        elif _ut.is_valid_enum_int(hdr, value):
            name = HD.ENUM_NAMES[value]
        else:
            msg = """Unrecognized enumerated value {} for header "{}".
                     See .header for allowed values.""".format(value, hdr)
            warnings.warn(msg)
            name = None
        return name
    return get_enum

def _enumsetter(hdr):
    def set_enum(self, value):
        if value is None:
            value = HD.INULL
        elif _ut.is_valid_enum_str(hdr, value):
            value = HD.ENUM_VALS[value]
        else:
            msg = 'Unrecognized enumerated value "{}" for header "{}"'
            raise ValueError(msg.format(value, hdr))
        self._hi[HD.INTHDRS.index(hdr)] = value
    return set_enum

# strings
def _strgetter(hdr):
    def get_str(self):
        value = self._hs[HD.STRHDRS.index(hdr)]
        if value == HD.SNULL:
            value = None
        try:
            value = value.strip()
        except AttributeError:
            # it's None.  no .strip method
            pass
        return value
    return get_str

def _strsetter(hdr):
    def set_str(self, value):
        if value is None:
            value = HD.SNULL
        elif len(value) > 8:
            msg = "Alphanumeric headers longer than 8 characters are "\
                  "right-truncated."
            warnings.warn(msg)
        # they will truncate themselves, since _hs is dtype '|S8'
        self._hs[HD.STRHDRS.index(hdr)] = value
    return set_str

# Factory for functions of .data (min, max, mean, len)
def _make_data_func(func, hdr):
    # returns a method that returns the value of func(self.data), or the
    # corresponding array header value, if data is None
    def do_data_func(self):
        try:
            value = func(self.data)
        except TypeError:
            # data=None (headonly=True)
            try:
                value = self._hf[HD.FLOATHDRS.index(hdr)]
                null = HD.INULL
            except ValueError:
                # hdr is 'npts', the only integer
                # Will this also trip if a data-centric header is misspelled?
                value = self._hi[HD.INTHDRS.index(hdr)]
                null = HD.INULL
            if value == null:
                value = None
        return value
    return do_data_func

# Factory function for setting relative time headers with either a relative
# time float or an absolute UTCDateTime
# used for: b, o, a, f, t0-t9
def _reltime_setter(hdr):
    def set_reltime(self, value):
        if isinstance(value, UTCDateTime):
            # get the offset from reftime
            offset = value - self.reftime
        else:
            # value is already a reftime offset.
            offset = value
        # make and use a _floatsetter to actually set it
        floatsetter = _floatsetter(hdr)
        floatsetter(self, offset)
    return set_reltime

# Factory function for setting geographic header values (evlo, evla, stalo, stalat)
# that will check lcalda and calculate and set dist, az, baz, gcarc
def _geosetter(hdr):
    def set_geo(self, value):
        # make and use a _floatsetter
        set_geo_float = _floatsetter(hdr)
        set_geo_float(self, value)
        if self.lcalda:
            # check and maybe set lcalda
            try:
                self._set_distances()
            except SacHeaderError:
                pass
    return set_geo


# OTHER GETTERS/SETTERS
def _set_lcalda(self, value):
    # make and use a bool setter for lcalda
    lcalda_setter = _boolsetter('lcalda')
    lcalda_setter(self, value)
    # try to set set distances if value evaluates to True
    if value:
        try:
            self._set_distances()
        except SacHeaderError:
            pass


def _get_e(self):
    if self.npts:
        e = self.b + (self.npts - 1) * self.delta
    else:
        e = self.b
    return e


def _set_iztype(self, iztype):
    """
    Setting the iztype will shift the relative time headers, such that the
    header that iztype points to is (near) zero, and all others are shifted
    together by the difference.

    Affected headers: b, o, a, f, t0-t9

    """
    # The Plan:
    # 1. find the seconds needed to shift the old reftime to the new one.
    # 2. shift the reference time onto the iztype header using that shift value.
    # 3. this triggers an _allt shift of all relative times by the same amount.
    # 4. If all goes well, actually set the iztype in the header.

    # 1.
    if iztype == 'iunkn':
        # no shift
        ref_val = 0.0
    elif iztype == 'iday':
        # seconds since midnight of reference day
        reftime = self.reftime
        ref_val = reftime - UTCDateTime(year=reftime.year, julday=reftime.julday)
    else:
        # a relative time header.
        # remove the 'i' (first character) in the iztype to get the header name
        ref_val = getattr(self, iztype[1:])
        if ref_val is None:
            msg = "Reference header for iztype '{}' is not set".format(iztype)
            raise SacError(msg)

    # 2. set a new reference time,
    # 3. which also shifts all non-null relative times (self._allt). 
    #    remainder microseconds may be in the reference header value, because
    #    nzmsec can't hold them.
    self.reftime = self.reftime + ref_val

    # 4. no exceptions yet. actually set the iztype
    #    make an _enumsetter for iztype and use it, for its enum checking.
    izsetter = _enumsetter('iztype')
    izsetter(self, iztype)

# kevnm is 16 characters, split into two 8-character fields
# intercept and handle in while getting and setting
def _get_kevnm(self):
    # TODO: handle/understand ".decode()" stuff better
    kevnm = self._hs[HD.STRHDRS.index('kevnm')]
    kevnm2 = self._hs[HD.STRHDRS.index('kevnm2')]
    if kevnm == HD.SNULL:
        kevnm = ''
    if kevnm2 == HD.SNULL:
        kevnm2 = ''
    value = (kevnm.decode() + kevnm2.decode()).strip()
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

    SACTrace instances preserve relationships between header values.

    Attributes
    ----------
    reftime : datetime.datetime
        Read-only reference time, calculated from nzyear, nzjday, nzhour,
        nzmin, nzsec, nzmsec.
    Any valid header name is also an attribute. See below, pysac.header, or
    individial attribution docstrings for more header information.

                                 THE SAC HEADER

    NOTE: All header names and string values are lowercase. Header value
    access should be through instance attributes.

    """ + HD.HEADER_DOCSTRING

    def __init__(self, leven=True, delta=1.0, b=0.0, e=0.0, iztype='ib',
                 nvhdr=6, npts=0, iftype='itime', nzyear=1970, nzjday=1,
                 nzhour=0, nzmin=0, nzsec=0, nzmsec=0, lcalda=False,
                 lpspol=True, lovrok=True, internal0=2.0, cmpaz=0.0,
                 cmpinc=0.0, kcmpnm='Z', data=None, **kwargs):
        """
        Initialize a SACTrace object using header key-value pairs and a
        numpy.ndarray for the data, both optional.

        If not provided, the required headers are set to valid default values.
        The default instance is an evenly-space Z trace, with a sample rate of
        1.0, and len(data) or 0 npts, starting at 1970-01-01T00:00:00.000000.

        Parameters
        ----------
        Any SAC header key-value pair is a valid argument. See the class
            documentation for header descriptions.
        data : numpy.ndarray (optional)
            Associated time-series data vector. If omitted, None is set as the
            instance data attribute.

        Examples
        --------
        >>> sac = SACTrace(nzyear=1995, nzmsec=50, data=np.arange(100))
        >>> print sac
        Reference Time = 01/01/1995 (001) 00:00:00.050000
                iztype IB: begin time
        b          = 0.0
        cmpaz      = 0.0
        cmpinc     = 0.0
        delta      = 1.0
        e          = 99.0
        iftype     = itime
        internal0  = 2.0
        iztype     = ib
        kcmpnm     = Z
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
        # 1. Build the default header dictionary and update with provided values.
        # 2. Convert header dict to arrays (util.dict_to_header_arrays
        #   initializes the arrays and fills in without checking.
        # 3. set the _h[fis] and data arrays on self.

        #data = kwargs.pop('data', data)

        # 1.
        # build the required header from provided or default values
        header = {'leven': leven, 'npts': npts, 'delta': delta, 'b': b, 'e': e,
                  'iztype': iztype, 'nvhdr': nvhdr, 'iftype': iftype,
                  'nzyear': nzyear, 'nzjday': nzjday, 'nzhour': nzhour,
                  'nzmin': nzmin, 'nzsec': nzsec, 'nzmsec': nzmsec,
                  'lcalda': lcalda, 'lpspol': lpspol, 'lovrok': lovrok,
                  'internal0': internal0, 'cmpaz': cmpaz, 'cmpinc': cmpinc,
                  'kcmpnm': kcmpnm}

        #required = ['delta', 'b', 'npts', ...]
        #provided = locals()
        #for hdr in required:
        #    header[hdr] = kwargs.pop(hdr, provided[hdr])

        # combine header with remaining non-required args.
        # XXX: user can put non-SAC key:value pairs into the header.
        header.update(kwargs)

        # -------------------------- DATA ARRAY -------------------------------
        if data is None:
            # this is like "headonly=True"
            pass
        else:
            if not isinstance(data, np.ndarray):
                raise SacError("data needs to be a numpy.ndarray")
            else:
                # Only copy the data if they are not of the required type
                # XXX: why require little endian instead of native byte order?
                data = np.require(data, native_str('<f4'))

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

    # ---------------------------- SET UP HEADERS -----------------------------
    # SAC header values are set up as attributes, with getters and setters
    # format: header = property(getter_function, setter_function)
    #
    # Attributes are outside of __init__ b/c properties are just instructions
    # about how to retrieve attributes (getters/setters), and therefore shared
    # across all instances, not attribute data themselves.
    #
    # This section looks ugly, but it allows for the following:
    # 1. Relationships/checks between header variables can be done in setters
    # 2. The underlying header array structure is retained, for quick writing.
    # 3. Header access looks like simple attribute-access syntax.
    #    Looks funny to read here, but natural to use.

    # NOTE: To make something read-only, omit the second argument to "property".

    # FLOATS
    delta = property(_floatgetter('delta'), _floatsetter('delta'), doc=HD.DOC['delta'])
    depmin = property(_make_data_func(min, 'depmin'), doc=HD.DOC['depmin'])
    depmax = property(_make_data_func(max, 'depmax'), doc=HD.DOC['depmax'])
    scale = property(_floatgetter('scale'), _floatsetter('scale'), doc=HD.DOC['scale'])
    odelta = property(_floatgetter('odelta'), _floatsetter('odelta'), doc=HD.DOC['odelta'])
    b = property(_floatgetter('b'), _reltime_setter('b'), doc=HD.DOC['b'])
    e = property(_get_e, doc=HD.DOC['e'])
    o = property(_floatgetter('o'), _reltime_setter('o'), doc=HD.DOC['o'])
    a = property(_floatgetter('a'), _reltime_setter('a'), doc=HD.DOC['a'])
    internal0 = property(_floatgetter('internal0'), _floatsetter('internal0'))
    t0 = property(_floatgetter('t0'), _reltime_setter('t0'), doc=HD.DOC['t0'])
    t1 = property(_floatgetter('t1'), _reltime_setter('t1'), doc=HD.DOC['t1'])
    t2 = property(_floatgetter('t2'), _reltime_setter('t2'), doc=HD.DOC['t2'])
    t3 = property(_floatgetter('t3'), _reltime_setter('t3'), doc=HD.DOC['t3'])
    t4 = property(_floatgetter('t4'), _reltime_setter('t4'), doc=HD.DOC['t4'])
    t5 = property(_floatgetter('t5'), _reltime_setter('t5'), doc=HD.DOC['t5'])
    t6 = property(_floatgetter('t6'), _reltime_setter('t6'), doc=HD.DOC['t6'])
    t7 = property(_floatgetter('t7'), _reltime_setter('t7'), doc=HD.DOC['t7'])
    t8 = property(_floatgetter('t8'), _reltime_setter('t8'), doc=HD.DOC['t8'])
    t9 = property(_floatgetter('t9'), _reltime_setter('t9'), doc=HD.DOC['t9'])
    f = property(_floatgetter('f'), _reltime_setter('f'), doc=HD.DOC['f'])
    stla = property(_floatgetter('stla'), _geosetter('stla'), doc=HD.DOC['stla'])
    stlo = property(_floatgetter('stlo'), _geosetter('stlo'), doc=HD.DOC['stlo'])
    stel = property(_floatgetter('stel'), _floatsetter('stel'), doc=HD.DOC['stel'])
    stdp = property(_floatgetter('stdp'), _floatsetter('stdp'), doc=HD.DOC['stdp'])
    evla = property(_floatgetter('evla'), _geosetter('evla'), doc=HD.DOC['evla'])
    evlo = property(_floatgetter('evlo'), _geosetter('evlo'), doc=HD.DOC['evlo'])
    evdp = property(_floatgetter('evdp'), _floatsetter('evsp'), doc=HD.DOC['evdp'])
    mag = property(_floatgetter('mag'), _floatsetter('mag'), doc=HD.DOC['mag'])
    user0 = property(_floatgetter('user0'), _floatsetter('user0'), doc=HD.DOC['user0'])
    user1 = property(_floatgetter('user1'), _floatsetter('user1'), doc=HD.DOC['user1'])
    user2 = property(_floatgetter('user2'), _floatsetter('user2'), doc=HD.DOC['user2'])
    user3 = property(_floatgetter('user3'), _floatsetter('user3'), doc=HD.DOC['user3'])
    user4 = property(_floatgetter('user4'), _floatsetter('user4'), doc=HD.DOC['user4'])
    user5 = property(_floatgetter('user5'), _floatsetter('user5'), doc=HD.DOC['user5'])
    user6 = property(_floatgetter('user6'), _floatsetter('user6'), doc=HD.DOC['user6'])
    user7 = property(_floatgetter('user7'), _floatsetter('user7'), doc=HD.DOC['user7'])
    user8 = property(_floatgetter('user8'), _floatsetter('user8'), doc=HD.DOC['user8'])
    user9 = property(_floatgetter('user9'), _floatsetter('user9'), doc=HD.DOC['user9'])
    dist = property(_floatgetter('dist'), _floatsetter('dist'), doc=HD.DOC['dist'])
    az = property(_floatgetter('az'), _floatsetter('az'), doc=HD.DOC['az'])
    baz = property(_floatgetter('baz'), _floatsetter('baz'), doc=HD.DOC['baz'])
    gcarc = property(_floatgetter('gcarc'), _floatsetter('gcarc'), doc=HD.DOC['gcarc'])
    depmen = property(_make_data_func(np.mean, 'depmen'), doc=HD.DOC['depmen'])
    cmpaz = property(_floatgetter('cmpaz'), _floatsetter('cmpaz'), doc=HD.DOC['cmpaz'])
    cmpinc = property(_floatgetter('cmpinc'), _floatsetter('cmpinc'), doc=HD.DOC['cmpinc'])
    #
    # INTS
    nzyear = property(_intgetter('nzyear'), _intsetter('nzyear'), doc=HD.DOC['nzyear'])
    nzjday = property(_intgetter('nzjday'), _intsetter('nzjday'), doc=HD.DOC['nzjday'])
    nzhour = property(_intgetter('nzhour'), _intsetter('nzhour'), doc=HD.DOC['nzhour'])
    nzmin = property(_intgetter('nzmin'), _intsetter('nzmin'), doc=HD.DOC['nzmin'])
    nzsec = property(_intgetter('nzsec'), _intsetter('nzsec'), doc=HD.DOC['nzsec'])
    nzmsec = property(_intgetter('nzmsec'), _intsetter('nzmsec'), doc=HD.DOC['nzmsec'])
    nvhdr = property(_intgetter('nvhdr'), _intsetter('nvhdr'), doc=HD.DOC['nvhdr'])
    norid = property(_intgetter('norid'), _intsetter('norid'), doc=HD.DOC['norid'])
    nevid = property(_intgetter('nevid'), _intsetter('nevid'), doc=HD.DOC['nevid'])
    npts = property(_make_data_func(len, 'npts'), doc=HD.DOC['npts'])
    nwfid = property(_intgetter('nwfid'), _intsetter('nwfid'), doc=HD.DOC['nwfid'])
    iftype = property(_enumgetter('iftype'), _enumsetter('iftype'), doc=HD.DOC['iftype'])
    idep = property(_enumgetter('idep'), _enumsetter('idep'), doc=HD.DOC['idep'])
    iztype = property(_enumgetter('iztype'), _set_iztype, doc=HD.DOC['iztype'])
    iinst = property(_intgetter('iinst'), _intsetter('iinst'), doc=HD.DOC['iinst'])
    istreg = property(_intgetter('istreg'), _intsetter('istreg'), doc=HD.DOC['istreg'])
    ievreg = property(_intgetter('ievreg'), _intsetter('ievreg'), doc=HD.DOC['ievreg'])
    ievtyp = property(_enumgetter('ievtyp'), _enumsetter('ievtyp'), doc=HD.DOC['ievtyp'])
    iqual = property(_enumgetter('iqual'), _enumsetter('iqual'), doc=HD.DOC['iqual'])
    isynth = property(_enumgetter('isythn'), _enumsetter('isynth'), doc=HD.DOC['isynth'])
    imagtyp = property(_enumgetter('imagtyp'), _enumsetter('imagtyp'), doc=HD.DOC['imagtyp'])
    imagsrc = property(_enumgetter('imagsrc'), _enumsetter('imagsrc'), doc=HD.DOC['imagsrc'])
    leven = property(_boolgetter('leven'), _boolsetter('leven'), doc=HD.DOC['leven'])
    lpspol = property(_boolgetter('lpspol'), _boolsetter('lpspol'), doc=HD.DOC['lpspol'])
    lovrok = property(_boolgetter('lovrok'), _boolsetter('lovrok'), doc=HD.DOC['lovrok'])
    lcalda = property(_boolgetter('lcalda'), _set_lcalda, doc=HD.DOC['lcalda'])
    unused23 = property(_intgetter('unused23'), _intsetter('unused23'))
    #
    # STRINGS
    kstnm = property(_strgetter('kstnm'), _strsetter('kstnm'), doc=HD.DOC['kstnm'])
    kevnm = property(_get_kevnm, _set_kevnm, doc=HD.DOC['kevnm'])
    khole = property(_strgetter('khole'), _strsetter('khole'), doc=HD.DOC['khole'])
    ko = property(_strgetter('ko'), _strsetter('ko'), doc=HD.DOC['ko'])
    ka = property(_strgetter('ka'), _strsetter('ka'), doc=HD.DOC['ka'])
    kt0 = property(_strgetter('kt0'), _strsetter('kt0'), doc=HD.DOC['kt0'])
    kt1 = property(_strgetter('kt1'), _strsetter('kt1'), doc=HD.DOC['kt1'])
    kt2 = property(_strgetter('kt2'), _strsetter('kt2'), doc=HD.DOC['kt2'])
    kt3 = property(_strgetter('kt3'), _strsetter('kt3'), doc=HD.DOC['kt3'])
    kt4 = property(_strgetter('kt4'), _strsetter('kt4'), doc=HD.DOC['kt4'])
    kt5 = property(_strgetter('kt5'), _strsetter('kt5'), doc=HD.DOC['kt5'])
    kt6 = property(_strgetter('kt6'), _strsetter('kt6'), doc=HD.DOC['kt6'])
    kt7 = property(_strgetter('kt7'), _strsetter('kt7'), doc=HD.DOC['kt7'])
    kt8 = property(_strgetter('kt8'), _strsetter('kt8'), doc=HD.DOC['kt8'])
    kt9 = property(_strgetter('kt9'), _strsetter('kt9'), doc=HD.DOC['kt9'])
    kf = property(_strgetter('kf'), _strsetter('kf'), doc=HD.DOC['kf'])
    kuser0 = property(_strgetter('kuser0'), _strsetter('kuser0'), doc=HD.DOC['kuser0'])
    kuser1 = property(_strgetter('kuser1'), _strsetter('kuser1'), doc=HD.DOC['kuser1'])
    kuser2 = property(_strgetter('kuser2'), _strsetter('kuser2'), doc=HD.DOC['kuser2'])
    kcmpnm = property(_strgetter('kcmpnm'), _strsetter('kcmpnm'), doc=HD.DOC['kcmpnm'])
    knetwk = property(_strgetter('knetwk'), _strsetter('knetwk'), doc=HD.DOC['knetwk'])
    kdatrd = property(_strgetter('kdatrd'), _strsetter('kdatrd'))
    kinst = property(_strgetter('kinst'), _strsetter('kinst'), doc=HD.DOC['kinst'])

    @property
    def _header(self):
        """
        Convenient read-only dictionary of non-null header array values.

        Header value access should be through instance attributes. All header
        names and string values are lowercase. Computed every time, so use
        frugally. See class docstring for header descriptions.

        """
        return _io.header_arrays_to_dict(self._hf, self._hi, self._hs, nulls=False)

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
                assert self._hf.dtype.byteorder == self._hi.dtype.byteorder == \
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

        Notes
        -----
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

            # find the milliseconds and leftover microseconds in the new reftime
            milliseconds, rem_microseconds = _ut.split_microseconds(new_reftime.microsecond)

            # snap the new reftime to the most recent milliseconds
            # (subtract the leftover microseconds)
            new_reftime.microsecond -= rem_microseconds

            self.nzyear = new_reftime.year
            self.nzjday = new_reftime.julday
            self.nzhour = new_reftime.hour
            self.nzmin = new_reftime.minute
            self.nzsec = new_reftime.second
            self.nzmsec = new_reftime.microsecond / 1000

            # get the float seconds between the old and new reftimes
            shift = old_reftime - new_reftime

            # shift the relative time headers
            self._allt(np.float32(shift))

        except AttributeError:
            msg = "New reference time must be an obspy.UTCDateTime instance."
            raise ValueError(msg)

    # --------------------------- I/O METHODS ---------------------------------
    @classmethod
    def read(cls, source, headonly=False, ascii=False, byteorder=None, checksize=False):
        """
        Construct an instance from a binary or ASCII file on disk.

        Parameters
        ----------
        source : str or File-like object
            Full path or File-like object from SAC binary or ASCII file on disk.
        headonly, ascii : bool
            If headonly is True, don't read the data array.
            If ascii is True, the file is a SAC ASCII type.
        byteorder : str {'little', 'big'}, optional
            If omitted or None, automatic byte-order checking is done for
            binary files.  If byteorder is specified and incorrect, an
            exception is raised. Successful binary byteorder is stored as
            instance attribute .byteorder. This flag is ignored for ASCII files
            and .byteorder is set to native.
        checksize : bool, default False
            If true, check that theoretical file size from header matches disk size.

        Raises
        ------
        SacIOError
            checksize failed, byteorder was wrong, header arrays are wrong size.

        Examples
        --------
        >>> sac = SACTrace.read(filename, headonly=True)
        >>> try:
                sac.validate('data_hdrs')
            except SacInvalidContentError:
                sac = SACTrace.read(filename, headonly=False)
                sac.validate('data_hdrs')

        See Also
        --------
        SACTrace.validate

        """
        if ascii:
            hf, hi, hs, data = _io.read_sac_ascii(source, headonly=headonly)
        else:
            hf, hi, hs, data = _io.read_sac(source, headonly=headonly,
                                            byteorder=byteorder,
                                            checksize=checksize)

        return cls._from_arrays(hf, hi, hs, data)


    def write(self, dest, headonly=False, ascii=False, byteorder=None):
        """
        Parameters
        ----------
        dest : str or File-like object
            Full path or File-like object from SAC binary or ASCII file on disk.
        headonly, ascii : bool
            If headonly is True, it is assumed that the user intends to
            overwrite/modify only the header arrays of an existing file, and
            data headers are not flushed/updated before writing.
            If ascii is True, the file is an ASCII (alphanumeric) type file.
        byteorder : str {'little', 'big'}, optional
            Desired output byte order.  If omitted, instance byte order is used.
            If data=None, better make sure the file you're writing to has the
            same byte order as headers you're writing.

        """

        if headonly:
            data = None
        else:
            data = self.data
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

        Parameters
        ----------
        hf, hi, hs : numpy.ndarray
            Header arrays of dtype 'f4', int, '|S08', respectively.
        data : numpy.ndarray
            Corresponding seismogram array.

        Examples
        --------
        >>> sac = SACTrace._from_arrays()
        >>> print sac
        Reference Time = XX/XX/XX (XXX) XX:XX:XX.XXXXXX
                iztype not set
            lcalda     = False
            leven      = False
            lovrok     = False
            lpspol     = False

        See Also
        --------
        SACTrace.read : construct from a SAC binary or ASCII file on disk.
        SACTrace.write : write to a SAC binary or ASCII file on disk.
        SACTrace.from_obspy_trace : construct from an ObsPy Trace instance.

        """
        hf0, hi0, hs0 = _io.init_header_arrays()

        if hf is None:
            hf = hf0
        if hi is None:
            hi = hi0
        if hs is None:
            hs = hs0

        # get the default instance, but replace the arrays
        # itializes arrays twice, but it beats converting empty arrays to a
        # dict and then passing it to __init__, i think
        sac = cls()
        sac._hf = hf
        sac._hi = hi
        sac._hs = hs
        sac.data = data

        return sac


    # TO/FROM OBSPY TRACES
    @classmethod
    def from_obspy_trace(cls, trace, ignore_old_header=False):
        """
        Construct an instance from an ObsPy Trace.

        Parameters
        ----------
        trace : obspy.Trace instance
        ignore_old_header : bool
            If a trace.stats.sac dictionary is present, ignore it and build a
            new SAC header from scratch. Otherwise, keep all the old values,
            including reference time and iztype, and just overprint new header
            values related to the data vector and a few trace.stats fields.

        """
        # keep_sac_header
        header = _ut.obspy_to_sac_header(trace.stats, ignore_old_header)

        # handle the data headers
        data = trace.data
        try:
            if len(data) == 0:
                # data is a empty numpy.array
                data = None
        except TypeError:
            # data is None
            data = None

        hf, hi, hs = _io.dict_to_header_arrays(header)
        sac = cls._from_arrays(hf, hi, hs, data)
        sac._flush_headers()

        return sac


    def to_obspy_trace(self, debug_headers=False):
        """
        Return a dictionary suitable for an Obspy Stats header.

        Required headers:  nz-time fields, npts, delta, calib, kcmpnm, kstnm,
        ...?

        Parameters
        ----------
        debug_headers : bool
            Include _all_ SAC headers into the Trace.stats.sac dictionary.

        Examples
        --------
        >>> sac = SACTrace()
        >>> tr = Trace(data=sac.data, header=sac.to_obspy_header())

        """
        self.validate('reftime')
        self.validate('delta')
        try:
            self.validate('data_hdrs')
        except SacInvalidContentError:
            self._flush_headers()
            self.validate('data_hdrs')
        # TODO: does a sac file need to have iztype specified, or just 'b'?
        #self.validate('reltime')

        sachdr = _io.header_arrays_to_dict(self._hf, self._hi, self._hs,
                                           nulls=debug_headers)
        # TODO: logic to use debug_headers for real

        stats = _ut.sac_to_obspy_header(sachdr)

        return Trace(data=self.data, header=stats)

    # ---------------------- other properties/methods -------------------------
    def validate(self, *tests):
        """
        Check validity of loaded SAC file content, such as header/data consistency.

        Supply one or more of the following validity tests to perform:

        Parameters
        ----------
        tests : str {'delta', 'logicals', 'data_hdrs', 'enums', 'reftime', 'reltime'}
            Perform one or more of the following validity tests:

            'delta' : Time step "delta" is positive.
            'logicals' : Logical values are 0, 1, or null
            'data_hdrs' : Length, min, mean, max of data array match header values.
            'enums' : Check validity of enumerated values.
            'reftime' : Reference time values in header are all set.
            'reltime' : Relative time values in header can be absolutely referenced.
            'all' : Do all tests.

        Raises
        ------
        SacInvalidContentError
            If any specified tests fail.
        ValueError
            'data_hdrs' is specified and data is None, empty array

        Examples
        --------
        >>> sac = SACTrace.read(filename)
        >>> try:
                sac.validate('delta')
            except SacInvalidContentError as e:
                # i'm sure this is what they meant:-)
                sac.delta *= -1.0
                sac.validate('delta')

        >>> sac.data += 5.0
        >>> try:
                sac.validate('data_hdrs')
            except SacInvalidContentError:
                sac._flush_headers()
                sac.validate('data_hdrs')

        """
        _io.validate_sac_content(self._hf, self._hi, self._hs, self.data, *tests)

        return

    def _format_header_str(self, hdrlist='all'):
        """
        Produce a print-friendly string of header values for __repr__ ,
        .listhdr(), and .lh()

        """
        # interpret hdrlist
        if hdrlist == 'all':
            hdrlist = sorted(self._header.keys())
        elif hdrlist == 'picks':
            hdrlist = sorted(('b', 'e', 'o', 'a', 'f', 't0', 't1', 't2', 't3',
                              't4', 't5', 't6', 't7', 't8', 't9'))
        elif hdrlist == 'special':
            warnings.warn("Not implemented: use <arrow-up> in your interpreter")
            hdrlist = sorted(self._header.keys())
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
            warnings.warn("Reference time information incomplete.")
            header_str.append("Reference Time = XX/XX/XX (XXX) XX:XX:XX.XXXXXX")
        #
        # reftime type
        # TODO: use enumerated value dict here?
        iztype = self.iztype
        if iztype == None:
            header_str.append("\tiztype not set")
        elif iztype == 'ib':
            header_str.append("\tiztype IB: begin time")
        elif iztype == 'io':
            header_str.append("\tiztype IO: origin time")
        elif iztype == 'ia':
            header_str.append("\tiztype IA: first arrival time")
        elif iztype[1] == 't':
            vals = (iztype.upper(), iztype[1:])
            header_str.append("\tiztype {}: user-defined time {}".format(*vals))
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
            except AttributeError as e:
                header_str.append(hdrfmt.format(hdr, self._header[hdr]))

        return '\n'.join(header_str)


    def listhdr(self, hdrlist='all'):
        """
        Print header values.  Default, all non-null values.

        Parameters
        ----------
        hdrlist : str {'all', 'picks'} or iterable of header fields
            'all' (default) prints all non-null values.
            'picks' prints fields which are used to define time picks.

        An iterable of header fields can look like 'bea' or ('b', 'e', 'a').

        >>> sac = SACTrace.read('tests/data/test.sac')
        >>> sac.lh()
        Reference Time = 07/18/1978 (199) 08:00:00.000000
         unrecognized iztype: None
        b	= 10.0
        delta	= 1.0
        depmax	= 1.0
        depmin	= -1.0
        e	= 109.0
        istreg	= 1
        kcmpnm	= Q
        kevnm	= FUNCGEN: SINE
        kstnm	= STA
        npts	= 100
        nvhdr	= 6
        nzhour	= 8
        nzjday	= 199
        nzmin	= 0
        nzmsec	= 0
        nzsec	= 0
        nzyear	= 1978

        """
        # http://ds.iris.edu/files/sac-manual/commands/listhdr.html
        print self._format_header_str(hdrlist)

    def lh(self, *args, **kwargs):
        """Alias of listhdr method."""
        self.listhdr(*args, **kwargs)

    def __str__(self):
        return self._format_header_str()

    def __repr__(self):
        # I'm not proud...
        # XXX: run self._flush_headers first?
        # ', '.join(["{{0.{}}}".format(hdr) for hdr in sac._header]).format(sac)
        h = self._header
        argstr = (", {}={!r}" * len(h)).format(*chain.from_iterable(h.items()))[2:]
        return self.__class__.__name__ + "(" + argstr + ")"

    def copy(self):
        return deepcopy(self)

    def _flush_headers(self):
        """
        Flush to the header arrays any header property values that may not be
        reflected there, such as data min/max/mean, npts, e.

        """
        # XXX: do I really care which byte order it is?
        self.data = np.require(self.data, native_str('<f4'))
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

        >>> sac.o = UTCDateTime(....)
        >>> sac.iztype = 'io'

        The iztype setter will deal with shifting the time values.

        """
        for hdr in ['b', 'o', 'a', 'f'] + ['t'+str(i) for i in range(10)]:
            val = getattr(self, hdr)
            if val is not None:
                setattr(self, hdr, val + shift)

    def _set_distances(self, force=False):
        """
        Calculate dist, az, baz, gcarc.  If force=True, ignore lcalda.

        """
        if self.lcalda or force:
            try:
                m, az, baz = gps2DistAzimuth(self.evla, self.evlo, self.stla, self.stlo)
                dist = m / 1000.0
                gcarc = kilometer2degrees(dist)
                self.az = az
                self.baz = baz
                self.dist = dist
                self.gcarc = gcarc
            except TypeError:
                # one of the geographic values is None
                msg = "Not enough information to calculate distances and azimuths."
                raise SacHeaderError(msg)
        else:
            msg = "lcalda is False or unset. To set distances, set it to True."
            raise SacError(msg)
