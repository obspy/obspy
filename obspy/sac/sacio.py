#!/usr/bin/env python
# ------------------------------------------------------------------
# Filename: sacio.py
#  Purpose: Read & Write Seismograms, Format SAC.
#   Author: Yannik Behr, C. J. Ammon's
#    Email: yannik.behr@vuw.ac.nz
#
# Copyright (C) 2008-2012 Yannik Behr, C. J. Ammon's
# ------------------------------------------------------------------
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

from obspy import UTCDateTime, Trace
from obspy.core.compatibility import frombuffer
from obspy.core.util import gps2DistAzimuth, AttribDict
from obspy.core import compatibility
import numpy as np
import os
import time
import warnings
"""
Low-level module internally used for handling SAC files

An object-oriented version of C. J. Ammon's SAC I/O module.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & C. J. Ammon
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""


# we put here everything but the time, they are going to stats.starttime
# left SAC attributes, right trace attributes, see also
# http://www.iris.edu/KB/questions/13/SAC+file+format
convert_dict = {'npts': 'npts',
                'delta': 'delta',
                'kcmpnm': 'channel',
                'kstnm': 'station',
                'scale': 'calib',
                'knetwk': 'network',
                'khole': 'location'}

# all the sac specific extras, the SAC reference time specific headers are
# handled separately and are directly controlled by trace.stats.starttime.
SAC_EXTRA = ('depmin', 'depmax', 'odelta', 'o', 'a', 't0', 't1', 't2', 't3',
             't4', 't5', 't6', 't7', 't8', 't9', 'f', 'stla', 'stlo', 'stel',
             'stdp', 'evla', 'evlo', 'evdp', 'mag', 'user0', 'user1', 'user2',
             'user3', 'user4', 'user5', 'user6', 'user7', 'user8', 'user9',
             'dist', 'az', 'baz', 'gcarc', 'depmen', 'cmpaz', 'cmpinc',
             'nvhdr', 'norid', 'nevid', 'nwfid', 'iftype', 'idep', 'iztype',
             'iinst', 'istreg', 'ievreg', 'ievtype', 'iqual', 'isynth',
             'imagtyp', 'imagsrc', 'leven', 'lpspol', 'lovrok', 'lcalda',
             'kevnm', 'ko', 'ka', 'kt0', 'kt1', 'kt2', 'kt3', 'kt4', 'kt5',
             'kt6', 'kt7', 'kt8', 'kt9', 'kf', 'kuser0', 'kuser1', 'kuser2',
             'kdatrd', 'kinst', 'cmpinc', 'xminimum', 'xmaximum', 'yminimum',
             'ymaximum', 'unused6', 'unused7', 'unused8', 'unused9',
             'unused10', 'unused11', 'unused12')

FDICT = {'delta': 0, 'depmin': 1, 'depmax': 2, 'scale': 3,
         'odelta': 4, 'b': 5, 'e': 6, 'o': 7, 'a': 8, 'int1': 9,
         't0': 10, 't1': 11, 't2': 12, 't3': 13, 't4': 14,
         't5': 15, 't6': 16, 't7': 17, 't8': 18, 't9': 19,
         'f': 20, 'stla': 31, 'stlo': 32, 'stel': 33, 'stdp': 34,
         'evla': 35, 'evlo': 36, 'evdp': 38, 'mag': 39,
         'user0': 40, 'user1': 41, 'user2': 42, 'user3': 43,
         'user4': 44, 'user5': 45, 'user6': 46, 'user7': 47,
         'user8': 48, 'user9': 49, 'dist': 50, 'az': 51,
         'baz': 52, 'gcarc': 53, 'depmen': 56, 'cmpaz': 57,
         'cmpinc': 58, 'xminimum': 59, 'xmaximum': 60,
         'yminimum': 61, 'ymaximum': 62, 'unused6': 63,
         'unused7': 64, 'unused8': 65, 'unused9': 66,
         'unused10': 67, 'unused11': 68, 'unused12': 69}

IDICT = {'nzyear': 0, 'nzjday': 1, 'nzhour': 2, 'nzmin': 3,
         'nzsec': 4, 'nzmsec': 5, 'nvhdr': 6, 'norid': 7,
         'nevid': 8, 'npts': 9, 'nwfid': 11,
         'iftype': 15, 'idep': 16, 'iztype': 17, 'iinst': 19,
         'istreg': 20, 'ievreg': 21, 'ievtype': 22, 'iqual': 23,
         'isynth': 24, 'imagtyp': 25, 'imagsrc': 26,
         'leven': 35, 'lpspol': 36, 'lovrok': 37,
         'lcalda': 38}

SDICT = {'kstnm': 0, 'kevnm': 1, 'khole': 2, 'ko': 3, 'ka': 4,
         'kt0': 5, 'kt1': 6, 'kt2': 7, 'kt3': 8, 'kt4': 9,
         'kt5': 10, 'kt6': 11, 'kt7': 12, 'kt8': 13,
         'kt9': 14, 'kf': 15, 'kuser0': 16, 'kuser1': 17,
         'kuser2': 18, 'kcmpnm': 19, 'knetwk': 20,
         'kdatrd': 21, 'kinst': 22}

TWO_DIGIT_YEAR_MSG = ("SAC file with 2-digit year header field encountered. "
                      "This is not supported by the SAC file format standard. "
                      "Prepending '19'.")


class SacError(Exception):
    """
    Raised if the SAC file is corrupt or if necessary information
    in the SAC file is missing.
    """
    pass


class SacIOError(Exception):
    """
    Raised if the given SAC file can't be read.
    """
    pass


def _isText(filename, blocksize=512):
    """
    Check if it is a text or a binary file.
    """
    # This should  always be true if a file is a text-file and only true for a
    # binary file in rare occasions (see Recipe 173220 found on
    # http://code.activestate.com/)
    text_characters = str("").join(list(map(chr, range(32, 127))) +
                                   list("\n\r\t\b")).encode('ascii', 'ignore')
    _null_trans = compatibility.maketrans(b"", b"")
    with open(filename, 'rb') as fp:
        s = fp.read(blocksize)
    if b"\0" in s:
        return False

    if not s:  # Empty files are considered text
        return True

    # Get the non-text characters (maps a character to itself then
    # use the 'remove' option to get rid of the text characters.)
    t = s.translate(_null_trans, text_characters)

    # If more than 30% non-text characters, then
    # this is considered a binary file
    if len(t) / len(s) > 0.30:
        return 0
    return True


class SacIO(object):
    """
    Class for SAC file IO.

    Functions are given below, attributes/header
    fields (described below) can be directly accessed (via the
    :meth:`~obspy.sac.sacio.SacIO.__getattr__` method, see the link for
    an example).

    .. rubric::Description of attributes/header fields (based on SacIris_).

    .. _SacIris: http://www.iris.edu/manuals/sac/SAC_Manuals/FileFormatPt2.html

    ============ ==== =========================================================
    Field Name   Type Description
    ============ ==== =========================================================
    npts         N    Number of points per data component. [required]
    nvhdr        N    Header version number. Current value is the integer 6.
                      Older version data (NVHDR < 6) are automatically updated
                      when read into sac. [required]
    b            F    Beginning value of the independent variable. [required]
    e            F    Ending value of the independent variable. [required]
    iftype       I    Type of file [required]:
                          * ITIME {Time series file}
                          * IRLIM {Spectral file---real and imaginary}
                          * IAMPH {Spectral file---amplitude and phase}
                          * IXY {General x versus y data}
                          * IXYZ {General XYZ (3-D) file}
    leven        L    TRUE if data is evenly spaced. [required]
    delta        F    Increment between evenly spaced samples (nominal value).
                      [required]
    odelta       F    Observed increment if different from nominal value.
    idep         I    Type of dependent variable:
                          * IUNKN (Unknown)
                          * IDISP (Displacement in nm)
                          * IVEL (Velocity in nm/sec)
                          * IVOLTS (Velocity in volts)
                          * IACC (Acceleration in nm/sec/sec)
    scale        F    Multiplying scale factor for dependent variable
                      [not currently used]
    depmin       F    Minimum value of dependent variable.
    depmax       F    Maximum value of dependent variable.
    depmen       F    Mean value of dependent variable.
    nzyear       N    GMT year corresponding to reference (zero) time in file.
    nyjday       N    GMT julian day.
    nyhour       N    GMT hour.
    nzmin        N    GMT minute.
    nzsec        N    GMT second.
    nzmsec       N    GMT millisecond.
    iztype       I    Reference time equivalence:
                          * IUNKN (5): Unknown
                          * IB (9): Begin time
                          * IDAY (10): Midnight of reference GMT day
                          * IO (11): Event origin time
                          * IA (12): First arrival time
                          * ITn (13-22): User defined time pick n, n=0,9
    o            F    Event origin time (seconds relative to reference time.)
    a            F    First arrival time (seconds relative to reference time.)
    ka           K    First arrival time identification.
    f            F    Fini or end of event time (seconds relative to reference
                      time.)
    tn           F    User defined time {ai n}=0,9 (seconds picks or markers
                      relative to reference time).
    kt{ai n}     K    A User defined time {ai n}=0,9.  pick identifications
    kinst        K    Generic name of recording instrument
    iinst        I    Type of recording instrument. [currently not used]
    knetwk       K    Name of seismic network.
    kstnm        K    Station name.
    istreg       I    Station geographic region. [not currently used]
    stla         F    Station latitude (degrees, north positive)
    stlo         F    Station longitude (degrees, east positive).
    stel         F    Station elevation (meters). [not currently used]
    stdp         F    Station depth below surface (meters). [not currently
                      used]
    cmpaz        F    Component azimuth (degrees, clockwise from north).
    cmpinc       F    Component incident angle (degrees, from vertical).
    kcmpnm       K    Component name.
    lpspol       L    TRUE if station components have a positive polarity
                      (left-hand rule).
    kevnm        K    Event name.
    ievreg       I    Event geographic region. [not currently used]
    evla         F    Event latitude (degrees north positive).
    evlo         F    Event longitude (degrees east positive).
    evel         F    Event elevation (meters). [not currently used]
    evdp         F    Event depth below surface (meters). [not currently used]
    mag          F    Event magnitude.
    imagtyp      I    Magnitude type:
                          * IMB (Bodywave Magnitude)
                          * IMS (Surfacewave Magnitude)
                          * IML (Local Magnitude)
                          * IMW (Moment Magnitude)
                          * IMD (Duration Magnitude)
                          * IMX (User Defined Magnitude)
    imagsrc      I    Source of magnitude information:
                          * INEIC (National Earthquake Information Center)
                          * IPDE (Preliminary Determination of Epicenter)
                          * IISC (International Seismological Centre)
                          * IREB (Reviewed Event Bulletin)
                          * IUSGS (US Geological Survey)
                          * IBRK (UC Berkeley)
                          * ICALTECH (California Institute of Technology)
                          * ILLNL (Lawrence Livermore National Laboratory)
                          * IEVLOC (Event Location (computer program) )
                          * IJSOP (Joint Seismic Observation Program)
                          * IUSER (The individual using SAC2000)
                          * IUNKNOWN (unknown)
    ievtyp       I    Type of event:
                          * IUNKN (Unknown)
                          * INUCL (Nuclear event)
                          * IPREN (Nuclear pre-shot event)
                          * IPOSTN (Nuclear post-shot event)
                          * IQUAKE (Earthquake)
                          * IPREQ (Foreshock)
                          * IPOSTQ (Aftershock)
                          * ICHEM (Chemical explosion)
                          * IQB (Quarry or mine blast confirmed by quarry)
                          * IQB1 (Quarry/mine blast with designed shot
                            info-ripple fired)
                          * IQB2 (Quarry/mine blast with observed shot
                            info-ripple fired)
                          * IQMT (Quarry/mining-induced events:
                            tremors and rockbursts)
                          * IEQ (Earthquake)
                          * IEQ1 (Earthquakes in a swarm or aftershock
                            sequence)
                          * IEQ2 (Felt earthquake)
                          * IME (Marine explosion)
                          * IEX (Other explosion)
                          * INU (Nuclear explosion)
                          * INC (Nuclear cavity collapse)
                          * IO\_ (Other source of known origin)
                          * IR (Regional event of unknown origin)
                          * IT (Teleseismic event of unknown origin)
                          * IU (Undetermined or conflicting information)
                          * IOTHER (Other)
    nevid        N    Event ID (CSS 3.0)
    norid        N    Origin ID (CSS 3.0)
    nwfid        N    Waveform ID (CSS 3.0)
    khole        k    Hole identification if nuclear event.
    dist         F    Station to event distance (km).
    az           F    Event to station azimuth (degrees).
    baz          F    Station to event azimuth (degrees).
    gcarc        F    Station to event great circle arc length (degrees).
    lcalda       L    TRUE if DIST AZ BAZ and GCARC are to be calculated from
                      st event coordinates.
    iqual        I    Quality of data [not currently used]:
                          * IGOOD (Good data)
                          * IGLCH (Glitches)
                          * IDROP (Dropouts)
                          * ILOWSN (Low signal to noise ratio)
                          * IOTHER (Other)
    isynth       I    Synthetic data flag [not currently used]:
                          * IRLDTA (Real data)
                          * ????? (Flags for various synthetic seismogram
                            codes)
    user{ai n}   F    User defined variable storage area {ai n}=0,9.
    kuser{ai n}  K    User defined variable storage area {ai n}=0,2.
    lovrok       L    TRUE if it is okay to overwrite this file on disk.
    ============ ==== =========================================================
    """

    def __init__(self, filen=False, headonly=False, alpha=False,
                 debug_headers=False):
        self.byteorder = 'little'
        self.InitArrays()
        self.debug_headers = debug_headers
        if filen is False:
            return
        # parse Trace object if we get one
        if isinstance(filen, Trace):
            self.readTrace(filen)
            return
        if alpha:
            if headonly:
                self.ReadSacXYHeader(filen)
            else:
                self.ReadSacXY(filen)
        elif headonly:
            self.ReadSacHeader(filen)
        else:
            self.ReadSacFile(filen)

    def InitArrays(self):
        """
        Function to initialize the floating, character and integer
        header arrays (self.hf, self.hs, self.hi) with dummy values. This
        function is useful for writing SAC files from artificial data,
        thus the header arrays are not filled by a read method
        beforehand

        :return: Nothing
        """
        # The SAC header has 70 floats, then 40 integers, then 192 bytes
        # in strings. Store them in array (and convert the char to a
        # list). That's a total of 632 bytes.
        #
        # allocate the array for header floats
        self.hf = np.ndarray(70, dtype=native_str('<f4'))
        self.hf[:] = -12345.0
        #
        # allocate the array for header integers
        self.hi = np.ndarray(40, dtype=native_str('<i4'))
        self.hi[:] = -12345
        #
        # allocate the array for header characters
        self.hs = np.ndarray(24, dtype=native_str('|S8'))
        self.hs[:] = b'-12345  '  # setting default value
        # allocate the array for the points
        self.seis = np.ndarray([], dtype=native_str('<f4'))

    def fromarray(self, trace, begin=0.0, delta=1.0, distkm=0,
                  starttime=UTCDateTime("1970-01-01T00:00:00.000000")):
        """
        Create a SAC file from an numpy.ndarray instance

        >>> t = SacIO()
        >>> b = np.arange(10)
        >>> t.fromarray(b)
        >>> t.GetHvalue('npts')
        10
        """
        if not isinstance(trace, np.ndarray):
            raise SacError("input needs to be of instance numpy.ndarray")
        else:
            # Only copy the data if they are not of the required type
            self.seis = np.require(trace, native_str('<f4'))
        # convert start time to sac reference time, if it is not default
        if begin == -12345:
            reftime = starttime
        else:
            reftime = starttime - begin
        # if there are any micro-seconds, use begin to store them
        # integer arithmetic
        millisecond = reftime.microsecond // 1000
        # integer arithmetic
        microsecond = (reftime.microsecond - millisecond * 1000)
        if microsecond != 0:
            begin += microsecond * 1e-6
        # set a few values that are required to create a valid SAC-file
        self.SetHvalue('int1', 2)
        self.SetHvalue('cmpaz', 0)
        self.SetHvalue('cmpinc', 0)
        self.SetHvalue('nvhdr', 6)
        self.SetHvalue('leven', 1)
        self.SetHvalue('lpspol', 1)
        self.SetHvalue('lcalda', 0)
        self.SetHvalue('lovrok', 1)
        self.SetHvalue('nzyear', reftime.year)
        self.SetHvalue('nzjday', reftime.strftime("%j"))
        self.SetHvalue('nzhour', reftime.hour)
        self.SetHvalue('nzmin', reftime.minute)
        self.SetHvalue('nzsec', reftime.second)
        self.SetHvalue('nzmsec', millisecond)
        self.SetHvalue('kcmpnm', 'Z')
        self.SetHvalue('evla', 0)
        self.SetHvalue('evlo', 0)
        self.SetHvalue('iftype', 1)
        self.SetHvalue('npts', len(trace))
        self.SetHvalue('delta', delta)
        self.SetHvalue('b', begin)
        self.SetHvalue('e', begin + (len(trace) - 1) * delta)
        self.SetHvalue('iztype', 9)
        self.SetHvalue('dist', distkm)

    def GetHvalue(self, item):
        """
        Read SAC-header variable.

        :param item: header variable name (e.g. 'npts' or 'delta')

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> tr = SacIO('test.sac') # doctest: +SKIP
        >>> tr.GetHvalue('npts') # doctest: +SKIP
        100

        This is equivalent to:

        >>> SacIO().GetHvalueFromFile('test.sac','npts') # doctest: +SKIP
        100

        Or:

        >>> tr = SacIO('test.sac') # doctest: +SKIP
        >>> tr.npts # doctest: +SKIP
        100

        """
        key = item.lower()  # convert the item to lower case
        if key in FDICT:
            index = FDICT[key]
            return(self.hf[index])
        elif key in IDICT:
            index = IDICT[key]
            return(self.hi[index])
        elif key in SDICT:
            index = SDICT[key]
            if index == 0:
                myarray = self.hs[0].decode()
            elif index == 1:
                myarray = self.hs[1].decode() + self.hs[2].decode()
            else:
                myarray = self.hs[index + 1].decode()  # extra 1 from item #2
            return myarray
        else:
            raise SacError("Cannot find header entry for: " + item)

    def SetHvalue(self, item, value):
        """
        Assign new value to SAC-header variable.

        :param item: SAC-header variable name
        :param value: numeric or string value to be assigned to header
                      variable.

        >>> from obspy.sac import SacIO
        >>> tr = SacIO()
        >>> print(tr.GetHvalue('kstnm').strip())
        -12345
        >>> tr.SetHvalue('kstnm', 'STA_NEW')
        >>> print(tr.GetHvalue('kstnm').strip())
        STA_NEW
        """
        key = item.lower()  # convert the item to lower case
        #
        if key in FDICT:
            index = FDICT[key]
            self.hf[index] = float(value)
        elif key in IDICT:
            index = IDICT[key]
            self.hi[index] = int(value)
        elif key in SDICT:
            index = SDICT[key]
            if value:
                value = '%-8s' % value
            else:
                value = '-12345  '
            if index == 0:
                self.hs[0] = value.encode('ascii', 'strict')
            elif index == 1:
                value1 = '%-8s' % value[0:8]
                value2 = '%-8s' % value[8:16]
                self.hs[1] = value1.encode('ascii', 'strict')
                self.hs[2] = value2.encode('ascii', 'strict')

            else:
                self.hs[index + 1] = value.encode('ascii', 'strict')
        else:
            raise SacError("Cannot find header entry for: " + item)

    def IsSACfile(self, name, fsize=True, lenchk=False):
        """
        Test for a valid SAC file using arrays.

        :param f: filename (Sac binary).
        """
        try:
            npts = self.GetHvalue('npts')
        except:
            raise SacError("Unable to read number of points from header")
        if lenchk and npts != len(self.seis):
            raise SacError("Number of points in header and " +
                           "length of trace inconsistent!")
        if fsize:
            st = os.stat(name)  # file's size = st[6]
            sizecheck = st[6] - (632 + 4 * int(npts))
            # size check info
            if sizecheck != 0:
                msg = "File-size and theoretical size are inconsistent: %s\n" \
                      "Check that headers are consistent with time series."
                raise SacError(msg % name)
        # get the SAC file version number
        version = self.GetHvalue('nvhdr')
        if version < 0 or version > 20:
            raise SacError("Unknown header version!")
        if self.GetHvalue('delta') <= 0:
            raise SacError("Delta < 0 is not a valid header entry!")

    def ReadSacHeader(self, fname):
        """
        Reads only the header portion of a binary SAC-file.

        :param f: filename (SAC binary).

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> tr = SacIO() # doctest: +SKIP
        >>> tr.ReadSacHeader('test.sac') # doctest: +SKIP

        This is equivalent to:

        >>> tr = SacIO('test.sac', headonly=True)  # doctest: +SKIP
        """
        # check if file exists
        try:
            # open the file
            f = open(fname, 'rb')
        except IOError:
            raise SacIOError("No such file: " + fname)
        # --------------------------------------------------------------
        # parse the header
        #
        # The sac header has 70 floats, 40 integers, then 192 bytes
        #    in strings. Store them in array (and convert the char to a
        #    list). That's a total of 632 bytes.
        # --------------------------------------------------------------
        self.hf = frombuffer(f.read(4 * 70), dtype=native_str('<f4'))
        self.hi = frombuffer(f.read(4 * 40), dtype=native_str('<i4'))
        # read in the char values
        self.hs = frombuffer(f.read(24 * 8), dtype=native_str('|S8'))
        if len(self.hf) != 70 or len(self.hi) != 40 or len(self.hs) != 24:
            self.hf = self.hi = self.hs = None
            f.close()
            raise SacIOError("Cannot read all header values")
        try:
            self.IsSACfile(fname)
        except SacError as e:
            try:
                # if it is not a valid SAC-file try with big endian
                # byte order
                f.seek(0, 0)
                self.hf = frombuffer(f.read(4 * 70), dtype=native_str('>f4'))
                self.hi = frombuffer(f.read(4 * 40), dtype=native_str('>i4'))
                # read in the char values
                self.hs = frombuffer(f.read(24 * 8), dtype=native_str('|S8'))
                self.IsSACfile(fname)
                self.byteorder = 'big'
            except SacError as e:
                self.hf = self.hi = self.hs = None
                f.close()
                raise SacError(e)
        try:
            self._get_date()
        except SacError:
            warnings.warn('Cannot determine date')
        if self.GetHvalue('lcalda'):
            try:
                self._get_dist()
            except SacError:
                pass
        f.close()

    def WriteSacHeader(self, fname):
        """
        Writes an updated header to an
        existing binary SAC-file.

        :param f: filename (SAC binary).

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> tr = SacIO('test.sac') # doctest: +SKIP
        >>> tr.WriteSacBinary('test2.sac') # doctest: +SKIP
        >>> u = SacIO('test2.sac') # doctest: +SKIP
        >>> u.SetHvalue('kevnm','hullahulla') # doctest: +SKIP
        >>> u.WriteSacHeader('test2.sac') # doctest: +SKIP
        >>> u.GetHvalueFromFile('test2.sac',"kevnm") # doctest: +SKIP
        'hullahulla      '
        """
        # --------------------------------------------------------------
        # open the file
        #
        try:
            os.path.exists(fname)
        except IOError:
            warnings.warn("No such file: " + fname, category=Warning)
        else:
            f = open(fname, 'rb+')  # open file for modification
            f.seek(0, 0)  # set pointer to the file beginning
            try:
                # write the header
                f.write(self.hf.data)
                f.write(self.hi.data)
                f.write(self.hs.data)
            except Exception as e:
                f.close()
                raise SacError("Cannot write header to file: " + fname, e)
        f.close()

    def ReadSacFile(self, fname, fsize=True):
        """
        Read read in the header and data in a SAC file

        The header is split into three arrays - floats, ints, and strings and
        the data points are returned in the array seis

        :param f: filename (SAC binary)

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> tr = SacIO() # doctest: +SKIP
        >>> tr.ReadSacFile('test.sac') # doctest: +SKIP

        This is equivalent to:

        >>> tr = SacIO('test.sac')  # doctest: +SKIP
        """
        try:
            # open the file
            f = open(fname, 'rb')
        except IOError:
            raise SacIOError("No such file: " + fname)
        # --------------------------------------------------------------
        # parse the header
        #
        # The sac header has 70 floats, 40 integers, then 192 bytes
        #    in strings. Store them in array (and convert the char to a
        #    list). That's a total of 632 bytes.
        # --------------------------------------------------------------
        self.hf = frombuffer(f.read(4 * 70), dtype=native_str('<f4'))
        self.hi = frombuffer(f.read(4 * 40), dtype=native_str('<i4'))
        # read in the char values
        self.hs = frombuffer(f.read(24 * 8), dtype=native_str('|S8'))
        if len(self.hf) != 70 or len(self.hi) != 40 or len(self.hs) != 24:
            self.hf = self.hi = self.hs = None
            f.close()
            raise SacIOError("Cannot read all header values")
        # only continue if it is a SAC file
        try:
            self.IsSACfile(fname, fsize)
        except SacError:
            try:
                # if it is not a valid SAC-file try with big endian
                # byte order
                f.seek(0, 0)
                self.hf = frombuffer(f.read(4 * 70), dtype=native_str('>f4'))
                self.hi = frombuffer(f.read(4 * 40), dtype=native_str('>i4'))
                # read in the char values
                self.hs = frombuffer(f.read(24 * 8), dtype=native_str('|S8'))
                self.IsSACfile(fname, fsize)
                self.byteorder = 'big'
            except SacError as e:
                f.close()
                raise SacError(e)
        # --------------------------------------------------------------
        # read in the seismogram points
        # --------------------------------------------------------------
        # you just have to know it's in the 10th place
        # actually, it's in the SAC manual
        npts = self.hi[9]
        if self.byteorder == 'big':
            self.seis = frombuffer(f.read(npts * 4), dtype=native_str('>f4'))
        else:
            self.seis = frombuffer(f.read(npts * 4), dtype=native_str('<f4'))
        if len(self.seis) != npts:
            self.hf = self.hi = self.hs = self.seis = None
            f.close()
            raise SacIOError("Cannot read all data points")
        try:
            self._get_date()
        except SacError:
            warnings.warn('Cannot determine date')
        if self.GetHvalue('lcalda'):
            try:
                self._get_dist()
            except SacError:
                pass
        f.close()

    def ReadSacXY(self, fname):
        """
        Read SAC XY files (ascii)

        :param f: filename (SAC ascii).

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> tr = SacIO() # doctest: +SKIP
        >>> tr.ReadSacXY('testxy.sac') # doctest: +SKIP
        >>> tr.GetHvalue('npts') # doctest: +SKIP
        100

        This is equivalent to:

        >>> tr = SacIO('testxy.sac',alpha=True)  # doctest: +SKIP

        Reading only the header portion of alphanumeric SAC-files is currently
        not supported.
        """
        # open the file
        try:
            with open(fname, 'rb') as fh:
                data = fh.read()
        except IOError:
            raise SacIOError("No such file: " + fname)
        data = [_i.rstrip(b"\n\r") for _i in data.splitlines(True)]
        if len(data) < 14 + 8 + 8:
            raise SacIOError("%s is not a valid SAC file:" % fname)

        # --------------------------------------------------------------
        # parse the header
        #
        # The sac header has 70 floats, 40 integers, then 192 bytes
        #    in strings. Store them in array (and convert the char to a
        #    list). That's a total of 632 bytes.
        # --------------------------------------------------------------
        # read in the float values
        self.hf = np.array([i.split() for i in data[:14]],
                           dtype=native_str('<f4')).ravel()
        # read in the int values
        self.hi = np.array([i.split() for i in data[14: 14 + 8]],
                           dtype=native_str('<i4')).ravel()
        # reading in the string part is a bit more complicated
        # because every string field has to be 8 characters long
        # apart from the second field which is 16 characters long
        # resulting in a total length of 192 characters
        for i, j in enumerate(range(0, 24, 3)):
            line = data[14 + 8 + i]
            self.hs[j:j + 3] = np.fromstring(line, dtype=native_str('|S8'),
                                             count=3)
        # --------------------------------------------------------------
        # read in the seismogram points
        # --------------------------------------------------------------
        self.seis = np.array([i.split() for i in data[30:]],
                             dtype=native_str('<f4')).ravel()
        try:
            self._get_date()
        except SacError:
            warnings.warn('Cannot determine date')
        if self.GetHvalue('lcalda'):
            try:
                self._get_dist()
            except SacError:
                pass

    def ReadSacXYHeader(self, fname):
        """
        Read SAC XY files (ascii)

        :param f: filename (SAC ascii).

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> tr = SacIO() # doctest: +SKIP
        >>> tr.ReadSacXY('testxy.sac') # doctest: +SKIP
        >>> tr.GetHvalue('npts') # doctest: +SKIP
        100

        This is equivalent to:

        >>> tr = SacIO('testxy.sac',alpha=True)  # doctest: +SKIP

        Reading only the header portion of alphanumeric SAC-files is currently
        not supported.
        """
        # open the file
        try:
            with open(fname, 'rb') as fh:
                data = fh.read()
        except IOError:
            raise SacIOError("No such file: " + fname)
        data = [_i.rstrip(b"\n\r") for _i in data.splitlines(True)]
        if len(data) < 14 + 8 + 8:
            raise SacIOError("%s is not a valid SAC file:" % fname)

        # --------------------------------------------------------------
        # parse the header
        #
        # The sac header has 70 floats, 40 integers, then 192 bytes
        #    in strings. Store them in array (and convert the char to a
        #    list). That's a total of 632 bytes.
        # --------------------------------------------------------------
        # read in the float values
        self.hf = np.array([i.split() for i in data[:14]],
                           dtype=native_str('<f4')).ravel()
        # read in the int values
        self.hi = np.array([i.split() for i in data[14: 14 + 8]],
                           dtype=native_str('<i4')).ravel()
        # reading in the string part is a bit more complicated
        # because every string field has to be 8 characters long
        # apart from the second field which is 16 characters long
        # resulting in a total length of 192 characters
        for i, j in enumerate(range(0, 24, 3)):
            line = data[14 + 8 + i]
            self.hs[j:j + 3] = np.fromstring(line, dtype=native_str('|S8'),
                                             count=3)
        try:
            self.IsSACfile(fname, fsize=False)
        except SacError as e:
            raise SacError(e)
        try:
            self._get_date()
        except SacError:
            warnings.warn('Cannot determine date')
        if self.GetHvalue('lcalda'):
            try:
                self._get_dist()
            except SacError:
                pass

    def readTrace(self, trace):
        """
        Fill in SacIO object with data from obspy trace.
        Warning: Currently only the case of a previously empty SacIO object is
        safe!
        """
        # extracting relative SAC time as specified with b
        try:
            b = float(trace.stats['sac']['b'])
        except KeyError:
            b = 0.0
        # filling in SAC/sacio specific defaults
        self.fromarray(trace.data, begin=b, delta=trace.stats.delta,
                       starttime=trace.stats.starttime)
        # overwriting with ObsPy defaults
        for _j, _k in convert_dict.items():
            self.SetHvalue(_j, trace.stats[_k])
        # overwriting up SAC specific values
        # note that the SAC reference time values (including B and E) are
        # not used in here any more, they are already set by t.fromarray
        # and directly deduce from tr.starttime
        for _i in SAC_EXTRA:
            try:
                self.SetHvalue(_i, trace.stats.sac[_i])
            except (AttributeError, KeyError):
                pass
        return

    def WriteSacXY(self, ofname):
        """
        Write SAC XY file (ascii)

        :param f: filename (SAC ascii)

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> tr = SacIO('test.sac') # doctest: +SKIP
        >>> tr.WriteSacXY('test2.sac') # doctest: +SKIP
        >>> tr.IsValidXYSacFile('test2.sac') # doctest: +SKIP
        True
        """
        try:
            f = open(ofname, 'wb')
        except IOError:
            raise SacIOError("Cannot open file: " + ofname)
        # header
        try:
            np.savetxt(f, np.reshape(self.hf, (14, 5)),
                       fmt=native_str("%#15.7g%#15.7g%#15.7g%#15.7g%#15.7g"))
            np.savetxt(f, np.reshape(self.hi, (8, 5)),
                       fmt=native_str("%10d%10d%10d%10d%10d"))
            for i in range(0, 24, 3):
                f.write(self.hs[i:i + 3].data)
                f.write(b'\n')
        except:
            f.close()
            raise SacIOError("Cannot write header values: " + ofname)
        # traces
        npts = self.GetHvalue('npts')
        if npts == -12345 or npts == 0:
            f.close()
            return
        try:
            rows = npts // 5
            np.savetxt(f, np.reshape(self.seis[0:5 * rows], (rows, 5)),
                       fmt=native_str("%#15.7g%#15.7g%#15.7g%#15.7g%#15.7g"))
            np.savetxt(f, self.seis[5 * rows:], delimiter=b'\t')
        except:
            f.close()
            raise SacIOError("Cannot write trace values: " + ofname)
        f.close()

    def WriteSacBinary(self, ofname):
        """
        Write a SAC binary file using the head arrays and array seis.

        :param f: filename (SAC binary).

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> tr = SacIO('test.sac') # doctest: +SKIP
        >>> tr.WriteSacBinary('test2.sac') # doctest: +SKIP
        >>> os.stat('test2.sac')[6] == os.stat('test.sac')[6] # doctest: +SKIP
        True
        """
        try:
            f = open(ofname, 'wb+')
        except IOError:
            raise SacIOError("Cannot open file: " + ofname)
        try:
            self._chck_header()
            f.write(self.hf.data)
            f.write(self.hi.data)
            f.write(self.hs.data)
            f.write(self.seis.data)
        except Exception as e:
            f.close()
            msg = "Cannot write SAC-buffer to file: "
            raise SacIOError(msg, ofname, e)

        f.close()

    def PrintIValue(self, label='=', value=-12345):
        """
        Convenience function for printing undefined integer header values.
        """
        if value != -12345:
            print(label, value)

    def PrintFValue(self, label='=', value=-12345.0):
        """
        Convenience function for printing undefined float header values.
        """
        if value != -12345.0:
            print('%s %.8g' % (label, value))

    def PrintSValue(self, label='=', value='-12345'):
        """
        Convenience function for printing undefined string header values.
        """
        if value.find('-12345') == -1:
            print(label, value)

    def ListStdValues(self):  # h is a header list, s is a float list
        """
        Convenience function for printing common header values.

        :param: None
        :return: None

        >>> from obspy.sac import SacIO  # doctest: +SKIP
        >>> t = SacIO('test.sac')  # doctest: +SKIP
        >>> t.ListStdValues()  # doctest: +SKIP +NORMALIZE_WHITESPACE
        <BLANKLINE>
        Reference Time = 07/18/1978 (199) 8:0:0.0
        Npts  =  100
        Delta =  1
        Begin =  10
        End   =  109
        Min   =  -1
        Mean  =  8.7539462e-08
        Max   =  1
        Header Version =  6
        Station =  STA
        Channel =  Q
        Event       =  FUNCGEN: SINE

        If no header values are defined (i.e. all are equal 12345) than this
        function won't do anything.
        """
        #
        # Seismogram Info:
        #
        try:
            nzyear = self.GetHvalue('nzyear')
            nzjday = self.GetHvalue('nzjday')
            month = time.strptime(repr(nzyear) + " " + repr(nzjday),
                                  "%Y %j").tm_mon
            date = time.strptime(repr(nzyear) + " " + repr(nzjday),
                                 "%Y %j").tm_mday
            pattern = '\nReference Time = %2.2d/%2.2d/%d (%d) %d:%d:%d.%d'
            print(pattern % (month, date,
                             self.GetHvalue('nzyear'),
                             self.GetHvalue('nzjday'),
                             self.GetHvalue('nzhour'),
                             self.GetHvalue('nzmin'),
                             self.GetHvalue('nzsec'),
                             self.GetHvalue('nzmsec')))
        except ValueError:
            pass
        self.PrintIValue('Npts  = ', self.GetHvalue('npts'))
        self.PrintFValue('Delta = ', self.GetHvalue('delta'))
        self.PrintFValue('Begin = ', self.GetHvalue('b'))
        self.PrintFValue('End   = ', self.GetHvalue('e'))
        self.PrintFValue('Min   = ', self.GetHvalue('depmin'))
        self.PrintFValue('Mean  = ', self.GetHvalue('depmen'))
        self.PrintFValue('Max   = ', self.GetHvalue('depmax'))
        #
        self.PrintIValue('Header Version = ', self.GetHvalue('nvhdr'))
        #
        # station Info:
        #
        self.PrintSValue('Station = ', self.GetHvalue('kstnm'))
        self.PrintSValue('Channel = ', self.GetHvalue('kcmpnm'))
        self.PrintFValue('Station Lat  = ', self.GetHvalue('stla'))
        self.PrintFValue('Station Lon  = ', self.GetHvalue('stlo'))
        self.PrintFValue('Station Elev = ', self.GetHvalue('stel'))
        #
        # Event Info:
        #
        self.PrintSValue('Event       = ', self.GetHvalue('kevnm'))
        self.PrintFValue('Event Lat   = ', self.GetHvalue('evla'))
        self.PrintFValue('Event Lon   = ', self.GetHvalue('evlo'))
        self.PrintFValue('Event Depth = ', self.GetHvalue('evdp'))
        self.PrintFValue('Origin Time = ', self.GetHvalue('o'))
        #
        self.PrintFValue('Azimuth        = ', self.GetHvalue('az'))
        self.PrintFValue('Back Azimuth   = ', self.GetHvalue('baz'))
        self.PrintFValue('Distance (km)  = ', self.GetHvalue('dist'))
        self.PrintFValue('Distance (deg) = ', self.GetHvalue('gcarc'))

    def GetHvalueFromFile(self, thePath, theItem):
        """
        Quick access to a specific header item in specified file.

        :param f: filename (SAC binary)
        :type hn: str
        :param hn: header variable name

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> t = SacIO() # doctest: +SKIP
        >>> t.GetHvalueFromFile('test.sac','kcmpnm').rstrip() # doctest: +SKIP
        'Q'

        String header values have a fixed length of 8 or 16 characters. This
        can lead to errors for example if you concatenate strings and forget to
        strip off the trailing whitespace.
        """
        #
        #  Read in the Header
        #
        self.ReadSacHeader(thePath)
        #
        return(self.GetHvalue(theItem))

    def SetHvalueInFile(self, thePath, theItem, theValue):
        """
        Quick access to change a specific header item in a specified file.

        :param f: filename (SAC binary)
        :type hn: str
        :param hn: header variable name
        :type hv: str, float or int
        :param hv: header variable value (numeric or string value to be
            assigned to hn)
        :return: None

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> t = SacIO() # doctest: +SKIP
        >>> t.GetHvalueFromFile('test.sac','kstnm').rstrip() # doctest: +SKIP
        'STA'
        >>> t.SetHvalueInFile('test.sac','kstnm','blub') # doctest: +SKIP
        >>> t.GetHvalueFromFile('test.sac','kstnm').rstrip() # doctest: +SKIP
        'blub'
        """
        #
        #  Read in the Header
        #
        self.ReadSacHeader(thePath)
        #
        self.SetHvalue(theItem, theValue)
        self.WriteSacHeader(thePath)

    def IsValidSacFile(self, thePath):
        """
        Quick test for a valid SAC binary file (wraps 'IsSACfile').

        :param f: filename (SAC binary)
        :rtype: boolean (True or False)

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> SacIO().IsValidSacFile('test.sac') # doctest: +SKIP
        True
        >>> SacIO().IsValidSacFile('testxy.sac') # doctest: +SKIP
        False
        """
        #
        #  Read in the Header
        #
        try:
            self.ReadSacHeader(thePath)
        except SacError:
            return False
        except SacIOError:
            return False
        else:
            return True

    def IsValidXYSacFile(self, filename):
        """
        Quick test for a valid SAC ascii file.

        :param file: filename (SAC ascii)
        :rtype: boolean (True or False)

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> SacIO().IsValidXYSacFile('testxy.sac') # doctest: +SKIP
        True
        >>> SacIO().IsValidXYSacFile('test.sac') # doctest: +SKIP
        False
        """
        #
        #  Read in the Header
        #
        if _isText(filename):
            try:
                self.ReadSacXY(filename)
            except:
                return False
            return True
        else:
            return False

    def _get_date(self):
        """
        If date header values are set calculate date in julian seconds

        >>> t = SacIO()
        >>> t.fromarray(np.random.randn(100), delta=1.0,
        ...             starttime=UTCDateTime(1970,1,1))
        >>> t._get_date()
        >>> t.reftime.timestamp
        0.0
        >>> t.endtime.timestamp - t.reftime.timestamp
        100.0
        """
        # if any of the time-header values are still set to -12345 then
        # UTCDateTime raises an exception and reftime is set to 0.0
        try:
            ms = self.GetHvalue('nzmsec') * 1000
            yr = self.GetHvalue('nzyear')
            if 0 <= yr <= 99:
                warnings.warn(TWO_DIGIT_YEAR_MSG)
                yr += 1900
            self.reftime = UTCDateTime(year=yr,
                                       julday=self.GetHvalue('nzjday'),
                                       hour=self.GetHvalue('nzhour'),
                                       minute=self.GetHvalue('nzmin'),
                                       second=self.GetHvalue('nzsec'),
                                       microsecond=ms)
            b = float(self.GetHvalue('b'))
            if b != -12345.0:
                self.starttime = self.reftime + b
            else:
                self.starttime = self.reftime
            self.endtime = self.starttime + \
                self.GetHvalue('npts') * float(self.GetHvalue('delta'))
        except:
            try:
                self.reftime = UTCDateTime(0.0)
                b = float(self.GetHvalue('b'))
                if b != -12345.0:
                    self.starttime = self.reftime + b
                else:
                    self.starttime = self.reftime
                self.endtime = self.reftime + \
                    self.GetHvalue('npts') * float(self.GetHvalue('delta'))
            except:
                raise SacError("Cannot calculate date")

    def _chck_header(self):
        """
        If trace changed since read, adapt header values
        """
        self.seis = np.require(self.seis, native_str('<f4'))
        self.SetHvalue('npts', self.seis.size)
        if self.seis.size == 0:
            return
        self.SetHvalue('depmin', self.seis.min())
        self.SetHvalue('depmax', self.seis.max())
        self.SetHvalue('depmen', self.seis.mean())

    def _get_dist(self):
        """
        calculate distance from station and event coordinates

        >>> t = SacIO()
        >>> t.SetHvalue('evla',48.15)
        >>> t.SetHvalue('evlo',11.58333)
        >>> t.SetHvalue('stla',-41.2869)
        >>> t.SetHvalue('stlo',174.7746)
        >>> t._get_dist()
        >>> print('%.2f' % t.GetHvalue('dist'))
        18486.53
        >>> print('%.5f' % t.GetHvalue('az'))
        65.65415
        >>> print('%.4f' % t.GetHvalue('baz'))
        305.9755

        The original SAC-program calculates the distance assuming a
        average radius of 6371 km. Therefore, our routine should be more
        accurate.
        """
        eqlat = self.GetHvalue('evla')
        eqlon = self.GetHvalue('evlo')
        stlat = self.GetHvalue('stla')
        stlon = self.GetHvalue('stlo')
        d = self.GetHvalue('dist')
        if eqlat == -12345.0 or eqlon == -12345.0 or \
           stlat == -12345.0 or stlon == -12345.0:
            raise SacError('Insufficient information to calculate distance.')
        if d != -12345.0:
            raise SacError('Distance is already set.')
        dist, az, baz = gps2DistAzimuth(eqlat, eqlon, stlat, stlon)
        self.SetHvalue('dist', dist / 1000.)
        self.SetHvalue('az', az)
        self.SetHvalue('baz', baz)

    def swap_byte_order(self):
        """
        Swap byte order of SAC-file in memory.

        Currently seems to work only for conversion from big-endian to
        little-endian.

        :param: None
        :return: None

        >>> from obspy.sac import SacIO # doctest: +SKIP
        >>> t = SacIO('test.sac') # doctest: +SKIP
        >>> t.swap_byte_order() # doctest: +SKIP
        """
        if self.byteorder == 'big':
            bs = 'L'
        elif self.byteorder == 'little':
            bs = 'B'
        self.seis.byteswap(True)
        self.hf.byteswap(True)
        self.hi.byteswap(True)
        self.seis = self.seis.newbyteorder(bs)
        self.hf = self.hf.newbyteorder(bs)
        self.hi = self.hi.newbyteorder(bs)

    def __getattr__(self, hname):
        """
        convenience function to access header values

        :param hname: header variable name

        >>> tr = SacIO()
        >>> tr.fromarray(np.random.randn(100))
        >>> tr.npts == tr.GetHvalue('npts') # doctest: +SKIP
        True
        """
        return self.GetHvalue(hname)

    def get_obspy_header(self):
        """
        Return a dictionary that can be used as a header in creating a new
        :class:`~obspy.core.trace.Trace` object.
        Currently most likely an Exception will be raised if no SAC file was
        read beforehand!
        """
        header = {}
        # convert common header types of the ObsPy trace object
        for i, j in convert_dict.items():
            value = self.GetHvalue(i)
            if isinstance(value, (str, native_str)):
                null_term = value.find('\x00')
                if null_term >= 0:
                    value = value[:null_term]
                value = value.strip()
                if value == '-12345':
                    value = ''
            # fix for issue #156
            if i == 'delta':
                header['sampling_rate'] = \
                    np.float32(1.0) / np.float32(self.hf[0])
            else:
                header[j] = value
        if header['calib'] == -12345.0:
            header['calib'] = 1.0
        # assign extra header types of SAC
        header['sac'] = {}
        for i in SAC_EXTRA:
            header['sac'][i] = self.GetHvalue(i)
        # convert time to UTCDateTime
        header['starttime'] = self.starttime
        # always add the begin time (if it's defined) to get the given
        # SAC reference time, no matter which iztype is given
        # note that the B and E times should not be in the SAC_EXTRA
        # dictionary, as they would overwrite the self.fromarray which sets
        # them according to the starttime, npts and delta.
        header['sac']['b'] = float(self.GetHvalue('b'))
        header['sac']['e'] = float(self.GetHvalue('e'))
        # ticket #390
        if self.debug_headers:
            for i in ['nzyear', 'nzjday', 'nzhour', 'nzmin', 'nzsec', 'nzmsec',
                      'delta', 'scale', 'npts', 'knetwk', 'kstnm', 'kcmpnm']:
                header['sac'][i] = self.GetHvalue(i)
        return header


# UTILITIES
def attach_paz(tr, paz_file, todisp=False, tovel=False, torad=False,
               tohz=False):
    '''
    Attach tr.stats.paz AttribDict to trace from SAC paz_file

    This is experimental code, taken from
    obspy.gse2.libgse2.attach_paz and adapted to the SAC-pole-zero
    conventions. Especially the conversion from velocity to
    displacement and vice versa is still under construction. It works
    but I cannot guarantee that the values are correct. For more
    information on the SAC-pole-zero format see:
    http://www.iris.edu/files/sac-manual/commands/transfer.html. For a
    useful discussion on polezero files and transfer functions in
    general see:
    http://seis-uk.le.ac.uk/equipment/downloads/data_management/\
seisuk_instrument_resp_removal.pdf
    Also bear in mind that according to the SAC convention for
    pole-zero files CONSTANT is defined as:
    digitizer_gain*seismometer_gain*A0. This means that it does not
    have explicit information on the digitizer gain and seismometer
    gain which we therefore set to 1.0.

    Attaches to a trace a paz AttribDict containing poles zeros and gain.

    :param tr: An ObsPy :class:`~obspy.core.trace.Trace` object
    :param paz_file: path to pazfile or file pointer
    :param todisp: change a velocity transfer function to a displacement
                   transfer function by adding another zero
    :param tovel: change a displacement transfer function to a velocity
                  transfer function by removing one 0,0j zero
    :param torad: change to radians
    :param tohz: change to Hertz

    >>> from obspy import Trace
    >>> import io
    >>> tr = Trace()
    >>> f = io.StringIO("""ZEROS 3
    ... -5.032 0.0
    ... POLES 6
    ... -0.02365 0.02365
    ... -0.02365 -0.02365
    ... -39.3011 0.
    ... -7.74904 0.
    ... -53.5979 21.7494
    ... -53.5979 -21.7494
    ... CONSTANT 2.16e18""")
    >>> attach_paz(tr, f,torad=True)
    >>> for z in tr.stats.paz['zeros']:
    ...     print("%.2f %.2f" % (z.real, z.imag))
    -31.62 0.00
    0.00 0.00
    0.00 0.00
    '''

    poles = []
    zeros = []

    if isinstance(paz_file, (str, native_str)):
        paz_file = open(paz_file, 'r')

    while True:
        line = paz_file.readline()
        if not line:
            break
        # lines starting with * are comments
        if line.startswith('*'):
            continue
        if line.find('ZEROS') != -1:
            a = line.split()
            noz = int(a[1])
            for _k in range(noz):
                line = paz_file.readline()
                a = line.split()
                if line.find('POLES') != -1 or line.find('CONSTANT') != -1 or \
                   line.startswith('*') or not line:
                    while len(zeros) < noz:
                        zeros.append(complex(0, 0j))
                    break
                else:
                    zeros.append(complex(float(a[0]), float(a[1])))

        if line.find('POLES') != -1:
            a = line.split()
            nop = int(a[1])
            for _k in range(nop):
                line = paz_file.readline()
                a = line.split()
                if line.find('CONSTANT') != -1 or line.find('ZEROS') != -1 or \
                   line.startswith('*') or not line:
                    while len(poles) < nop:
                        poles.append(complex(0, 0j))
                    break
                else:
                    poles.append(complex(float(a[0]), float(a[1])))
        if line.find('CONSTANT') != -1:
            a = line.split()
            # in the observatory this is the seismometer gain [muVolt/nm/s]
            # the A0_normalization_factor is hardcoded to 1.0
            constant = float(a[1])
    paz_file.close()

    # To convert the velocity response to the displacement response,
    # multiplication with jw is used. This is equivalent to one more
    # zero in the pole-zero representation
    if todisp:
        zeros.append(complex(0, 0j))

    # To convert the displacement response to the velocity response,
    # division with jw is used. This is equivalent to one less zero
    # in the pole-zero representation
    if tovel:
        for i, zero in enumerate(list(zeros)):
            if zero == complex(0, 0j):
                zeros.pop(i)
                break
        else:
            raise Exception("Could not remove (0,0j) zero to change \
            displacement response to velocity response")

    # convert poles, zeros and gain in Hertz to radians
    if torad:
        tmp = [z * 2. * np.pi for z in zeros]
        zeros = tmp
        tmp = [p * 2. * np.pi for p in poles]
        poles = tmp
        # When extracting RESP files and SAC_PZ files
        # from a dataless SEED using the rdseed program
        # where the former is in Hz and the latter in radians,
        # there gains seem to be unaffected by this.
        # According to this document:
        # http://www.le.ac.uk/
        #         seis-uk/downloads/seisuk_instrument_resp_removal.pdf
        # the gain should also be converted when changing from
        # hertz to radians or vice versa. However, the rdseed programs
        # does not do this. I'm not entirely sure at this stage which one is
        # correct or if I have missed something. I've therefore decided
        # to leave it out for now, in order to stay compatible with the
        # rdseed program and the SAC program.
        # constant *= (2. * np.pi) ** 3

    # convert poles, zeros and gain in radian to Hertz
    if tohz:
        for i, z in enumerate(zeros):
            if abs(z) > 0.0:
                zeros[i] /= 2 * np.pi
        for i, p in enumerate(poles):
            if abs(p) > 0.0:
                poles[i] /= 2 * np.pi
        # constant /= (2. * np.pi) ** 3

    # fill up ObsPy Poles and Zeros AttribDict
    # In SAC pole-zero files CONSTANT is defined as:
    # digitizer_gain*seismometer_gain*A0

    tr.stats.paz = AttribDict()
    tr.stats.paz.seismometer_gain = 1.0
    tr.stats.paz.digitizer_gain = 1.0
    tr.stats.paz.poles = poles
    tr.stats.paz.zeros = zeros
    # taken from obspy.gse2.paz:145
    tr.stats.paz.sensitivity = tr.stats.paz.digitizer_gain * \
        tr.stats.paz.seismometer_gain
    tr.stats.paz.gain = constant


def attach_resp(tr, resp_file, todisp=False, tovel=False, torad=False,
                tohz=False):
    """
    Extract key instrument response information from a RESP file, which
    can be extracted from a dataless SEED volume by, for example, using
    the script obspy-dataless2resp or the rdseed program. At the moment,
    you have to determine yourself if the given response is for velocity
    or displacement and if the values are given in rad or Hz. This is
    still experimental code (see also documentation for
    :func:`obspy.sac.sacio.attach_paz`).
    Attaches to a trace a paz AttribDict containing poles, zeros, and gain.

    :param tr: An ObsPy :class:`~obspy.core.trace.Trace` object
    :param resp_file: path to RESP-file or file pointer
    :param todisp: change a velocity transfer function to a displacement
                   transfer function by adding another zero
    :param tovel: change a displacement transfer function to a velocity
                  transfer function by removing one 0,0j zero
    :param torad: change to radians
    :param tohz: change to Hertz

    >>> from obspy import Trace
    >>> tr = Trace()
    >>> respfile = os.path.join(os.path.dirname(__file__), 'tests', 'data',
    ...                         'RESP.NZ.CRLZ.10.HHZ')
    >>> attach_resp(tr, respfile, torad=True, todisp=False)
    >>> for k in sorted(tr.stats.paz):  # doctest: +NORMALIZE_WHITESPACE
    ...     print(k)
    digitizer_gain
    gain
    poles
    seismometer_gain
    sensitivity
    t_shift
    zeros
    >>> print(tr.stats.paz.poles)  # doctest: +SKIP
    [(-0.15931644664884559+0.15931644664884559j),
     (-0.15931644664884559-0.15931644664884559j),
     (-314.15926535897933+202.31856689118268j),
     (-314.15926535897933-202.31856689118268j)]
    """
    if not hasattr(resp_file, 'write'):
        resp_filep = open(resp_file, 'r')
    else:
        resp_filep = resp_file

    zeros_pat = r'B053F10-13'
    poles_pat = r'B053F15-18'
    a0_pat = r'B053F07'
    sens_pat = r'B058F04'
    t_shift_pat = r'B057F08'
    t_shift = 0.0
    poles = []
    zeros = []
    while True:
        line = resp_filep.readline()
        if not line:
            break
        if line.startswith(a0_pat):
            a0 = float(line.split(':')[1])
        if line.startswith(sens_pat):
            sens = float(line.split(':')[1])
        if line.startswith(poles_pat):
            tmp = line.split()
            poles.append(complex(float(tmp[2]), float(tmp[3])))
        if line.startswith(zeros_pat):
            tmp = line.split()
            zeros.append(complex(float(tmp[2]), float(tmp[3])))
        if line.startswith(t_shift_pat):
            t_shift += float(line.split(':')[1])
    constant = a0 * sens

    if not hasattr(resp_file, 'write'):
        resp_filep.close()

    if torad:
        tmp = [z * 2. * np.pi for z in zeros]
        zeros = tmp
        tmp = [p * 2. * np.pi for p in poles]
        poles = tmp

    if todisp:
        zeros.append(complex(0, 0j))

    # To convert the displacement response to the velocity response,
    # division with jw is used. This is equivalent to one less zero
    # in the pole-zero representation
    if tovel:
        for i, zero in enumerate(list(zeros)):
            if zero == complex(0, 0j):
                zeros.pop(i)
                break
        else:
            raise Exception("Could not remove (0,0j) zero to change \
            displacement response to velocity response")

    # convert poles, zeros and gain in radian to Hertz
    if tohz:
        for i, z in enumerate(zeros):
            if abs(z) > 0.0:
                zeros[i] /= 2 * np.pi
        for i, p in enumerate(poles):
            if abs(p) > 0.0:
                poles[i] /= 2 * np.pi
        constant /= (2. * np.pi) ** 3

    # fill up ObsPy Poles and Zeros AttribDict
    # In SAC pole-zero files CONSTANT is defined as:
    # digitizer_gain*seismometer_gain*A0

    tr.stats.paz = AttribDict()
    tr.stats.paz.seismometer_gain = sens
    tr.stats.paz.digitizer_gain = 1.0
    tr.stats.paz.poles = poles
    tr.stats.paz.zeros = zeros
    # taken from obspy.gse2.paz:145
    tr.stats.paz.sensitivity = tr.stats.paz.digitizer_gain * \
        tr.stats.paz.seismometer_gain
    tr.stats.paz.gain = constant
    tr.stats.paz.t_shift = t_shift


if __name__ == "__main__":
    import doctest
    doctest.testmod()
