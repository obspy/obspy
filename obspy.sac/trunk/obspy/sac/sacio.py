#!/usr/bin/env python
#-------------------------------------------------------------------
# Filename: sacio.py
#  Purpose: Read & Write Seismograms, Format SAC.
#   Author: Yannik Behr, C. J. Ammon's
#    Email: yannik.behr@vuw.ac.nz
#
# Copyright (C) 2008-2010 Yannik Behr, C. J. Ammon's
#--------------------------------------------------------------------

"""
An object-oriented version of C. J. Ammon's SAC I/O module.
:license: GNU Lesser General Public License, Version 3 (LGPLv3)
"""

from obspy.core import UTCDateTime
from obspy.core.util import NamedTemporaryFile, path
import numpy as np
import os
import time
import copy


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


class ReadSac(object):
    """
    Class for SAC file IO.

    Functions are given below, attributes/header
    fields (described below) can be directly accessed (via the
    :meth:`~obspy.sac.sacio.ReadSac.__getattr__` method, see the link for
    an example).

    Description of attributes/header fields (based on SacIris_).

    .. _SacIris: http://www.iris.edu/manuals/sac/SAC_Manuals/FileFormatPt2.html

    ============ === ==========================================================
    Field Name   TP  Description
    ============ === ==========================================================
    npts         N   Number of points per data component. [required]
    nvhdr        N   Header version number. Current value is the integer 6.
                     Older version data (NVHDR < 6) are automatically updated
                     when read into sac. [required]
    b            F   Beginning value of the independent variable. [required]
    e            F   Ending value of the independent variable. [required]
    iftype       I   Type of file [required]:
                     = ITIME {Time series file}
                     = IRLIM {Spectral file---real and imaginary}
                     = IAMPH {Spectral file---amplitude and phase}
                     = IXY {General x versus y data}
                     = IXYZ {General XYZ (3-D) file}
    leven        L   TRUE if data is evenly spaced. [required]
    delta        F   Increment between evenly spaced samples (nominal value).
                     [required]
    odelta       F   Observed increment if different from nominal value.
    idep         I   Type of dependent variable:
                     = IUNKN (Unknown)
                     = IDISP (Displacement in nm)
                     = IVEL (Velocity in nm/sec)
                     = IVOLTS (Velocity in volts)
                     = IACC (Acceleration in nm/sec/sec)
    scale        F   Multiplying scale factor for dependent variable
                     [not currently used]
    depmin       F   Minimum value of dependent variable.
    depmax       F   Maximum value of dependent variable.
    depmen       F   Mean value of dependent variable.
    nzyear       N   GMT year corresponding to reference (zero) time in file.
    nyjday       N   GMT julian day.
    nyhour       N   GMT hour.
    nzmin        N   GMT minute.
    nzsec        N   GMT second.
    nzmsec       N   GMT millisecond.
    iztype       I   Reference time equivalence:
                     = IUNKN (Unknown)
                     = IB (Begin time)
                     = IDAY (Midnight of refernece GMT day)
                     = IO (Event origin time)
                     = IA (First arrival time)
                     = ITn (User defined n=0 time 9) pick n
    o            F   Event origin time (seconds relative to reference time.)
    a            F   First arrival time (seconds relative to reference time.)
    ka           K   First arrival time identification.
    f            F   Fini or end of event time (seconds relative to reference
                     time.)
    tn           F   User defined time {ai n}=0,9 (seconds picks or markers
                     relative to reference time).
    kt{ai n}     K   A User defined time {ai n}=0,9.  pick identifications
    kinst        K   Generic name of recording instrument
    iinst        I   Type of recording instrument. [currently not used]
    knetwk       K   Name of seismic network.
    kstnm        K   Station name.
    istreg       I   Station geographic region. [not currently used]
    stla         F   Station latitude (degrees, north positive)
    stlo         F   Station longitude (degrees, east positive).
    stel         F   Station elevation (meters). [not currently used]
    stdp         F   Station depth below surface (meters). [not currently used]
    cmpaz        F   Component azimuth (degrees, clockwise from north).
    cmpinc       F   Component incident angle (degrees, from vertical).
    kcmpnm       K   Component name.
    lpspol       L   TRUE if station components have a positive polarity
                     (left-hand rule).
    kevnm        K   Event name.
    ievreg       I   Event geographic region. [not currently used]
    evla         F   Event latitude (degrees north positive).
    evlo         F   Event longitude (degrees east positive).
    evel         F   Event elevation (meters). [not currently used]
    evdp         F   Event depth below surface (meters). [not currently used]
    mag          F   Event magnitude.
    imagtyp      I   Magnitude type:
                     = IMB (Bodywave Magnitude)
                     = IMS (Surfacewave Magnitude)
                     = IML (Local Magnitude)
                     = IMW (Moment Magnitude)
                     = IMD (Duration Magnitude)
                     = IMX (User Defined Magnitude)
    imagsrc      I   Source of magnitude information:
                     = INEIC (National Earthquake Information Center)
                     = IPDE (Preliminary Determination of Epicenter)
                     = IISC (Internation Seismological Centre)
                     = IREB (Reviewed Event Bulletin)
                     = IUSGS (US Geological Survey)
                     = IBRK (UC Berkeley)
                     = ICALTECH (California Institute of Technology)
                     = ILLNL (Lawrence Livermore National Laboratory)
                     = IEVLOC (Event Location (computer program) )
                     = IJSOP (Joint Seismic Observation Program)
                     = IUSER (The individual using SAC2000)
                     = IUNKNOWN (unknown)
    ievtyp       I   Type of event:
                     = IUNKN (Unknown)
                     = INUCL (Nuclear event)
                     = IPREN (Nuclear pre-shot event)
                     = IPOSTN (Nuclear post-shot event)
                     = IQUAKE (Earthquake)
                     = IPREQ (Foreshock)
                     = IPOSTQ (Aftershock)
                     = ICHEM (Chemical explosion)
                     = IQB (Quarry or mine blast confirmed by quarry)
                     = IQB1 (Quarry/mine blast with designed shot info-ripple
                     fired)
                     = IQB2 (Quarry/mine blast with observed shot info-ripple
                     fired)
                     = IQMT (Quarry/mining-induced events:
                             tremors and rockbursts)
                     = IEQ (Earthquake)
                     = IEQ1 (Earthquakes in a swarm or aftershock sequence)
                     = IEQ2 (Felt earthquake)
                     = IME (Marine explosion)
                     = IEX (Other explosion)
                     = INU (Nuclear explosion)
                     = INC (Nuclear cavity collapse)
                     = IO_ (Other source of known origin)
                     = IR (Regional event of unknown origin)
                     = IT (Teleseismic event of unknown origin)
                     = IU (Undetermined or conflicting information)
                     = IOTHER (Other)
    nevid        N   Event ID (CSS 3.0)
    norid        N   Origin ID (CSS 3.0)
    nwfid        N   Waveform ID (CSS 3.0)
    khole        k   Hole identification if nuclear event.
    dist         F   Station to event distance (km).
    az           F   Event to station azimuth (degrees).
    baz          F   Station to event azimuth (degrees).
    gcarc        F   Station to event great circle arc length (degrees).
    lcalda       L   TRUE if DIST AZ BAZ and GCARC are to be calculated from st
                     event coordinates.
    iqual        I   Quality of data [not currently used]:
                     = IGOOD (Good data)
                     = IGLCH (Glitches)
                     = IDROP (Dropouts)
                     = ILOWSN (Low signal to noise ratio)
                     = IOTHER (Other)
    isynth       I   Synthetic data flag [not currently used]:
                     = IRLDTA (Real data)
                     = ????? (Flags for various synthetic seismogram codes)
    user{ai n}   F   User defined variable storage area {ai n}=0,9.
    kuser{ai n}  K   User defined variable storage area {ai n}=0,2.
    lovrok       L   TRUE if it is okay to overwrite this file on disk.
    ============ === ==========================================================
    """

    def __init__(self, filen=False, headonly=False, alpha=False):
        self.fdict = {'delta': 0, 'depmin': 1, 'depmax': 2, 'scale': 3,
                      'odelta': 4, 'b': 5, 'e': 6, 'o': 7, 'a': 8, 'int1': 9,
                      't0': 10, 't1': 11, 't2': 12, 't3': 13, 't4': 14,
                      't5': 15, 't6': 16, 't7': 17, 't8': 18, 't9': 19,
                      'f': 20, 'stla': 31, 'stlo': 32, 'stel': 33, 'stdp': 34,
                      'evla': 35, 'evlo': 36, 'evdp': 38, 'mag': 39,
                      'user0': 40, 'user1': 41, 'user2': 42, 'user3': 43,
                      'user4': 44, 'user5': 45, 'user6': 46, 'user7': 47,
                      'user8': 48, 'user9': 49, 'dist': 50, 'az': 51,
                      'baz': 52, 'gcarc': 53, 'depmen': 56, 'cmpaz': 57,
                      'cmpinc': 58}

        self.idict = {'nzyear': 0, 'nzjday': 1, 'nzhour': 2, 'nzmin': 3,
                      'nzsec': 4, 'nzmsec': 5, 'nvhdr': 6, 'norid': 7,
                      'nevid': 8, 'npts': 9, 'nwfid': 11,
                      'iftype': 15, 'idep': 16, 'iztype': 17, 'iinst': 19,
                      'istreg': 20, 'ievreg': 21, 'ievtype': 22, 'iqual': 23,
                      'isynth': 24, 'imagtyp': 25, 'imagsrc': 26,
                      'leven': 35, 'lpspol': 36, 'lovrok': 37,
                      'lcalda': 38}

        self.sdict = {'kstnm': 0, 'kevnm': 1, 'khole': 2, 'ko': 3, 'ka': 4,
                      'kt0': 5, 'kt1': 6, 'kt2': 7, 'kt3': 8, 'kt4': 9,
                      'kt5': 10, 'kt6': 11, 'kt7': 12, 'kt8': 13,
                      'kt9': 14, 'kf': 15, 'kuser0': 16, 'kuser1': 17,
                      'kuser2': 18, 'kcmpnm': 19, 'knetwk': 20,
                      'kdatrd': 21, 'kinst': 22}
        self.byteorder = 'little'
        self.InitArrays()
        self.headonly = headonly
        if filen:
            if alpha:
                self.__call__(filen, alpha=True)
            else:
                self.__call__(filen)

    def __call__(self, filename, alpha=False):
        if alpha:
            self.ReadSacXY(filename)
        elif self.headonly:
            self.ReadSacHeader(filename)
        else:
            self.ReadSacFile(filename)

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
        # in strings. Store them in array (an convert the char to a
        # list). That's a total of 632 bytes.
        #
        # allocate the array for header floats
        self.hf = np.ndarray(70, dtype='<f4')
        self.hf[:] = -12345.0
        #
        # allocate the array for header integers
        self.hi = np.ndarray(40, dtype='<i4')
        self.hi[:] = -12345
        #
        # allocate the array for header characters
        self.hs = np.ndarray(24, dtype='|S8')
        self.hs[:] = '-12345   ' # setting default value
        # allocate the array for the points
        self.seis = np.ndarray([], dtype='<f4')

    def fromarray(self, trace, begin=0.0, delta=1.0, distkm=0, 
                  starttime=UTCDateTime("1970-01-01T00:00:00.000000")):
        """
        Create a SAC file from an numpy.ndarray instance

        >>> t = ReadSac()
        >>> b = np.arange(10)
        >>> t.fromarray(b)
        >>> t.GetHvalue('npts')
        10
        """
        if not isinstance(trace, np.ndarray):
            raise SacError("input needs to be of instance numpy.ndarray")
        else:
            # Only copy the data if they are not of the required type
            self.seis = np.require(trace, '<f4')
        ### set a few values that are required to create a valid SAC-file
        self.SetHvalue('int1', 2)
        self.SetHvalue('cmpaz', 0)
        self.SetHvalue('cmpinc', 0)
        self.SetHvalue('nvhdr', 6)
        self.SetHvalue('leven', 1)
        self.SetHvalue('lpspol', 1)
        self.SetHvalue('lcalda', 0)
        self.SetHvalue('nzyear', starttime.year)
        self.SetHvalue('nzjday', starttime.strftime("%j"))
        self.SetHvalue('nzhour', starttime.hour)
        self.SetHvalue('nzmin', starttime.minute)
        self.SetHvalue('nzsec', starttime.second)
        self.SetHvalue('nzmsec', starttime.microsecond / 1e3)
        self.SetHvalue('kcmpnm', 'Z')
        self.SetHvalue('evla', 0)
        self.SetHvalue('evlo', 0)
        self.SetHvalue('iftype', 1)
        self.SetHvalue('npts', len(trace))
        self.SetHvalue('delta', delta)
        self.SetHvalue('b', begin)
        self.SetHvalue('e', begin + len(trace)*delta)
        self.SetHvalue('iztype', 9)
        self.SetHvalue('dist', distkm)

    def GetHvalue(self, item):
        """
        Read SAC-header variable.
        
        :param item: header variable name (e.g. 'npts' or 'delta')

        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> tr = ReadSac('test.sac') # doctest: +SKIP
        >>> tr.GetHvalue('npts') # doctest: +SKIP
        100

        This is equivalent to:
        
        >>> ReadSac().GetHvalueFromFile('test.sac','npts') # doctest: +SKIP
        100

        Or:
        
        >>> tr = ReadSac('test.sac') # doctest: +SKIP
        >>> tr.npts # doctest: +SKIP
        100

        """
        key = item.lower() # convert the item to lower case

        if key in self.fdict:
            index = self.fdict[key]
            return(self.hf[index])
        elif key in self.idict:
            index = self.idict[key]
            return(self.hi[index])
        elif key in self.sdict:
            index = self.sdict[key]
            if index == 0:
                myarray = self.hs[0]
            elif index == 1:
                myarray = self.hs[1] + self.hs[2]
            else:
                myarray = self.hs[index + 1] # extra 1 is from item #2
            return myarray
        else:
            raise SacError("Cannot find header entry for: ", item)

    def SetHvalue(self, item, value):
        """
        Assign new value to SAC-header variable.

        :param item: SAC-header variable name
        :param value: numeric or string value to be assigned to header-variable.

        >>> from obspy.sac import * # doctest: +SKIP
        >>> tr = ReadSac('test.sac') # doctest: +SKIP
        >>> tr.GetHvalue('kstnm') # doctest: +SKIP
        'STA     '
        >>> tr.SetHvalue('kstnm','STA_NEW') # doctest: +SKIP
        >>> tr.GetHvalue('kstnm') # doctest: +SKIP
        'STA_NEW '


        """
        key = item.lower() # convert the item to lower case
        #
        if key in self.fdict:
                index = self.fdict[key]
                self.hf[index] = float(value)
        elif key in self.idict:
                index = self.idict[key]
                self.hi[index] = int(value)
        elif key in self.sdict:
                index = self.sdict[key]
                value = '%-8s' % value
                if index == 0:
                        self.hs[0] = value
                elif index == 1:
                    value1 = '%-8s' % value[0:8]
                    value2 = '%-8s' % value[8:16]
                    self.hs[1] = value1
                    self.hs[2] = value2
                else:
                        self.hs[index + 1] = value
        else:
            raise SacError("Cannot find header entry for: ", item)

    def IsSACfile(self, name, fsize=True, lenchk=False):
        """
        Test for a valid SAC file using arrays.

        :param f: filename (Sac binary).
        
        """
        npts = self.GetHvalue('npts')
        if lenchk:
            if npts != len(self.seis):
                raise SacError("Number of points in header and" + \
                               "length of trace inconsistent!")
        if fsize:
            st = os.stat(name) #file's size = st[6]
            sizecheck = st[6] - (632 + 4 * npts)
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

        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> tr = ReadSac() # doctest: +SKIP
        >>> tr.ReadSacHeader('test.sac') # doctest: +SKIP

        This is equivalent to:
        
        >>> tr = ReadSac('test.sac',headonly=True)  # doctest: +SKIP

        """
        #### check if file exists
        try:
            #### open the file
            f = open(fname, 'r')
        except IOError:
            raise SacIOError("No such file:" + fname)
        try:
            #--------------------------------------------------------------
            # parse the header
            #
            # The sac header has 70 floats, 40 integers, then 192 bytes
            #    in strings. Store them in array (an convert the char to a
            #    list). That's a total of 632 bytes.
            #--------------------------------------------------------------
            self.hf = np.fromfile(f, dtype='<f4', count=70)
            self.hi = np.fromfile(f, dtype='<i4', count=40)
            # read in the char values
            self.hs = np.fromfile(f, dtype='|S8', count=24)
        except EOFError, e:
            self.hf = self.hi = self.hs = None
            f.close()
            raise SacIOError("Cannot read all header values: ", e)
        try:
            self.IsSACfile(fname)
        except SacError, e:
            try:
                # if it is not a valid SAC-file try with big endian
                # byte order
                f.seek(0, 0)
                self.hf = np.fromfile(f, dtype='>f4', count=70)
                self.hi = np.fromfile(f, dtype='>i4', count=40)
                # read in the char values
                self.hs = np.fromfile(f, dtype='|S8', count=24)
                self.IsSACfile(fname)
                self.byteorder = 'big'
            except SacError, e:
                self.hf = self.hi = self.hs = None
                f.close()
                raise SacError(e)
        try:
            self._get_date_()
        except SacError:
            pass
        if self.GetHvalue('lcalda'):
            try:
                self._get_dist_()
            except SacError:
                pass

    def WriteSacHeader(self, fname):
        """
        Writes an updated header to an
        existing binary SAC-file.

        :param f: filename (SAC binary).

        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> tr = ReadSac('test.sac') # doctest: +SKIP
        >>> tr.WriteSacBinary('test2.sac') # doctest: +SKIP
        >>> u = ReadSac('test2.sac') # doctest: +SKIP
        >>> u.SetHvalue('kevnm','hullahulla') # doctest: +SKIP
        >>> u.WriteSacHeader('test2.sac') # doctest: +SKIP
        >>> u.GetHvalueFromFile('test2.sac',"kevnm") # doctest: +SKIP
        'hullahulla      '
        """
        #--------------------------------------------------------------
        # open the file
        #
        try:
            os.path.exists(fname)
        except IOError:
            print "No such file:" + fname
        else:
            f = open(fname, 'r+') # open file for modification
            f.seek(0, 0) # set pointer to the file beginning
            try:
                # write the header
                self.hf.tofile(f)
                self.hi.tofile(f)
                self.hs.tofile(f)
            except Exception, e:
                f.close()
                raise SacError("Cannot write header to file: ", fname, '  ', e)

    def ReadSacFile(self, fname):
        """
        Read read in the header and data in a SAC file

        The header is split into three arrays - floats, ints, and strings and
        the data points are returned in the array seis

        
        :param f: filename (SAC binary)

        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> tr = ReadSac() # doctest: +SKIP
        >>> tr.ReadSacFile('test.sac') # doctest: +SKIP

        This is equivalent to:
        
        >>> tr = ReadSac('test.sac')  # doctest: +SKIP

        """
        try:
            #### open the file
            f = open(fname, 'rb')
        except IOError:
            raise SacIOError("No such file:" + fname)
        try:
            #--------------------------------------------------------------
            # parse the header
            #
            # The sac header has 70 floats, 40 integers, then 192 bytes
            #    in strings. Store them in array (an convert the char to a
            #    list). That's a total of 632 bytes.
            #--------------------------------------------------------------
            self.hf = np.fromfile(f, dtype='<f4', count=70)
            self.hi = np.fromfile(f, dtype='<i4', count=40)
            # read in the char values
            self.hs = np.fromfile(f, dtype='|S8', count=24)
        except EOFError, e:
            raise SacIOError("Cannot read any or no header values: ", e)
        ##### only continue if it is a SAC file
        try:
            self.IsSACfile(fname)
        except SacError:
            try:
                # if it is not a valid SAC-file try with big endian
                # byte order
                f.seek(0, 0)
                self.hf = np.fromfile(f, dtype='>f4', count=70)
                self.hi = np.fromfile(f, dtype='>i4', count=40)
                # read in the char values
                self.hs = np.fromfile(f, dtype='|S8', count=24)
                self.IsSACfile(fname)
                self.byteorder = 'big'
            except SacError, e:
                raise SacError(e)
        #--------------------------------------------------------------
        # read in the seismogram points
        #--------------------------------------------------------------
        # you just have to know it's in the 10th place
        # actually, it's in the SAC manual
        npts = self.hi[9]
        try:
            if self.byteorder == 'big':
                self.seis = np.fromfile(f, dtype='>f4', count=npts)
            else:
                self.seis = np.fromfile(f, dtype='<f4', count=npts)
        except EOFError, e:
            self.hf = self.hi = self.hs = self.seis = None
            f.close()
            msg = "Cannot read any or only some data points: "
            raise SacIOError(msg, e)
        try:
            self._get_date_()
        except SacError:
            pass
        if self.GetHvalue('lcalda'):
            try:
                self._get_dist_()
            except SacError:
                pass

    def ReadSacXY(self, fname):
        """
        Read SAC XY files (ascii)

        :param f: filename (SAC ascii).

        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> tr = ReadSac() # doctest: +SKIP
        >>> tr.ReadSacXY('testxy.sac') # doctest: +SKIP
        >>> tr.GetHvalue('npts') # doctest: +SKIP
        100

        This is equivalent to:
        
        >>> tr = ReadSac('testxy.sac',alpha=True) # doctest: +SKIP 

        Reading only the header portion of alphanumeric SAC-files is currently not supported.
        """
        ###### open the file
        try:
            f = open(fname, 'r')
        except IOError:
            raise SacIOError("No such file:" + fname)
        else:
            try:
                #--------------------------------------------------------------
                # parse the header
                #
                # The sac header has 70 floats, 40 integers, then 192 bytes
                #    in strings. Store them in array (an convert the char to a
                #    list). That's a total of 632 bytes.
                #--------------------------------------------------------------
                # read in the float values
                self.hf = np.fromfile(f, dtype='<f4', count=70, sep=" ")
                # read in the int values
                self.hi = np.fromfile(f, dtype='<i4', count=40, sep=" ")
                # reading in the string part is a bit more complicated
                # because every string field has to be 8 characters long
                # apart from the second field which is 16 characters long
                # resulting in a total length of 192 characters
                for i in xrange(0, 24, 3):
                    self.hs[i:i + 3] = np.fromfile(f, dtype='|S8', count=3)
                    f.readline() # strip the newline
                #--------------------------------------------------------------
                # read in the seismogram points
                #--------------------------------------------------------------
                self.seis = np.loadtxt(f, dtype='<f4').ravel()
            except IOError, e:
                self.hf = self.hs = self.hi = self.seis = None
                f.close()
                raise SacIOError("%s is not a valid SAC file:" % fname, e)
            try:
                self.IsSACfile(fname, fsize=False, lenchk=True)
            except SacError, e:
                f.close()
                raise SacError(e)
            else:
                try:
                    self._get_date_()
                except SacError:
                    pass
                if self.GetHvalue('lcalda'):
                    try:
                        self._get_dist_()
                    except SacError:
                        pass

    def WriteSacXY(self, ofname):
        """
        Write SAC XY file (ascii)

        :param f: filename (SAC ascii)

        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> tr = ReadSac('test.sac') # doctest: +SKIP
        >>> tr.WriteSacXY('test2.sac') # doctest: +SKIP
        >>> tr.IsValidXYSacFile('test2.sac') # doctest: +SKIP
        True


        """
        try:
            f = open(ofname, 'w')
        except IOError:
            raise SacIOError("Can't open file:" + ofname)
        else:
            try:
                np.savetxt(f, np.reshape(self.hf, (14, 5)),
                           fmt="%-8.6g %-8.6g %-8.6g %-8.6g %-8.6g")
                np.savetxt(f, np.reshape(self.hi, (8, 5)),
                           fmt="%-8.6g %-8.6g %-8.6g %-8.6g %-8.6g")
                for i in xrange(0, 24, 3):
                    self.hs[i:i + 3].tofile(f)
                    f.write('\n')
            except:
                raise SacIOError("Can't write header values:" + ofname)
            else:
                try:
                    npts = self.GetHvalue('npts')
                    rows = npts / 5
                    np.savetxt(f, np.reshape(self.seis[0:5 * rows], (rows, 5)),
                               fmt="%15.7g\t%15.7g\t%15.7g\t%15.7g\t%15.7g")
                    np.savetxt(f, self.seis[5 * rows:], delimiter='\t')
                except:
                    raise SacIOError("Can't write trace values:" + ofname)

    def WriteSacBinary(self, ofname):
        """
        Write a SAC binary file using the head arrays and array seis.

        :param f: filename (SAC binary).

        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> tr = ReadSac('test.sac') # doctest: +SKIP
        >>> tr.WriteSacBinary('test2.sac') # doctest: +SKIP
        >>> os.stat('test2.sac')[6] == os.stat('test.sac')[6] # doctest: +SKIP
        True
        """
        try:
            f = open(ofname, 'wb+')
        except IOError:
            raise SacIOError("Cannot open file: ", ofname)
        else:
            try:
                self._chck_header_()
                self.hf.tofile(f)
                self.hi.tofile(f)
                self.hs.tofile(f)
                self.seis.tofile(f)
            except Exception, e:
                f.close()
                msg = "Cannot write SAC-buffer to file: "
                raise SacIOError(msg, ofname, e)

    def PrintIValue(self, label='=', value= -12345):
        """
        Convenience function for printing undefined integer header values.
        """
        if value != -12345:
            print label, value

    def PrintFValue(self, label='=', value= -12345.0):
        """
        Convenience function for printing undefined float header values.
        """
        if value != -12345.0:
            print '%s %.8g' % (label, value)

    def PrintSValue(self, label='=', value='-12345'):
        """
        Convenience function for printing undefined string header values.
        """
        if value.find('-12345') == -1:
            print label, value

    def ListStdValues(self): # h is a header list, s is a float list
        """
        Convenience function for printing common header values.

        :param: None
        :return: None

        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> t = ReadSac('test.sac') # doctest: +SKIP
        >>> t.ListStdValues() # doctest: +SKIP
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

        If no header values are defined (i.e. all are equal 12345) than this function
        won't do anything.
        """
        #
        # Seismogram Info:
        #
        try:
            nzyear = self.GetHvalue('nzyear')
            nzjday = self.GetHvalue('nzjday')
            month = time.strptime(`nzyear` + " " + `nzjday`, "%Y %j").tm_mon
            date = time.strptime(`nzyear` + " " + `nzjday`, "%Y %j").tm_mday
            pattern = '\nReference Time = %2.2d/%2.2d/%d (%d) %d:%d:%d.%d'
            print pattern % (month, date,
                             self.GetHvalue('nzyear'),
                             self.GetHvalue('nzjday'),
                             self.GetHvalue('nzhour'),
                             self.GetHvalue('nzmin'),
                             self.GetHvalue('nzsec'),
                             self.GetHvalue('nzmsec'))
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
        :type hn: string
        :param hn: header variable name

        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> t = ReadSac() # doctest: +SKIP
        >>> t.GetHvalueFromFile('test.sac','kcmpnm').rstrip() # doctest: +SKIP
        'Q'

        String header values have a fixed length of 8 or 16 characters. This can lead to errors
        for example if you concatenate strings and forget to strip off the trailing whitespace.
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
        :type hn: string
        :param hn: header variable name
        :type hv: string, float or integer
        :param hv: header variable value (numeric or string value to be assigned to hn)
        :return: None

        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> t = ReadSac() # doctest: +SKIP
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
        
        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> ReadSac().IsValidSacFile('test.sac') # doctest: +SKIP
        True
        >>> ReadSac().IsValidSacFile('testxy.sac') # doctest: +SKIP
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
        
        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> ReadSac().IsValidXYSacFile('testxy.sac') # doctest: +SKIP
        True
        >>> ReadSac().IsValidXYSacFile('test.sac') # doctest: +SKIP
        False

        """
        #
        #  Read in the Header
        #
        try:
            self.ReadSacXY(filename)
        except:
            return False
        else:
            return True

    def _get_date_(self):
        """
        If date header values are set calculate date in julian seconds

        >>> t = ReadSac('test.sac') # doctest: +SKIP
        >>> t.starttime.timestamp # doctest: +SKIP
        269596800.0
        >>> t.endtime.timestamp - t.starttime.timestamp # doctest: +SKIP
        100.0
        """
        ### if any of the time-header values are still set to
        ### -12345 then UTCDateTime raises an exception and
        ### starttime is set to 0.0
        try:
            ms = self.GetHvalue('nzmsec') * 1000
            self.starttime = UTCDateTime(year=self.GetHvalue('nzyear'),
                                         julday=self.GetHvalue('nzjday'),
                                         hour=self.GetHvalue('nzhour'),
                                         minute=self.GetHvalue('nzmin'),
                                         second=self.GetHvalue('nzsec'),
                                         microsecond=ms)
            self.endtime = self.starttime + \
                self.GetHvalue('npts') * float(self.GetHvalue('delta'))
        except:
            try:
                self.starttime = UTCDateTime(0.0)
                self.endtime = self.starttime + \
                    self.GetHvalue('npts') * float(self.GetHvalue('delta'))
            except:
                raise SacError("Cannot calculate date")

    def _chck_header_(self):
        """
        If trace changed since read, adapt header values
        """
        self.seis = np.require(self.seis, '<f4')
        self.SetHvalue('npts', self.seis.size)
        self.SetHvalue('depmin', self.seis.min())
        self.SetHvalue('depmax', self.seis.max())
        self.SetHvalue('depmen', self.seis.mean())

    def _get_dist_(self):
        """
        calculate distance from station and event coordinates

        >>> t = ReadSac('test.sac') # doctest: +SKIP
        >>> t.SetHvalue('evla',48.15) # doctest: +SKIP
        >>> t.SetHvalue('evlo',11.58333) # doctest: +SKIP
        >>> t.SetHvalue('stla',-41.2869) # doctest: +SKIP
        >>> t.SetHvalue('stlo',174.7746) # doctest: +SKIP
        >>> t._get_dist_() # doctest: +SKIP
        >>> print round(t.GetHvalue('dist'), 2) # doctest: +SKIP
        18486.53
        >>> print round(t.GetHvalue('az'), 5) # doctest: +SKIP
        65.65415
        >>> print round(t.GetHvalue('baz'), 4) # doctest: +SKIP
        305.9755

        The original SAC-program calculates the distance assuming a
        average radius of 6371 km. Therefore, our routine should be more
        accurate.
        """
        # Avoid top level dependency on obspy.signal. Thus if obspy.signal
        # is not allow only this function will not work, not the whole
        # module
        try:
            from obspy.signal import rotate
        except ImportError, e:
            print "ERROR: obspy.signal is needed for this function " + \
                  "and is not installed"
            raise e
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
        dist, az, baz = rotate.gps2DistAzimuth(eqlat, eqlon, stlat, stlon)
        self.SetHvalue('dist', dist / 1000.)
        self.SetHvalue('az', az)
        self.SetHvalue('baz', baz)

    def swap_byte_order(self):
        """
        Swap byte order of SAC-file in memory.

        Currently seems to work only for conversion from big-endian to little-endian.

        :param: None
        :return: None
        
        >>> from obspy.sac import ReadSac # doctest: +SKIP
        >>> t = ReadSac('test.sac') # doctest: +SKIP
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

        >>> tr = ReadSac('test.sac') # doctest: +SKIP
        >>> tr.npts == tr.GetHvalue('npts') # doctest: +SKIP
        True
        """
        return self.GetHvalue(hname)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

