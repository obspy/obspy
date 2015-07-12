# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  #NOQA

MODULE_DOCSTRING = """
SAC header specification, including documentation.

Header names, order, types, and nulls, as well as allowed enumerated values,
are specified here.  Header name strings, and their array order are contained
in separate float, int, and string tuples.  Enumerated values, and their
allowed string and integer values, are in dictionaries.  Header value
documentation is in a dictionary, for reuse throughout the package.

"""

# header documentation is large and used in several places, so we just write
# it once here and distributed it as needed.
DOC = {'npts': 'N    Number of points per data component. [required]',
       'nvhdr': '''N    Header version number. Current value is the integer 6.
                  Older version data (NVHDR < 6) are automatically updated
                  when read into sac. [required]''',
       'b': 'F    Beginning value of the independent variable. [required]',
       'e': 'F    Ending value of the independent variable. [required]',
       'iftype': '''I    Type of file [required]:
                  * ITIME {Time series file}
                  * IRLIM {Spectral file---real and imaginary}
                  * IAMPH {Spectral file---amplitude and phase}
                  * IXY {General x versus y data}
                  * IXYZ {General XYZ (3-D) file}''',
       'leven': 'L    TRUE if data is evenly spaced. [required]',
       'delta': 'F    Increment between evenly spaced samples (nominal value). [required]',
       'odelta': 'F    Observed increment if different from nominal value.',
       'idep': '''I    Type of dependent variable:
                  * IUNKN (Unknown)
                  * IDISP (Displacement in nm)
                  * IVEL (Velocity in nm/sec)
                  * IVOLTS (Velocity in volts)
                  * IACC (Acceleration in nm/sec/sec)''',
       'scale': 'F    Multiplying scale factor for dependent variable [not currently used]',
       'depmin': 'F    Minimum value of dependent variable.',
       'depmax': 'F    Maximum value of dependent variable.',
       'depmen': 'F    Mean value of dependent variable.',
       'nzyear': 'N    GMT year corresponding to reference (zero) time in file.',
       'nzjday': 'N    GMT julian day.',
       'nzhour': 'N    GMT hour.',
       'nzmin': 'N    GMT minute.',
       'nzsec': 'N    GMT second.',
       'nzmsec': 'N    GMT millisecond.',
       'iztype': '''I    Reference time equivalence:
                  * IUNKN (5): Unknown
                  * IB (9): Begin time
                  * IDAY (10): Midnight of reference GMT day
                  * IO (11): Event origin time
                  * IA (12): First arrival time
                  * ITn (13-22): User defined time pick n, n=0,9''',
       'o': 'F    Event origin time (seconds relative to reference time.)',
       'ko': 'K    Event origin time identification.',
       'a': 'F    First arrival time (seconds relative to reference time.)',
       'ka': 'K    First arrival time identification.',
       'f': 'F    Fini or end of event time (seconds relative to reference time.)',
       'kf': 'F    Fini or end of event time identification.',
       't0': 'F    User defined time (seconds picks or markers relative to reference time).',
       't1': 'F    User defined time (seconds picks or markers relative to reference time).',
       't2': 'F    User defined time (seconds picks or markers relative to reference time).',
       't3': 'F    User defined time (seconds picks or markers relative to reference time).',
       't4': 'F    User defined time (seconds picks or markers relative to reference time).',
       't5': 'F    User defined time (seconds picks or markers relative to reference time).',
       't6': 'F    User defined time (seconds picks or markers relative to reference time).',
       't7': 'F    User defined time (seconds picks or markers relative to reference time).',
       't8': 'F    User defined time (seconds picks or markers relative to reference time).',
       't9': 'F    User defined time (seconds picks or markers relative to reference time).',
       'kt0': 'F    User defined time pick identification.',
       'kt1': 'F    User defined time pick identification.',
       'kt2': 'F    User defined time pick identification.',
       'kt3': 'F    User defined time pick identification.',
       'kt4': 'F    User defined time pick identification.',
       'kt5': 'F    User defined time pick identification.',
       'kt6': 'F    User defined time pick identification.',
       'kt7': 'F    User defined time pick identification.',
       'kt8': 'F    User defined time pick identification.',
       'kt9': 'F    User defined time pick identification.',
       'kinst': 'K    Generic name of recording instrument',
       'iinst': 'I    Type of recording instrument. [currently not used]',
       'knetwk': 'K    Name of seismic network.',
       'kstnm': 'K    Station name.',
       'istreg': 'I    Station geographic region. [not currently used]',
       'stla': 'F    Station latitude (degrees, north positive)',
       'stlo': 'F    Station longitude (degrees, east positive).',
       'stel': 'F    Station elevation (meters). [not currently used]',
       'stdp': 'F    Station depth below surface (meters). [not currently used]',
       'cmpaz': 'F    Component azimuth (degrees, clockwise from north).',
       'cmpinc': 'F    Component incident angle (degrees, from vertical).',
       'kcmpnm': 'K    Component name.',
       'lpspol': 'L    TRUE if station components have a positive polarity (left-hand rule).',
       'kevnm': 'K    Event name.',
       'ievreg': 'I    Event geographic region. [not currently used]',
       'evla': 'F    Event latitude (degrees north positive).',
       'evlo': 'F    Event longitude (degrees east positive).',
       'evel': 'F    Event elevation (meters). [not currently used]',
       'evdp': 'F    Event depth below surface (meters). [not currently used]',
       'mag': 'F    Event magnitude.',
       'imagtyp': '''I    Magnitude type:
                  * IMB (52): Bodywave Magnitude
                  * IMS (53): Surfacewave Magnitude
                  * IML (54): Local Magnitude
                  * IMW (55): Moment Magnitude
                  * IMD (56): Duration Magnitude
                  * IMX (57): User Defined Magnitude''',
       'imagsrc': '''I    Source of magnitude information:
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
                  * IUNKNOWN (unknown)''',
       'ievtyp': '''I    Type of event:
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
                  * IO_ (Other source of known origin)
                  * IR (Regional event of unknown origin)
                  * IT (Teleseismic event of unknown origin)
                  * IU (Undetermined or conflicting information)
                  * IOTHER (Other)''',
       'nevid': 'N    Event ID (CSS 3.0)',
       'norid': 'N    Origin ID (CSS 3.0)',
       'nwfid': 'N    Waveform ID (CSS 3.0)',
       'khole': 'k    Hole identification if nuclear event.',
       'dist': 'F    Station to event distance (km).',
       'az': 'F    Event to station azimuth (degrees).',
       'baz': 'F    Station to event azimuth (degrees).',
       'gcarc': 'F    Station to event great circle arc length (degrees).',
       'lcalda': 'L    TRUE if DIST AZ BAZ and GCARC are to be calculated from st event coordinates.',
       'iqual': '''I    Quality of data [not currently used]:
                  * IGOOD (Good data)
                  * IGLCH (Glitches)
                  * IDROP (Dropouts)
                  * ILOWSN (Low signal to noise ratio)
                  * IOTHER (Other)''',
       'isynth': '''I    Synthetic data flag [not currently used]:
                  * IRLDTA (Real data)
                  * ????? (Flags for various synthetic seismogram codes)''',
       'user0': 'F    User defined variable storage area 0.',
       'user1': 'F    User defined variable storage area 1.',
       'user2': 'F    User defined variable storage area 2.',
       'user3': 'F    User defined variable storage area 3.',
       'user4': 'F    User defined variable storage area 4.',
       'user5': 'F    User defined variable storage area 5.',
       'user6': 'F    User defined variable storage area 6.',
       'user7': 'F    User defined variable storage area 7.',
       'user8': 'F    User defined variable storage area 8.',
       'user9': 'F    User defined variable storage area 9.',
       'kuser0': 'K    User defined variable storage area 0.',
       'kuser1': 'K    User defined variable storage area 1.',
       'kuser2': 'K    User defined variable storage area 2.',
       'lovrok': 'L    TRUE if it is okay to overwrite this file on disk.'}

HEADER_DOCSTRING = """
============ ==== =========================================================
Field Name   Type Description
============ ==== =========================================================
""" + \
'\n'.join(["{:10.10s} = {}".format(_hdr, _doc) for _hdr, _doc in sorted(DOC.items())]) + \
"\n============ ==== ========================================================="

# Module documentation string
__doc__ = MODULE_DOCSTRING + HEADER_DOCSTRING

# ------------ NULL VALUES ----------------------------------------------------
FNULL = -12345.0
INULL = -12345
SNULL = b'-12345  '

# ------------ HEADER NAMES, TYPES, ARRAY POSITIONS ---------------------------
# these are useful b/c they can be used forwards or backwards, like:
# FLOADHDRS.index('az') is 40, and FLOATHDRS[40] is 'az'.
FLOATHDRS = ('delta', 'depmin', 'depmax', 'scale', 'odelta', 'b', 'e', 'o',
             'a', 'internal0', 't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7',
             't8', 't9', 'f', 'resp0', 'resp1', 'resp2', 'resp3', 'resp4',
             'resp5', 'resp6', 'resp7', 'resp8', 'resp9', 'stla', 'stlo',
             'stel', 'stdp', 'evla', 'evlo', 'evel', 'evdp', 'mag', 'user0',
             'user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7',
             'user8', 'user9', 'dist', 'az', 'baz', 'gcarc', 'internal1',
             'internal2', 'depmen', 'cmpaz', 'cmpinc', 'xminimum', 'xmaximum',
             'yminimum', 'ymaximum', 'unused6', 'unused7', 'unused8',
             'unused9', 'unused10', 'unused11', 'unused12')

INTHDRS = ('nzyear', 'nzjday', 'nzhour', 'nzmin', 'nzsec', 'nzmsec', 'nvhdr',
           'norid', 'nevid', 'npts', 'internal3', 'nwfid', 'nxsize', 'nysize',
           'unused13', 'iftype', 'idep', 'iztype', 'unused14', 'iinst',
           'istreg', 'ievreg', 'ievtyp', 'iqual', 'isynth', 'imagtyp',
           'imagsrc', 'unused15', 'unused16', 'unused17', 'unused18',
           'unused19', 'unused20', 'unused21', 'unused22', 'leven', 'lpspol',
           'lovrok', 'lcalda', 'unused23')

STRHDRS = ('kstnm', 'kevnm', 'kevnm2', 'khole', 'ko', 'ka', 'kt0', 'kt1',
           'kt2', 'kt3', 'kt4', 'kt5', 'kt6', 'kt7', 'kt8', 'kt9', 'kf',
           'kuser0', 'kuser1', 'kuser2', 'kcmpnm', 'knetwk', 'kdatrd', 'kinst')

"""
NOTE:

kevnm also has a kevnm2 b/c it takes two array spaces.
'kevnm' lookups must be caught and handled differently.  This happens in the
SACTrace string property getters/setters, .io.dict_to_header_arrays
and .arrayio.header_arrays_to_dict.
"""
# NOTE: using namedtuples for header arrays sounds great, but they're immutable

# TODO: make a dict converter between {'<', '>', '='} and {'little', 'big'}

# ------------ ENUMERATED VALUES ----------------------------------------------
# These are stored in the header as integers.
# Their names and values are given in the mapping below.
# Some (many) are not used.
# TODO: this is ugly; rename things a bit
ENUM_VALS = {'itime': 1, 'irlim': 2, 'iamph': 3, 'ixy': 4, 'iunkn': 5,
             'idisp': 6, 'ivel': 7, 'iacc': 8, 'ib': 9, 'iday': 10, 'io': 11,
             'ia': 12, 'it0': 13, 'it1': 14, 'it2': 15, 'it3': 16, 'it4': 17,
             'it5': 18, 'it6': 19, 'it7': 20, 'it8': 21, 'it9': 22, 
             'iradnv': 23, 'itannv': 24, 'iradev': 25, 'itanev': 26,
             'inorth': 27, 'ieast': 28, 'ihorza': 29, 'idown': 30, 'iup': 31,
             'illlbb': 32, 'iwwsn1': 33, 'iwwsn2': 34, 'ihglp': 35, 'isro': 36,
             'inucl': 37, 'ipren': 38, 'ipostn': 39, 'iquake': 40, 'ipreq': 41,
             'ipostq': 42, 'ichem': 43, 'iother': 44, 'igood': 45, 'iglch': 46,
             'idrop': 47, 'ilowsn': 48, 'irldta': 49, 'ivolts': 50, 'imb': 52,
             'ims': 53, 'iml': 54, 'imw': 55, 'imd': 56, 'imx': 57,
             'ineic': 58, 'ipdeq': 59, 'ipdew': 60, 'ipde': 61, 'iisc': 62,
             'ireb': 63, 'iusgs': 64, 'ibrk': 65, 'icaltech': 66, 'illnl': 67,
             'ievloc': 68, 'ijsop': 69, 'iuser': 70, 'iunknown': 71, 'iqb': 72,
             'iqb1': 73, 'iqb2': 74, 'iqbx': 75, 'iqmt': 76, 'ieq': 77,
             'ieq1': 78, 'ieq2': 79, 'ime': 80, 'iex': 81, 'inu': 82,
             'inc': 83, 'io_': 84, 'il': 85, 'ir': 86, 'it': 87, 'iu': 88,
             'ieq3': 89, 'ieq0': 90, 'iex0': 91, 'iqc': 92, 'iqb0': 93,
             'igey': 94, 'ilit': 95, 'imet': 96, 'iodor': 97, 'ios': 103}

# reverse look-up: you have the number, want the string
ENUM_NAMES = dict((v, k) for k, v in ENUM_VALS.iteritems())


# accepted values, by header
ACCEPTED_VALS = {'iftype': ['itime', 'irlim', 'iamph', 'ixy'],
                 'idep': ['iunkn', 'idisp', 'ivel', 'ivolts', 'iacc'],
                 'iztype': ['iunkn', 'ib', 'iday', 'io', 'ia', 'it0', 'it1',
                            'it2', 'it3', 'it4', 'it5', 'it6', 'it7', 'it8',
                            'it9'],
                 'imagtyp': ['imb', 'ims', 'iml', 'imw', 'imd', 'imx'],
                 'imagsrc': ['ineic', 'ipde', 'iisc', 'ireb', 'iusgs', 'ipdeq',
                             'ibrk', 'icaltech', 'illnl', 'ievloc', 'ijsop',
                             'iuser', 'iunknown'],
                 'ievtyp': ['iunkn', 'inucl', 'ipren', 'ipostn', 'iquake',
                            'ipreq', 'ipostq', 'ichem', 'iqb', 'iqb1', 'iqb2',
                            'iqbx', 'iqmt', 'ieq', 'ieq1', 'ieq2', 'ime', 'iex',
                            'inu', 'inc', 'io_', 'il', 'ir', 'it', 'iu',
                            'iother'],
                 'iqual': ['igood', 'iglch', 'idrop', 'ilowsn', 'iother'],
                 'isynth': ['irldta']}

ACCEPTED_INT = ACCEPTED_VALS.copy()
for _hdr in ACCEPTED_INT:
    ACCEPTED_INT[_hdr] = [ENUM_VALS[_ival] for _ival in ACCEPTED_VALS[_hdr]]
