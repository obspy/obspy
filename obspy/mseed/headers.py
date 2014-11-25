# -*- coding: utf-8 -*-
"""
Defines the libmseed structures and blockettes.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import ctypes as C
import numpy as np
from obspy.core.util.libnames import _load_CDLL


HPTERROR = -2145916800000000

ENDIAN = {0: '<', 1: '>'}

# Import shared libmseed
clibmseed = _load_CDLL("mseed")


# XXX: Do we still support Python 2.4 ????
# Figure out Py_ssize_t (PEP 353).
#
# Py_ssize_t is only defined for Python 2.5 and above, so it defaults to
# ctypes.c_int for earlier versions.
#
# http://svn.python.org/projects/ctypes/trunk/
#           ctypeslib/ctypeslib/contrib/pythonhdr.py
if hasattr(C.pythonapi, 'Py_InitModule4'):
    Py_ssize_t = C.c_int
elif hasattr(C.pythonapi, 'Py_InitModule4_64'):
    Py_ssize_t = C.c_int64
else:
    # XXX: just hard code it for now
    Py_ssize_t = C.c_int64
    # raise TypeError("Cannot determine type of Py_ssize_t")

# Valid control headers in ASCII numbers.
SEED_CONTROL_HEADERS = [ord('V'), ord('A'), ord('S'), ord('T')]
MINI_SEED_CONTROL_HEADERS = [ord('D'), ord('R'), ord('Q'), ord('M')]
VALID_CONTROL_HEADERS = SEED_CONTROL_HEADERS + MINI_SEED_CONTROL_HEADERS

# expected data types for libmseed id: (numpy, ctypes)
DATATYPES = {b"a": C.c_char, b"i": C.c_int32, b"f": C.c_float,
             b"d": C.c_double}
SAMPLESIZES = {'a': 1, 'i': 4, 'f': 4, 'd': 8}

# Valid record lengths for Mini-SEED files.
VALID_RECORD_LENGTHS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                        131072, 262144, 524288, 1048576]

# allowed encodings:
# id: (name, sampletype a/i/f/d, default NumPy type, write support)
ENCODINGS = {0: ("ASCII", "a", np.dtype(native_str("|S1")).type, True),
             1: ("INT16", "i", np.dtype(np.int16), True),
             3: ("INT32", "i", np.dtype(np.int32), True),
             4: ("FLOAT32", "f", np.dtype(np.float32), True),
             5: ("FLOAT64", "d", np.dtype(np.float64), True),
             10: ("STEIM1", "i", np.dtype(np.int32), True),
             11: ("STEIM2", "i", np.dtype(np.int32), True),
             12: ("GEOSCOPE24", "f", np.dtype(np.float32), False),
             13: ("GEOSCOPE16_3", "f", np.dtype(np.float32), False),
             14: ("GEOSCOPE16_4", "f", np.dtype(np.float32), False),
             16: ("CDSN", "i", np.dtype(np.int32), False),
             30: ("SRO", "i", np.dtype(np.int32), False),
             32: ("DWWSSN", "i", np.dtype(np.int32), False)}

# Encodings not supported by libmseed and consequently ObsPy.
UNSUPPORTED_ENCODINGS = {
    2: "INT24",
    15: "US National Network compression",
    17: "Graefenberg 16 bit gain ranged",
    18: "IPG - Strasbourg 16 bit gain ranged",
    19: "STEIM (3) Comprssion",
    31: "HGLP Format",
    33: "RSTN 16 bit gain ranged"
}

# Maps fixed header activity flags bit number and the matching expected key in
# the flags_value
FIXED_HEADER_ACTIVITY_FLAGS = {0: 'calib_signal',
                               1: 'time_correction',
                               2: 'begin_event',
                               3: 'end_event',
                               4: 'positive_leap',
                               5: 'negative_leap',
                               6: 'event_in_progress'}

# Maps fixed header I/O and clock flags bit number and the matching expected
# key in the flags_value
FIXED_HEADER_IO_CLOCK_FLAGS = {0: 'sta_vol_parity_error_possible',
                               1: 'long_record_read',
                               2: 'short_record_read',
                               3: 'start_of_time_series',
                               4: 'end_of_time_series',
                               5: 'clock_locked'}

# Maps fixed header data quality flags bit number and the matching expected
# key in the flags_value
FIXED_HEADER_DATA_QUAL_FLAGS = {0: 'amplifier_sat_detected',
                                1: 'digitizer_clipping_detected',
                                2: 'spikes_detected',
                                3: 'glitches_detected',
                                4: 'missing_padded_data_present',
                                5: 'telemetry_sync_error',
                                6: 'digital_filter_maybe_charging',
                                7: 'time_tag_questionable'}

# Map the dtype to the samplecode. Redundant information but it is hard coded
# for performance reasons.
SAMPLETYPE = {"|S1": "a",
              "int16": "i",
              "int32": "i",
              "float32": "f",
              "float64": "d",
              np.dtype(native_str("|S1")).type: "a",
              np.dtype(np.int16).type: "i",
              np.dtype(np.int32).type: "i",
              np.dtype(np.float32).type: "f",
              np.dtype(np.float64).type: "d"}
# as defined in libmseed.h
MS_ENDOFFILE = 1
MS_NOERROR = 0


# SEED binary time
class BTime(C.Structure):
    _fields_ = [
        ('year', C.c_ushort),
        ('day', C.c_ushort),
        ('hour', C.c_ubyte),
        ('min', C.c_ubyte),
        ('sec', C.c_ubyte),
        ('unused', C.c_ubyte),
        ('fract', C.c_ushort),
    ]


# Fixed section data of header
class fsdh_s(C.Structure):
    _fields_ = [
        ('sequence_number', C.c_char * 6),
        ('dataquality', C.c_char),
        ('reserved', C.c_char),
        ('station', C.c_char * 5),
        ('location', C.c_char * 2),
        ('channel', C.c_char * 3),
        ('network', C.c_char * 2),
        ('start_time', BTime),
        ('numsamples', C.c_ushort),
        ('samprate_fact', C.c_short),
        ('samprate_mult', C.c_short),
        ('act_flags', C.c_ubyte),
        ('io_flags', C.c_ubyte),
        ('dq_flags', C.c_ubyte),
        ('numblockettes', C.c_ubyte),
        ('time_correct', C.c_int),
        ('data_offset', C.c_ushort),
        ('blockette_offset', C.c_ushort),
    ]


# Blockette 100, Sample Rate (without header)
class blkt_100_s(C.Structure):
    _fields_ = [
        ('samprate', C.c_float),
        ('flags', C.c_byte),
        ('reserved', C.c_ubyte * 3),
    ]
blkt_100 = blkt_100_s


# Blockette 200, Generic Event Detection (without header)
class blkt_200_s(C.Structure):
    _fields_ = [
        ('amplitude', C.c_float),
        ('period', C.c_float),
        ('background_estimate', C.c_float),
        ('flags', C.c_ubyte),
        ('reserved', C.c_ubyte),
        ('time', BTime),
        ('detector', C.c_char * 24),
    ]


# Blockette 201, Murdock Event Detection (without header)
class blkt_201_s(C.Structure):
    _fields_ = [
        ('amplitude', C.c_float),
        ('period', C.c_float),
        ('background_estimate', C.c_float),
        ('flags', C.c_ubyte),
        ('reserved', C.c_ubyte),
        ('time', BTime),
        ('snr_values', C.c_ubyte * 6),
        ('loopback', C.c_ubyte),
        ('pick_algorithm', C.c_ubyte),
        ('detector', C.c_char * 24),
    ]


# Blockette 300, Step Calibration (without header)
class blkt_300_s(C.Structure):
    _fields_ = [
        ('time', BTime),
        ('numcalibrations', C.c_ubyte),
        ('flags', C.c_ubyte),
        ('step_duration', C.c_uint),
        ('interval_duration', C.c_uint),
        ('amplitude', C.c_float),
        ('input_channel', C.c_char * 3),
        ('reserved', C.c_ubyte),
        ('reference_amplitude', C.c_uint),
        ('coupling', C.c_char * 12),
        ('rolloff', C.c_char * 12),
    ]


# Blockette 310, Sine Calibration (without header)
class blkt_310_s(C.Structure):
    _fields_ = [
        ('time', BTime),
        ('reserved1', C.c_ubyte),
        ('flags', C.c_ubyte),
        ('duration', C.c_uint),
        ('period', C.c_float),
        ('amplitude', C.c_float),
        ('input_channel', C.c_char * 3),
        ('reserved2', C.c_ubyte),
        ('reference_amplitude', C.c_uint),
        ('coupling', C.c_char * 12),
        ('rolloff', C.c_char * 12),
    ]


# Blockette 320, Pseudo-random Calibration (without header)
class blkt_320_s(C.Structure):
    _fields_ = [
        ('time', BTime),
        ('reserved1', C.c_ubyte),
        ('flags', C.c_ubyte),
        ('duration', C.c_uint),
        ('ptp_amplitude', C.c_float),
        ('input_channel', C.c_char * 3),
        ('reserved2', C.c_ubyte),
        ('reference_amplitude', C.c_uint),
        ('coupling', C.c_char * 12),
        ('rolloff', C.c_char * 12),
        ('noise_type', C.c_char * 8),
    ]


# Blockette 390, Generic Calibration (without header)
class blkt_390_s(C.Structure):
    _fields_ = [
        ('time', BTime),
        ('reserved1', C.c_ubyte),
        ('flags', C.c_ubyte),
        ('duration', C.c_uint),
        ('amplitude', C.c_float),
        ('input_channel', C.c_char * 3),
        ('reserved2', C.c_ubyte),
    ]


# Blockette 395, Calibration Abort (without header)
class blkt_395_s(C.Structure):
    _fields_ = [
        ('time', BTime),
        ('reserved', C.c_ubyte * 2),
    ]


# Blockette 400, Beam (without header)
class blkt_400_s(C.Structure):
    _fields_ = [
        ('azimuth', C.c_float),
        ('slowness', C.c_float),
        ('configuration', C.c_ushort),
        ('reserved', C.c_ubyte * 2),
    ]


# Blockette 405, Beam Delay (without header)
class blkt_405_s(C.Structure):
    _fields_ = [
        ('delay_values', C.c_ushort * 1),
    ]


# Blockette 500, Timing (without header)
class blkt_500_s(C.Structure):
    _fields_ = [
        ('vco_correction', C.c_float),
        ('time', BTime),
        ('usec', C.c_byte),
        ('reception_qual', C.c_ubyte),
        ('exception_count', C.c_uint),
        ('exception_type', C.c_char * 16),
        ('clock_model', C.c_char * 32),
        ('clock_status', C.c_char * 128),
    ]


# Blockette 1000, Data Only SEED (without header)
class blkt_1000_s(C.Structure):
    _fields_ = [
        ('encoding', C.c_ubyte),
        ('byteorder', C.c_ubyte),
        ('reclen', C.c_ubyte),
        ('reserved', C.c_ubyte),
    ]


# Blockette 1001, Data Extension (without header)
class blkt_1001_s(C.Structure):
    _fields_ = [
        ('timing_qual', C.c_ubyte),
        ('usec', C.c_byte),
        ('reserved', C.c_ubyte),
        ('framecnt', C.c_ubyte),
    ]
blkt_1001 = blkt_1001_s


# Blockette 2000, Opaque Data (without header)
class blkt_2000_s(C.Structure):
    _fields_ = [
        ('length', C.c_ushort),
        ('data_offset', C.c_ushort),
        ('recnum', C.c_uint),
        ('byteorder', C.c_ubyte),
        ('flags', C.c_ubyte),
        ('numheaders', C.c_ubyte),
        ('payload', C.c_char * 1),
    ]


# Blockette chain link, generic linkable blockette index
class blkt_link_s(C.Structure):
    pass

blkt_link_s._fields_ = [
    ('blktoffset', C.c_ushort),  # Blockette offset
    ('blkt_type', C.c_ushort),  # Blockette type
    ('next_blkt', C.c_ushort),  # Offset to next blockette
    ('blktdata', C.POINTER(None)),  # Blockette data
    ('blktdatalen', C.c_ushort),  # Length of blockette data in bytes
    ('next', C.POINTER(blkt_link_s))]
BlktLink = blkt_link_s


class StreamState_s(C.Structure):
    _fields_ = [
        ('packedrecords', C.c_longlong),  # Count of packed records
        ('packedsamples', C.c_longlong),  # Count of packed samples
        ('lastintsample', C.c_int),       # Value of last integer sample packed
        ('comphistory', C.c_byte),        # Control use of lastintsample for
                                          # compression history
    ]
StreamState = StreamState_s


class MSRecord_s(C.Structure):
    pass

MSRecord_s._fields_ = [
    ('record', C.POINTER(C.c_char)),  # Mini-SEED record
    ('reclen', C.c_int),              # Length of Mini-SEED record in bytes
                                      # Pointers to SEED data record structures
    ('fsdh', C.POINTER(fsdh_s)),      # Fixed Section of Data Header
    ('blkts', C.POINTER(BlktLink)),   # Root of blockette chain
    ('Blkt100',
     C.POINTER(blkt_100_s)),          # Blockette 100, if present
    ('Blkt1000',
     C.POINTER(blkt_1000_s)),         # Blockette 1000, if present
    ('Blkt1001',
     C.POINTER(blkt_1001_s)),         # Blockette 1001, if present
                                      # Common header fields in accessible form
    ('sequence_number', C.c_int),     # SEED record sequence number
    ('network', C.c_char * 11),       # Network designation, NULL terminated
    ('station', C.c_char * 11),       # Station designation, NULL terminated
    ('location', C.c_char * 11),      # Location designation, NULL terminated
    ('channel', C.c_char * 11),       # Channel designation, NULL terminated
    ('dataquality', C.c_char),        # Data quality indicator
    ('starttime', C.c_longlong),      # Record start time, corrected (first
                                      # sample)
    ('samprate', C.c_double),         # Nominal sample rate (Hz)
    ('samplecnt', C.c_int64),         # Number of samples in record
    ('encoding', C.c_byte),           # Data encoding format
    ('byteorder', C.c_byte),          # Byte order of record
                                      # Data sample fields
    ('datasamples', C.c_void_p),      # Data samples, 'numsamples' of type
                                      # 'sampletype'
    ('numsamples', C.c_int64),        # Number of data samples in datasamples
    ('sampletype', C.c_char),         # Sample type code: a, i, f, d
                                      # Stream oriented state information
    ('ststate',
     C.POINTER(StreamState)),         # Stream processing state information
]
MSRecord = MSRecord_s


class MSTrace_s(C.Structure):
    pass

MSTrace_s._fields_ = [
    ('network', C.c_char * 11),       # Network designation, NULL terminated
    ('station', C.c_char * 11),       # Station designation, NULL terminated
    ('location', C.c_char * 11),      # Location designation, NULL terminated
    ('channel', C.c_char * 11),       # Channel designation, NULL terminated
    ('dataquality', C.c_char),        # Data quality indicator
    ('type', C.c_char),               # MSTrace type code
    ('starttime', C.c_longlong),      # Time of first sample
    ('endtime', C.c_longlong),        # Time of last sample
    ('samprate', C.c_double),         # Nominal sample rate (Hz)
    ('samplecnt', C.c_int64),         # Number of samples in trace coverage
    ('datasamples', C.c_void_p),      # Data samples, 'numsamples' of type
                                      # 'sampletype'
    ('numsamples', C.c_int64),        # Number of data samples in datasamples
    ('sampletype', C.c_char),         # Sample type code: a, i, f, d
    ('prvtptr', C.c_void_p),          # Private pointer for general use
    ('ststate',
     C.POINTER(StreamState)),         # Stream processing state information
    ('next', C.POINTER(MSTrace_s)),   # Pointer to next trace
]
MSTrace = MSTrace_s


class MSTraceGroup_s(C.Structure):
    pass

MSTraceGroup_s._fields_ = [
    ('numtraces', C.c_int),            # Number of MSTraces in the trace chain
    ('traces', C.POINTER(MSTrace_s)),  # Root of the trace chain
]
MSTraceGroup = MSTraceGroup_s


# Define the high precision time tick interval as 1/modulus seconds */
# Default modulus of 1000000 defines tick interval as a microsecond */
HPTMODULUS = 1000000.0


# Reading Mini-SEED records from files
class MSFileParam_s(C.Structure):
    pass

MSFileParam_s._fields_ = [
    ('fp', C.POINTER(Py_ssize_t)),
    ('filename', C.c_char * 512),
    ('rawrec', C.c_char_p),
    ('readlen', C.c_int),
    ('readoffset', C.c_int),
    ('packtype', C.c_int),
    ('packhdroffset', C.c_long),
    ('filepos', C.c_long),
    ('filesize', C.c_long),
    ('recordcount', C.c_int),
]
MSFileParam = MSFileParam_s


class U_DIFF(C.Union):
    """
    Union for Steim objects.
    """
    _fields_ = [
        ("byte", C.c_int8 * 4),  # 4 1-byte differences.
        ("hw", C.c_int16 * 2),  # 2 halfword differences.
        ("fw", C.c_int32),  # 1 fullword difference.
    ]


class FRAME(C.Structure):
    """
    Frame in a seed data record.
    """
    _fields_ = [
        ("ctrl", C.c_uint32),  # control word for frame.
        ("w", U_DIFF * 14),  # compressed data.
    ]


# Declare function of libmseed library, argument parsing
clibmseed.mst_init.argtypes = [C.POINTER(MSTrace)]
clibmseed.mst_init.restype = C.POINTER(MSTrace)

clibmseed.mst_free.argtypes = [C.POINTER(C.POINTER(MSTrace))]
clibmseed.mst_free.restype = C.c_void_p

clibmseed.mst_initgroup.argtypes = [C.POINTER(MSTraceGroup)]
clibmseed.mst_initgroup.restype = C.POINTER(MSTraceGroup)

clibmseed.mst_freegroup.argtypes = [C.POINTER(C.POINTER(MSTraceGroup))]
clibmseed.mst_freegroup.restype = C.c_void_p

clibmseed.msr_init.argtypes = [C.POINTER(MSRecord)]
clibmseed.msr_init.restype = C.POINTER(MSRecord)

clibmseed.ms_readmsr_r.argtypes = [
    C.POINTER(C.POINTER(MSFileParam)), C.POINTER(C.POINTER(MSRecord)),
    C.c_char_p, C.c_int, C.POINTER(Py_ssize_t), C.POINTER(C.c_int), C.c_short,
    C.c_short, C.c_short]
clibmseed.ms_readmsr_r.restypes = C.c_int

clibmseed.ms_readtraces.argtypes = [
    C.POINTER(C.POINTER(MSTraceGroup)), C.c_char_p, C.c_int, C.c_double,
    C.c_double, C.c_short, C.c_short, C.c_short, C.c_short]
clibmseed.ms_readtraces.restype = C.c_int

clibmseed.ms_readtraces_timewin.argtypes = [
    C.POINTER(C.POINTER(MSTraceGroup)), C.c_char_p, C.c_int, C.c_double,
    C.c_double, C.c_int64, C.c_int64, C.c_short, C.c_short, C.c_short,
    C.c_short]
clibmseed.ms_readtraces_timewin.restype = C.c_int

clibmseed.msr_starttime.argtypes = [C.POINTER(MSRecord)]
clibmseed.msr_starttime.restype = C.c_int64

clibmseed.msr_endtime.argtypes = [C.POINTER(MSRecord)]
clibmseed.msr_endtime.restype = C.c_int64

clibmseed.ms_detect.argtypes = [C.c_char_p, C.c_int]
clibmseed.ms_detect.restype = C.c_int

clibmseed.msr_unpack_steim2.argtypes = [
    C.POINTER(FRAME), C.c_int, C.c_int, C.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.POINTER(C.c_int32), C.POINTER(C.c_int32), C.c_int, C.c_int]
clibmseed.msr_unpack_steim2.restype = C.c_int

clibmseed.msr_unpack_steim1.argtypes = [
    C.POINTER(FRAME), C.c_int, C.c_int, C.c_int,
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.POINTER(C.c_int32), C.POINTER(C.c_int32), C.c_int, C.c_int]
clibmseed.msr_unpack_steim2.restype = C.c_int

# tricky, C.POINTER(C.c_char) is a pointer to single character fields
# this is completely different to C.c_char_p which is a string
clibmseed.mst_packgroup.argtypes = [
    C.POINTER(MSTraceGroup), C.CFUNCTYPE(
        C.c_void_p, C.POINTER(C.c_char), C.c_int, C.c_void_p),
    C.c_void_p, C.c_int, C.c_short, C.c_short, C.POINTER(C.c_int), C.c_short,
    C.c_short, C.POINTER(MSRecord)]
clibmseed.mst_packgroup.restype = C.c_int

clibmseed.msr_addblockette.argtypes = [C.POINTER(MSRecord),
                                       C.POINTER(C.c_char),
                                       C.c_int, C.c_int, C.c_int]
clibmseed.msr_addblockette.restype = C.POINTER(BlktLink)

clibmseed.msr_parse.argtypes = [C.POINTER(C.c_char), C.c_int,
                                C.POINTER(C.POINTER(MSRecord)),
                                C.c_int, C.c_int, C.c_int]
clibmseed.msr_parse.restype = C.c_int

#####################################
# Define the C structures.
#####################################


# Container for a continuous trace segment, linkable
class MSTraceSeg(C.Structure):
    pass


MSTraceSeg._fields_ = [
    ('starttime', C.c_longlong),      # Time of first sample
    ('endtime', C.c_longlong),        # Time of last sample
    ('samprate', C.c_double),         # Nominal sample rate (Hz)
    ('samplecnt', C.c_int64),         # Number of samples in trace coverage
    ('datasamples', C.c_void_p),      # Data samples, 'numsamples' of type
                                      # 'sampletype'
    ('numsamples', C.c_int64),        # Number of data samples in datasamples
    ('sampletype', C.c_char),         # Sample type code: a, i, f, d
    ('prvtptr', C.c_void_p),          # Private pointer for general use, unused
                                      # by libmseed
    ('prev', C.POINTER(MSTraceSeg)),  # Pointer to previous segment
    ('next', C.POINTER(MSTraceSeg))   # Pointer to next segment
]


# Container for a trace ID, linkable
class MSTraceID(C.Structure):
    pass

MSTraceID._fields_ = [
    ('network', C.c_char * 11),       # Network designation, NULL terminated
    ('station', C.c_char * 11),       # Station designation, NULL terminated
    ('location', C.c_char * 11),      # Location designation, NULL terminated
    ('channel', C.c_char * 11),       # Channel designation, NULL terminated
    ('dataquality', C.c_char),        # Data quality indicator
    ('srcname', C.c_char * 45),       # Source name (Net_Sta_Loc_Chan_Qual),
                                      # NULL terminated
    ('type', C.c_char),               # Trace type code
    ('earliest', C.c_longlong),       # Time of earliest sample
    ('latest', C.c_longlong),         # Time of latest sample
    ('prvtptr', C.c_void_p),          # Private pointer for general use, unused
                                      # by libmseed
    ('numsegments', C.c_int),         # Number of segments for this ID
    ('first',
     C.POINTER(MSTraceSeg)),          # Pointer to first of list of segments
    ('last', C.POINTER(MSTraceSeg)),  # Pointer to last of list of segments
    ('next', C.POINTER(MSTraceID))    # Pointer to next trace
]


# Container for a continuous trace segment, linkable
class MSTraceList(C.Structure):
    pass

MSTraceList._fields_ = [
    ('numtraces', C.c_int),            # Number of traces in list
    ('traces', C.POINTER(MSTraceID)),  # Pointer to list of traces
    ('last', C.POINTER(MSTraceID))     # Pointer to last used trace in list
]


# Data selection structure time window definition containers
class SelectTime(C.Structure):
    pass

SelectTime._fields_ = [
    ('starttime', C.c_longlong),  # Earliest data for matching channels
    ('endtime', C.c_longlong),    # Latest data for matching channels
    ('next', C.POINTER(SelectTime))
]


# Data selection structure definition containers
class Selections(C.Structure):
    pass

Selections._fields_ = [
    ('srcname', C.c_char * 100),  # Matching (globbing) source name:
                                  # Net_Sta_Loc_Chan_Qual
    ('timewindows', C.POINTER(SelectTime)),
    ('next', C.POINTER(Selections))
]


# Container for a continuous linked list of records.
class ContinuousSegment(C.Structure):
    pass

ContinuousSegment._fields_ = [
    ('starttime', C.c_longlong),
    ('endtime', C.c_longlong),
    ('samprate', C.c_double),
    ('sampletype', C.c_char),
    ('hpdelta', C.c_longlong),
    ('samplecnt', C.c_int64),
    ('timing_quality', C.c_uint8),
    ('calibration_type', C.c_int8),
    ('datasamples', C.c_void_p),  # Data samples, 'numsamples' of type
                                  # 'sampletype'
    ('firstRecord', C.c_void_p),
    ('lastRecord', C.c_void_p),
    ('next', C.POINTER(ContinuousSegment)),
    ('previous', C.POINTER(ContinuousSegment))
]


# A container for continuous segments with the same id
class LinkedIDList(C.Structure):
    pass

LinkedIDList._fields_ = [
    ('network', C.c_char * 11),      # Network designation, NULL terminated
    ('station', C.c_char * 11),      # Station designation, NULL terminated
    ('location', C.c_char * 11),     # Location designation, NULL terminated
    ('channel', C.c_char * 11),      # Channel designation, NULL terminated
    ('dataquality', C.c_char),       # Data quality indicator
    ('firstSegment',
     C.POINTER(ContinuousSegment)),  # Pointer to first of list of segments
    ('lastSegment',
     C.POINTER(ContinuousSegment)),  # Pointer to last of list of segments
    ('next',
     C.POINTER(LinkedIDList)),       # Pointer to next id
    ('previous',
     C.POINTER(LinkedIDList)),       # Pointer to previous id
]


#########################################
# Done with the C structures definitions.
#########################################

# Set the necessary arg- and restypes.
clibmseed.readMSEEDBuffer.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int8, ndim=1,
                           flags=native_str('C_CONTIGUOUS')),
    C.c_int,
    C.POINTER(Selections),
    C.c_int8,
    C.c_int,
    C.c_int8,
    C.c_int8,
    C.c_int,
    C.CFUNCTYPE(C.c_long, C.c_int, C.c_char),
    C.CFUNCTYPE(C.c_void_p, C.c_char_p),
    C.CFUNCTYPE(C.c_void_p, C.c_char_p)
]

clibmseed.readMSEEDBuffer.restype = C.POINTER(LinkedIDList)

clibmseed.msr_free.argtypes = [C.POINTER(C.POINTER(MSRecord))]
clibmseed.msr_free.restype = C.c_void_p

clibmseed.mstl_init.restype = C.POINTER(MSTraceList)
clibmseed.mstl_free.argtypes = [C.POINTER(C.POINTER(MSTraceList)), C.c_int]


clibmseed.lil_free.argtypes = [C.POINTER(LinkedIDList)]
clibmseed.lil_free.restype = C.c_void_p


clibmseed.allocate_bytes.argtypes = (C.c_int,)
clibmseed.allocate_bytes.restype = C.c_void_p


# Python callback functions for C
def __PyFile_callback(_f):
    return 1
_PyFile_callback = C.CFUNCTYPE(C.c_int, Py_ssize_t)(__PyFile_callback)
