# -*- coding: utf-8 -*-
import ctypes as C  # NOQA

import numpy as np

from obspy.signal.headers import clibevresp


clibevresp.twoPi = 3.141

ENUM_ERROR_CODES = {
    2: (IOError, 'Open file error'),
    3: (Exception, 'RE compilation failed'),
    4: (Exception, 'Merge error'),  # Unused
    5: (Exception, 'Swap failed'),
    6: (Exception, 'Usage error'),  # Should not happen
    7: (ValueError, 'Bad out units'),
    -1: (MemoryError, 'Out of memory'),
    -2: (Exception, 'Non existent field'),  # Unused
    -3: (ValueError, 'Undefined prefix'),
    -4: (ValueError, 'Parse error'),
    -5: (ValueError, 'Illegal RESP format'),
    -6: (ValueError, 'Undef sepstr'),
    -7: (IOError, 'Unrecognized file type'),
    -8: (EOFError, 'Unexpected EOF'),
    -9: (IndexError, 'Array bounds exceeded'),
    -10: (ValueError, 'Improper data type'),
    -11: (NotImplementedError, 'Unsupported file type'),
    -12: (ValueError, 'Illegal filter specification'),
    -13: (IndexError, 'No stage matched'),
    -14: (NotImplementedError, 'Unrecognized units')
}

ENUM_UNITS = {
    "UNDEF_UNITS": 0,
    "DIS": 1,
    "VEL": 2,
    "ACC": 3,
    "COUNTS": 4,
    "VOLTS": 5,
    "DEFAULT": 6,
    "PRESSURE": 7,
    "TESLA": 8
}

ENUM_FILT_TYPES = {
    "UNDEF_FILT": 0,
    "LAPLACE_PZ": 1,
    "ANALOG_PZ": 2,
    "IIR_PZ": 3,
    "FIR_SYM_1": 4,
    "FIR_SYM_2": 5,
    "FIR_ASYM": 6,
    "LIST": 7,
    "GENERIC": 8,
    "DECIMATION": 9,
    "GAIN": 10,
    "REFERENCE": 11,
    "FIR_COEFFS": 12,
    "IIR_COEFFS": 13
}

ENUM_STAGE_TYPES = {
    "UNDEF_STAGE": 0,
    "PZ_TYPE": 1,
    "IIR_TYPE": 2,
    "FIR_TYPE": 3,
    "GAIN_TYPE": 4,
    "LIST_TYPE": 5,
    "IIR_COEFFS_TYPE": 6,
    "GENERIC_TYPE": 7
}


class ComplexNumber(C.Structure):
    _fields_ = [
        ("real", C.c_double),
        ("imag", C.c_double),
    ]


class PoleZeroType(C.Structure):
    _fields_ = [
        ("nzeros", C.c_int),
        ("npoles", C.c_int),
        ("a0", C.c_double),
        ("a0_freq", C.c_double),
        ("zeros", C.POINTER(ComplexNumber)),
        ("poles", C.POINTER(ComplexNumber)),
    ]


class CoeffType(C.Structure):
    _fields_ = [
        ("nnumer", C.c_int),
        ("ndenom", C.c_int),
        ("numer", C.POINTER(C.c_double)),
        ("denom", C.POINTER(C.c_double)),
        ("h0", C.c_double),
    ]


class FIRType(C.Structure):
    _fields_ = [
        ("ncoeffs", C.c_int),
        ("coeffs", C.POINTER(C.c_double)),
        ("h0", C.c_double)
    ]


class ListType(C.Structure):
    _fields_ = [
        ("nresp", C.c_int),
        ("freq", C.POINTER(C.c_double)),
        ("amp", C.POINTER(C.c_double)),
        ("phase", C.POINTER(C.c_double))
    ]


class GenericType(C.Structure):
    _fields_ = [
    ]


class DecimationType(C.Structure):
    _fields_ = [
        ("sample_int", C.c_double),
        ("deci_fact", C.c_int),
        ("deci_offset", C.c_int),
        ("estim_delay", C.c_double),
        ("applied_corr", C.c_double)
    ]


class GainType(C.Structure):
    _fields_ = [
        ("gain", C.c_double),
        ("gain_freq", C.c_double)
    ]


class ReferType(C.Structure):
    _fields_ = [
    ]


class BlktInfoUnion(C.Union):
    _fields_ = [
        ("pole_zero", PoleZeroType),
        ("fir", FIRType),
        ("list", ListType),
        ("decimation", DecimationType),
        ("gain", GainType),
        ("coeff", CoeffType)
    ]


class Blkt(C.Structure):
    pass


Blkt._fields_ = [
    ("type", C.c_int),
    ("blkt_info", BlktInfoUnion),
    # ("blkt_info", PoleZeroType),
    ("next_blkt", C.POINTER(Blkt))
]


class Stage(C.Structure):
    pass


Stage._fields_ = [
    ("sequence_no", C.c_int),
    ("input_units", C.c_int),
    ("output_units", C.c_int),
    ("first_blkt", C.POINTER(Blkt)),
    ("next_stage", C.POINTER(Stage))
]

STALEN = 64
NETLEN = 64
LOCIDLEN = 64
CHALEN = 64
DATIMLEN = 23
MAXLINELEN = 256

# needed ?
OUTPUTLEN = 256
TMPSTRLEN = 64
UNITS_STR_LEN = 16
UNITSLEN = 20
BLKTSTRLEN = 4
FLDSTRLEN = 3
MAXFLDLEN = 50
MAXLINELEN = 256
FIR_NORM_TOL = 0.02


class Channel(C.Structure):
    pass


Channel._fields_ = [
    ("staname", C.c_char * STALEN),
    ("network", C.c_char * NETLEN),
    ("locid", C.c_char * LOCIDLEN),
    ("chaname", C.c_char * CHALEN),
    ("beg_t", C.c_char * DATIMLEN),
    ("end_t", C.c_char * DATIMLEN),
    ("first_units", C.c_char * MAXLINELEN),
    ("last_units", C.c_char * MAXLINELEN),
    ("sensit", C.c_double),
    ("sensfreq", C.c_double),
    ("calc_sensit", C.c_double),
    ("calc_delay", C.c_double),
    ("estim_delay", C.c_double),
    ("applied_corr", C.c_double),
    ("sint", C.c_double),
    ("nstages", C.c_int),
    ("first_stage", C.POINTER(Stage)),
]


clibevresp.curr_file = C.c_char_p.in_dll(clibevresp, 'curr_file')


# int _obspy_calc_resp(struct channel *chan, double *freq, int nfreqs,
#                      struct complex *output,
#                      char *out_units, int start_stage, int stop_stage,
#                      int useTotalSensitivityFlag)
clibevresp._obspy_calc_resp.argtypes = [
    C.POINTER(Channel),
    np.ctypeslib.ndpointer(dtype=np.float64,  # freqs
                           ndim=1,
                           flags='C_CONTIGUOUS'),
    C.c_int,
    np.ctypeslib.ndpointer(dtype=np.complex128,  # output
                           ndim=1,
                           flags='C_CONTIGUOUS'),
    C.c_char_p,
    C.c_int,
    C.c_int,
    C.c_int]
clibevresp._obspy_calc_resp.restype = C.c_int


# int _obspy_check_channel(struct channel *chan)
clibevresp._obspy_check_channel.argtypes = [C.POINTER(Channel)]
clibevresp._obspy_check_channel.restype = C.c_int


# int _obspy_norm_resp(struct channel *chan, int start_stage, int stop_stage)
clibevresp._obspy_norm_resp.argtypes = [C.POINTER(Channel), C.c_int, C.c_int,
                                        C.c_int]
clibevresp._obspy_norm_resp.restype = C.c_int


# Only useful for debugging thus not officially included as every import of
# this file results in the function pointer being created thus slowing it down.
# void print_chan(struct channel *chan, int start_stage, int stop_stage,
#                 int stdio_flag, int listinterp_out_flag,
#                 int listinterp_in_flag, int useTotalSensitivityFlag)
# clibevresp.print_chan.argtypes = [C.POINTER(channel), C.c_int, C.c_int,
#                                  C.c_int, C.c_int, C.c_int, C.c_int]
# clibevresp.print_chan.restype = C.c_void_p
