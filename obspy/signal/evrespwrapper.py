import numpy as np
import ctypes as C
from obspy.signal.headers import clibevresp  # , C_COMPLEX, RESPONSE


class complex_number(C.Structure):
    _fields_ = [
        ("real", C.c_double),
        ("imag", C.c_double),
    ]


class pole_zeroType(C.Structure):
    _fields_ = [
        ("nzeros", C.c_int),
        ("npoles", C.c_int),
        ("a0", C.c_double),
        ("a0_freq", C.c_double),
        ("zeros", C.POINTER(complex_number)),
        ("poles", C.POINTER(complex_number)),
    ]


class blkt_info_union(C.Union):
    _fields_ = [
        ("pole_zero", C.POINTER(pole_zeroType))
    ]


class blkt(C.Structure):
    pass

blkt._fields_ = [
    ("type", C.c_int),
    ("blkt_info", C.POINTER(blkt_info_union)),
    ("next_blkt", C.POINTER(blkt))
]


class stage(C.Structure):
    pass

stage._fields_ = [
    ("sequence_no", C.c_int),
    ("input_units", C.c_int),
    ("output_units", C.c_int),
    ("first_blkt", C.POINTER(blkt)),
    ("next_stage", C.POINTER(stage))
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


class channel(C.Structure):
    pass

channel._fields_ = [
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
    ("calc_delay", C.c_double),
    ("estim_delay", C.c_double),
    ("applied_corr", C.c_double),
    ("sint", C.c_double),
    ("nstages", C.c_int),
    ("first_stage", C.POINTER(channel)),
]


# void calc_resp(struct channel *chan, double *freq, int nfreqs,
#                struct complex *output,
#                char *out_units, int start_stage, int stop_stage,
#                int useTotalSensitivityFlag)
clibevresp.calc_resp.argtypes = [
    C.POINTER(channel),
    C.POINTER(C.c_double),
    C.c_int,
    np.ctypeslib.ndpointer(dtype='complex128',  # output
                           ndim=1,
                           flags='C_CONTIGUOUS'),
    C.c_char_p,
    C.c_int,
    C.c_int,
    C.c_int]
clibevresp.calc_resp.restype = C.c_void_p
