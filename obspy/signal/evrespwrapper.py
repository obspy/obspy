import ctypes as C
from obspy.signal.headers import clibevresp


STALEN = clibevresp.STALEN
NETLEN = clibevresp.NETLEN
LOCIDLEN = clibevresp.LOCIDLEN
CHALEN = clibevresp.CHALEN


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
    ("blkt_info", C.POINTER()),
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
