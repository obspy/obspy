# -*- coding: utf-8 -*-
MODULE_NAME = "obspy.io.segy"


def _patch_header(header, ebcdic=False):
    """
    Helper function to patch a textual header to include the revision
    number and the end header mark.
    """
    revnum = "C39 SEG Y REV1"

    if ebcdic:
        revnum = revnum.encode("EBCDIC-CP-BE")
        end_header = "C40 END EBCDIC        ".encode("EBCDIC-CP-BE")
    else:
        revnum = revnum.encode("ascii")
        end_header = "C40 END TEXTUAL HEADER".encode("ascii")

    header = header[:3200-160] + revnum + header[3200-146:]
    header = header[:3200-80] + end_header + header[3200-58:]

    return header
