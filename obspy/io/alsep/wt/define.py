# -*- coding: utf-8 -*-
# ------------------------------
# Work Tape Structure
# ------------------------------
# header_size: 16 octets
# frame_size:  96 octets
#
# Normally header is duplicated.
# (Single header file also exists)
# -----------------------------
# |    header (16 octets)     |
# |---------------------------|
# |    header (16 octets)     |
# |---------------------------|
# |    frame (96 octets)      |
# |                           |
# |---------------------------|
# |    frame (96 octets)      |
# |                           |
# |---------------------------|
# |    ...                    |
# -----------------------------
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA


SIZE_WT_HEADER = 16
SIZE_WT_FRAME = 96
