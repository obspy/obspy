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
SIZE_WT_HEADER = 16
SIZE_WT_FRAME = 96
