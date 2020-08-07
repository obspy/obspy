# -*- coding: utf-8 -*-
"""
obspy.io.win - WIN read support for ObsPy
=========================================
This module provides read support for WIN waveform data. This format is
written by different dataloggers, including Hakusan LS-7000XT Datamark
dataloggers. There are two subformats, A0 and A1. To our knowledge, A0 is the
only one supported with the current code. A0 conforms to the
data format of the WIN system developed by Earthquake Research Institute
(ERI), the University of Tokyo.

:copyright:
    The ObsPy Development Team (devs@obspy.org), Thomas Lecocq, Adolfo Inza &
    Philippe Lesage
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

"""
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
