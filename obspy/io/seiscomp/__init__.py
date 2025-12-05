# -*- coding: utf-8 -*-
"""
obspy.io.seiscomp - SeisComP XML inventory and event file support for ObsPy
========================================================================

This module provides read support for SeisComP XML inventory files & 
read and write support for SeisComP XML event files.

Note that the "sc3ml" suffix has now evolved to the general "scml" to 
coincide with SeisComp dropping the version number in its namespace. We
still allow the "SC3ML" format label but this may be dropped eventually.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
