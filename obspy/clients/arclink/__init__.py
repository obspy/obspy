# -*- coding: utf-8 -*-
"""
obspy.clients.arclink - ArcLink/WebDC request client for ObsPy
==============================================================

DEPRECATED -- ArcLink protocol is officially deprecated

ArcLink protocol has been officially deprecated and some main servers have been
shut down already. Please consider using other methods like FDSN web services
to fetch data.  ArcLink functionality is now untested in ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import os
import warnings

from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning

# Raise a deprecation warning. This has been requested by the EIDA management
# board.
msg = ('ArcLink protocol has been officially deprecated and some '
       'main servers have been shut down already. Please consider '
       'using other methods like FDSN web services to fetch data. '
       'ArcLink functionality is now untested in ObsPy.')
# suppress warning on docs build
if os.environ.get('SPHINX') != 'true':
    warnings.warn(msg, category=ObsPyDeprecationWarning)

from .client import Client  # NOQA
