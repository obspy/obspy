# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import sys

from obspy.core.util.deprecation_helpers import \
    DynamicAttributeImportRerouteModule


# Remove once 0.11 has been released.
sys.modules[__name__] = DynamicAttributeImportRerouteModule(
    name=__name__, doc=__doc__, locs=locals(),
    original_module=sys.modules[__name__],
    import_map={},
    function_map={
        "is_StationXML": "obspy.io.stationxml.core._is_stationxml",
        "read_StationXML": "obspy.io.stationxml.core._read_stationxml",
        "validate_StationXML":
            "obspy.io.stationxml.core.validate_stationxml",
        "write_StationXML": "obspy.io.stationxml.core._write_stationxml"})
