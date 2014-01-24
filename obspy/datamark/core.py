# -*- coding: utf-8 -*-
"""
WIN bindings to ObsPy core module.
"""

import warnings

from obspy.win.core import isWIN as isDATAMARK
from obspy.win.core import readWIN as readDATAMARK


msg = 'Module obspy.datamark is deprecated! Use ' + \
      'obspy.win instead. Use read(file,\'WIN\') ' +\
      'Instead of read(file,\'DATAMARK\').'

warnings.warn(msg, category=DeprecationWarning)