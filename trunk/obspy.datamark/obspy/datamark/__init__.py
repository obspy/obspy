# -*- coding: utf-8 -*-
"""
obspy.datamark -DATAMARK read support for ObsPy
=======================================================
TO EDIT
Thomas Lecocq based on others (refs will be included in the release version)
"""

from obspy.core.util import _getVersionString


__version__ = _getVersionString("obspy.datamark")


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
