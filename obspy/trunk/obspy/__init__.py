# -*- coding: utf-8 -*-
"""
ObsPy: A Python Toolbox for Seismology/Seismological Observatories

Capabilities include unified read and write support for GSE2, Mini-SEED,
SAC, Dataless-Seed, filtering, instrument simulation, triggering, imaging,
arclink, seishub as well as experimental support for Fissures and Seisan.
The capabilities are provided by several ObsPy packages.

:note: Currently only obspy.core, obspy.gse2, obspy.mseed and
       obspy.imaging are automatically installed. Other modules can be
       found at www.obspy.org
:copyright: The ObsPy Development Team (devs@obspy.org)
:license: Depending on the package, either GNU Lesser General Public
          License, Version 3 (**LGPLv3**), or GNU General Public License,
          Version 2 (**GPLv2**).
"""

# A namespace package will not have any method, class or any other
# objects defined in that level.
# :see: http://baijum81.livejournal.com/22412.html

# DON'T IMPORT HERE ANYTHING!

# See http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    from pkgutil import extend_path
    __path__ = extend_path(__path__, __name__)
