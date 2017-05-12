# -*- coding: utf-8 -*-
"""
Base constants for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
# NO IMPORTS FROM OBSPY OR FUTURE IN THIS FILE! (file gets used at
# installation time)
import sys


HARD_DEPENDENCIES = [
    "future", "numpy", "scipy", "matplotlib", "lxml.etree", "setuptools",
    "sqlalchemy", "decorator", "requests"]
OPTIONAL_DEPENDENCIES = [
    "flake8", "pyimgur", "pyproj", "pep8-naming", "m2crypto", "osgeo.gdal",
    "mpl_toolkits.basemap", "mock", "pyflakes", "geographiclib", "cartopy"]
DEPENDENCIES = HARD_DEPENDENCIES + OPTIONAL_DEPENDENCIES

# defining ObsPy modules currently used by runtests and the path function
DEFAULT_MODULES = ['clients.filesystem', 'core', 'db', 'geodetics', 'imaging',
                   'io.ah', 'io.arclink', 'io.ascii', 'io.cmtsolution',
                   'io.cnv', 'io.css', 'io.win', 'io.gcf', 'io.gse2',
                   'io.json', 'io.kinemetrics', 'io.kml', 'io.mseed', 'io.ndk',
                   'io.nied', 'io.nlloc', 'io.nordic', 'io.pdas', 'io.pde',
                   'io.quakeml', 'io.reftek', 'io.sac', 'io.scardec',
                   'io.seg2', 'io.segy', 'io.seisan', 'io.sh', 'io.shapefile',
                   'io.seiscomp', 'io.stationtxt', 'io.stationxml', 'io.wav',
                   'io.xseed', 'io.y', 'io.zmap', 'realtime', 'scripts',
                   'signal', 'taup']
NETWORK_MODULES = ['clients.arclink', 'clients.earthworm', 'clients.fdsn',
                   'clients.iris', 'clients.neic', 'clients.seedlink',
                   'clients.seishub', 'clients.syngine']
ALL_MODULES = DEFAULT_MODULES + NETWORK_MODULES

# default order of automatic format detection
WAVEFORM_PREFERRED_ORDER = ['MSEED', 'SAC', 'GSE2', 'SEISAN', 'SACXY', 'GSE1',
                            'Q', 'SH_ASC', 'SLIST', 'TSPAIR', 'Y', 'PICKLE',
                            'SEGY', 'SU', 'SEG2', 'WAV', 'WIN', 'CSS',
                            'NNSA_KB_CORE', 'AH', 'PDAS', 'KINEMETRICS_EVT',
                            'GCF']
EVENT_PREFERRED_ORDER = ['QUAKEML', 'NLLOC_HYP']
# waveform plugins accepting a byteorder keyword
WAVEFORM_ACCEPT_BYTEORDER = ['MSEED', 'Q', 'SAC', 'SEGY', 'SU']

NATIVE_BYTEORDER = sys.byteorder == 'little' and '<' or '>'
