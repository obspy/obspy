#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
ObsPy - a Python framework for seismological observatories.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats,
clients to access data centers and seismological signal processing routines
which allow the manipulation of seismological time series (see Beyreuther et
al. 2010, Megies et al. 2011).

The goal of the ObsPy project is to facilitate rapid application development
for seismology.

For more information visit https://www.obspy.org.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
# Importing setuptools monkeypatches some of distutils commands so things like
# 'python setup.py develop' work. Wrap in try/except so it is not an actual
# dependency. Inplace installation with pip works also without importing
# setuptools.
try:
    import setuptools  # @UnusedImport # NOQA
except ImportError:
    pass

try:
    import numpy  # @UnusedImport # NOQA
except ImportError:
    msg = ("No module named numpy. "
           "Please install numpy first, it is needed before installing ObsPy.")
    raise ImportError(msg)

import fnmatch
import glob
import inspect
import os
import sys
import platform
from distutils.util import change_root
from distutils.errors import DistutilsSetupError

from numpy.distutils.core import setup
from numpy.distutils.ccompiler import get_default_compiler
from numpy.distutils.command.build import build
from numpy.distutils.command.install import install
from numpy.distutils.exec_command import exec_command, find_executable
from numpy.distutils.misc_util import Configuration


# Directory of the current file in the (hopefully) most reliable way
# possible, according to krischer
SETUP_DIRECTORY = os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe())))

# Import the version string.
# Any .py files that are used at install time must be registered in
# obspy.core.tests.test_util_misc.UtilMiscTestCase.test_no_obspy_imports!
UTIL_PATH = os.path.join(SETUP_DIRECTORY, "obspy", "core", "util")
sys.path.insert(0, UTIL_PATH)
from version import get_git_version  # @UnresolvedImport
from libnames import _get_lib_name  # @UnresolvedImport
sys.path.pop(0)

LOCAL_PATH = os.path.join(SETUP_DIRECTORY, "setup.py")
DOCSTRING = __doc__.split("\n")

# check for MSVC
if platform.system() == "Windows" and (
        'msvc' in sys.argv or
        '-c' not in sys.argv and
        get_default_compiler() == 'msvc'):
    IS_MSVC = True
else:
    IS_MSVC = False

# Use system libraries? Set later...
EXTERNAL_EVALRESP = False
EXTERNAL_LIBMSEED = False

# package specific settings
KEYWORDS = [
    'ArcLink', 'array', 'array analysis', 'ASC', 'beachball',
    'beamforming', 'cross correlation', 'database', 'dataless',
    'Dataless SEED', 'DMX', 'earthquakes', 'Earthworm', 'EIDA',
    'envelope', 'ESRI', 'events', 'FDSN', 'features', 'filter',
    'focal mechanism', 'FOCMEC', 'GCF', 'GSE1', 'GSE2', 'hob', 'Tau-P',
    'IASPEI', 'imaging', 'IMS', 'instrument correction',
    'instrument simulation', 'IRIS', 'ISF', 'kinemetrics', 'KML', 'magnitude',
    'MiniSEED', 'misfit', 'mopad', 'MSEED', 'NDK', 'NERA', 'NERIES',
    'NonLinLoc', 'NLLOC', 'Nordic', 'NRL', 'observatory', 'ORFEUS', 'PDAS',
    'picker', 'processing', 'PQLX', 'Q', 'real time', 'realtime', 'REFTEK',
    'REFTEK130', 'RG-1.6', 'RT-130', 'RESP', 'response file', 'RT', 'SAC',
    'scardec', 'sc3ml', 'SDS', 'SEED', 'SeedLink', 'SEG-2', 'SEG Y', 'SEISAN',
    'SeisHub', 'Seismic Handler', 'seismology', 'seismogram', 'seismograms',
    'shapefile', 'signal', 'slink', 'spectrogram', 'StationXML', 'taper',
    'taup', 'travel time', 'trigger', 'VERCE', 'WAV', 'waveform', 'WaveServer',
    'WaveServerV', 'WebDC', 'web service', 'WIN', 'Winston', 'XML-SEED',
    'XSEED']

# when bumping to numpy 1.9.0: replace bytes() in io.reftek with np.tobytes()
# and in obspy/io/mseed/core.py change downcasting check to numpy.can_cast()
# when bumping to numpy 1.7.0: get rid of if/else when loading npz file to PPSD
# and get rid of helper function _np_copy_astype() in obspy/io/mseed/core.py
INSTALL_REQUIRES = [
    'future>=0.12.4',
    'numpy>=1.6.1',
    'scipy>=0.9.0',
    'matplotlib>=1.1.0',
    'lxml',
    'setuptools',
    'sqlalchemy',
    'decorator',
    'requests']
EXTRAS_REQUIRE = {
    'tests': ['flake8>=2', 'pyimgur', 'pyproj', 'pep8-naming'],
    # arclink decryption also works with: pycrypto, m2crypto, pycryptodome
    'arclink': ['cryptography'],
    'io.shapefile': ['pyshp'],
    }
# PY2
if sys.version_info[0] == 2:
    EXTRAS_REQUIRE['tests'].append('mock')

ENTRY_POINTS = {
    'console_scripts': [
        'obspy-flinn-engdahl = obspy.scripts.flinnengdahl:main',
        'obspy-runtests = obspy.scripts.runtests:main',
        'obspy-reftek-rescue = obspy.scripts.reftekrescue:main',
        'obspy-print = obspy.scripts._print:main',
        'obspy-sds-report = obspy.scripts.sds_html_report:main',
        'obspy-indexer = obspy.db.scripts.indexer:main',
        'obspy-scan = obspy.imaging.scripts.scan:main',
        'obspy-plot = obspy.imaging.scripts.plot:main',
        'obspy-mopad = obspy.imaging.scripts.mopad:main',
        'obspy-mseed-recordanalyzer = '
        'obspy.io.mseed.scripts.recordanalyzer:main',
        'obspy-dataless2xseed = obspy.io.xseed.scripts.dataless2xseed:main',
        'obspy-xseed2dataless = obspy.io.xseed.scripts.xseed2dataless:main',
        'obspy-dataless2resp = obspy.io.xseed.scripts.dataless2resp:main',
        ],
    'obspy.plugin.waveform': [
        'TSPAIR = obspy.io.ascii.core',
        'SLIST = obspy.io.ascii.core',
        'PICKLE = obspy.core.stream',
        'CSS = obspy.io.css.core',
        'WIN = obspy.io.win.core',
        'KINEMETRICS_EVT = obspy.io.kinemetrics.core',
        'GSE1 = obspy.io.gse2.core',
        'GSE2 = obspy.io.gse2.core',
        'MSEED = obspy.io.mseed.core',
        'NNSA_KB_CORE = obspy.io.css.core',
        'PDAS = obspy.io.pdas.core',
        'SAC = obspy.io.sac.core',
        'SACXY = obspy.io.sac.core',
        'Y = obspy.io.y.core',
        'SEG2 = obspy.io.seg2.seg2',
        'SEGY = obspy.io.segy.core',
        'SU = obspy.io.segy.core',
        'SEISAN = obspy.io.seisan.core',
        'Q = obspy.io.sh.core',
        'SH_ASC = obspy.io.sh.core',
        'WAV = obspy.io.wav.core',
        'AH = obspy.io.ah.core',
        'KNET = obspy.io.nied.knet',
        'GCF = obspy.io.gcf.core',
        'REFTEK130 = obspy.io.reftek.core',
        'RG16 = obspy.io.rg16.core',
        'DMX = obspy.io.dmx.core',
        ],
    'obspy.plugin.waveform.TSPAIR': [
        'isFormat = obspy.io.ascii.core:_is_tspair',
        'readFormat = obspy.io.ascii.core:_read_tspair',
        'writeFormat = obspy.io.ascii.core:_write_tspair',
        ],
    'obspy.plugin.waveform.SLIST': [
        'isFormat = obspy.io.ascii.core:_is_slist',
        'readFormat = obspy.io.ascii.core:_read_slist',
        'writeFormat = obspy.io.ascii.core:_write_slist',
        ],
    'obspy.plugin.waveform.PICKLE': [
        'isFormat = obspy.core.stream:_is_pickle',
        'readFormat = obspy.core.stream:_read_pickle',
        'writeFormat = obspy.core.stream:_write_pickle',
        ],
    'obspy.plugin.waveform.CSS': [
        'isFormat = obspy.io.css.core:_is_css',
        'readFormat = obspy.io.css.core:_read_css',
        ],
    'obspy.plugin.waveform.NNSA_KB_CORE': [
        'isFormat = obspy.io.css.core:_is_nnsa_kb_core',
        'readFormat = obspy.io.css.core:_read_nnsa_kb_core',
        ],
    'obspy.plugin.waveform.WIN': [
        'isFormat = obspy.io.win.core:_is_win',
        'readFormat = obspy.io.win.core:_read_win',
        ],
    'obspy.plugin.waveform.KINEMETRICS_EVT': [
        'isFormat = obspy.io.kinemetrics.core:is_evt',
        'readFormat = obspy.io.kinemetrics.core:read_evt',
        ],
    'obspy.plugin.waveform.GSE1': [
        'isFormat = obspy.io.gse2.core:_is_gse1',
        'readFormat = obspy.io.gse2.core:_read_gse1',
        ],
    'obspy.plugin.waveform.GSE2': [
        'isFormat = obspy.io.gse2.core:_is_gse2',
        'readFormat = obspy.io.gse2.core:_read_gse2',
        'writeFormat = obspy.io.gse2.core:_write_gse2',
        ],
    'obspy.plugin.waveform.MSEED': [
        'isFormat = obspy.io.mseed.core:_is_mseed',
        'readFormat = obspy.io.mseed.core:_read_mseed',
        'writeFormat = obspy.io.mseed.core:_write_mseed',
        ],
    'obspy.plugin.waveform.PDAS': [
        'isFormat = obspy.io.pdas.core:_is_pdas',
        'readFormat = obspy.io.pdas.core:_read_pdas',
        ],
    'obspy.plugin.waveform.SAC': [
        'isFormat = obspy.io.sac.core:_is_sac',
        'readFormat = obspy.io.sac.core:_read_sac',
        'writeFormat = obspy.io.sac.core:_write_sac',
        ],
    'obspy.plugin.waveform.SACXY': [
        'isFormat = obspy.io.sac.core:_is_sac_xy',
        'readFormat = obspy.io.sac.core:_read_sac_xy',
        'writeFormat = obspy.io.sac.core:_write_sac_xy',
        ],
    'obspy.plugin.waveform.SEG2': [
        'isFormat = obspy.io.seg2.seg2:_is_seg2',
        'readFormat = obspy.io.seg2.seg2:_read_seg2',
        ],
    'obspy.plugin.waveform.SEGY': [
        'isFormat = obspy.io.segy.core:_is_segy',
        'readFormat = obspy.io.segy.core:_read_segy',
        'writeFormat = obspy.io.segy.core:_write_segy',
        ],
    'obspy.plugin.waveform.SU': [
        'isFormat = obspy.io.segy.core:_is_su',
        'readFormat = obspy.io.segy.core:_read_su',
        'writeFormat = obspy.io.segy.core:_write_su',
        ],
    'obspy.plugin.waveform.SEISAN': [
        'isFormat = obspy.io.seisan.core:_is_seisan',
        'readFormat = obspy.io.seisan.core:_read_seisan',
        ],
    'obspy.plugin.waveform.Q': [
        'isFormat = obspy.io.sh.core:_is_q',
        'readFormat = obspy.io.sh.core:_read_q',
        'writeFormat = obspy.io.sh.core:_write_q',
        ],
    'obspy.plugin.waveform.SH_ASC': [
        'isFormat = obspy.io.sh.core:_is_asc',
        'readFormat = obspy.io.sh.core:_read_asc',
        'writeFormat = obspy.io.sh.core:_write_asc',
        ],
    'obspy.plugin.waveform.WAV': [
        'isFormat = obspy.io.wav.core:_is_wav',
        'readFormat = obspy.io.wav.core:_read_wav',
        'writeFormat = obspy.io.wav.core:_write_wav',
        ],
    'obspy.plugin.waveform.Y': [
        'isFormat = obspy.io.y.core:_is_y',
        'readFormat = obspy.io.y.core:_read_y',
        ],
    'obspy.plugin.waveform.AH': [
        'isFormat = obspy.io.ah.core:_is_ah',
        'readFormat = obspy.io.ah.core:_read_ah',
        'writeFormat = obspy.io.ah.core:_write_ah1'
        ],
    'obspy.plugin.waveform.KNET': [
        'isFormat = obspy.io.nied.knet:_is_knet_ascii',
        'readFormat = obspy.io.nied.knet:_read_knet_ascii',
        ],
    'obspy.plugin.waveform.GCF': [
        'isFormat = obspy.io.gcf.core:_is_gcf',
        'readFormat = obspy.io.gcf.core:_read_gcf',
        ],
    'obspy.plugin.waveform.REFTEK130': [
        'isFormat = obspy.io.reftek.core:_is_reftek130',
        'readFormat = obspy.io.reftek.core:_read_reftek130',
        ],
    'obspy.plugin.waveform.RG16': [
        'isFormat = obspy.io.rg16.core:_is_rg16',
        'readFormat = obspy.io.rg16.core:_read_rg16',
    ],
    'obspy.plugin.waveform.DMX': [
        'isFormat = obspy.io.dmx.core:_is_dmx',
        'readFormat = obspy.io.dmx.core:_read_dmx',
    ],
    'obspy.plugin.event': [
        'QUAKEML = obspy.io.quakeml.core',
        'SC3ML = obspy.io.seiscomp.event',
        'ZMAP = obspy.io.zmap.core',
        'MCHEDR = obspy.io.pde.mchedr',
        'JSON = obspy.io.json.core',
        'NDK = obspy.io.ndk.core',
        'NLLOC_HYP = obspy.io.nlloc.core',
        'NLLOC_OBS = obspy.io.nlloc.core',
        'NORDIC = obspy.io.nordic.core',
        'CNV = obspy.io.cnv.core',
        'CMTSOLUTION = obspy.io.cmtsolution.core',
        'SCARDEC = obspy.io.scardec.core',
        'SHAPEFILE = obspy.io.shapefile.core',
        'KML = obspy.io.kml.core',
        'FNETMT = obspy.io.nied.fnetmt',
        'GSE2 = obspy.io.gse2.bulletin',
        'IMS10BULLETIN = obspy.io.iaspei.core',
        'EVT = obspy.io.sh.evt',
        'FOCMEC = obspy.io.focmec.core',
        'HYPODDPHA = obspy.io.hypodd.pha'
        ],
    'obspy.plugin.event.QUAKEML': [
        'isFormat = obspy.io.quakeml.core:_is_quakeml',
        'readFormat = obspy.io.quakeml.core:_read_quakeml',
        'writeFormat = obspy.io.quakeml.core:_write_quakeml',
        ],
    'obspy.plugin.event.SC3ML': [
        'isFormat = obspy.io.seiscomp.core:_is_sc3ml',
        'readFormat = obspy.io.seiscomp.event:_read_sc3ml',
        'writeFormat = obspy.io.seiscomp.event:_write_sc3ml',
        ],
    'obspy.plugin.event.MCHEDR': [
        'isFormat = obspy.io.pde.mchedr:_is_mchedr',
        'readFormat = obspy.io.pde.mchedr:_read_mchedr',
        ],
    'obspy.plugin.event.JSON': [
        'writeFormat = obspy.io.json.core:_write_json',
        ],
    'obspy.plugin.event.ZMAP': [
        'isFormat = obspy.io.zmap.core:_is_zmap',
        'readFormat = obspy.io.zmap.core:_read_zmap',
        'writeFormat = obspy.io.zmap.core:_write_zmap',
        ],
    'obspy.plugin.event.CNV': [
        'writeFormat = obspy.io.cnv.core:_write_cnv',
        ],
    'obspy.plugin.event.NDK': [
        'isFormat = obspy.io.ndk.core:_is_ndk',
        'readFormat = obspy.io.ndk.core:_read_ndk',
        ],
    'obspy.plugin.event.NLLOC_HYP': [
        'isFormat = obspy.io.nlloc.core:is_nlloc_hyp',
        'readFormat = obspy.io.nlloc.core:read_nlloc_hyp',
        ],
    'obspy.plugin.event.NLLOC_OBS': [
        'writeFormat = obspy.io.nlloc.core:write_nlloc_obs',
        ],
    'obspy.plugin.event.NORDIC': [
        'writeFormat = obspy.io.nordic.core:write_select',
        'readFormat = obspy.io.nordic.core:read_nordic',
        'isFormat = obspy.io.nordic.core:_is_sfile'
        ],
    'obspy.plugin.event.CMTSOLUTION': [
        'isFormat = obspy.io.cmtsolution.core:_is_cmtsolution',
        'readFormat = obspy.io.cmtsolution.core:_read_cmtsolution',
        'writeFormat = obspy.io.cmtsolution.core:_write_cmtsolution'
        ],
    'obspy.plugin.event.SCARDEC': [
        'isFormat = obspy.io.scardec.core:_is_scardec',
        'readFormat = obspy.io.scardec.core:_read_scardec',
        'writeFormat = obspy.io.scardec.core:_write_scardec'
        ],
    'obspy.plugin.event.FNETMT': [
        'isFormat = obspy.io.nied.fnetmt:_is_fnetmt_catalog',
        'readFormat = obspy.io.nied.fnetmt:_read_fnetmt_catalog',
        ],
    'obspy.plugin.event.GSE2': [
        'isFormat = obspy.io.gse2.bulletin:_is_gse2',
        'readFormat = obspy.io.gse2.bulletin:_read_gse2',
        ],
    'obspy.plugin.event.SHAPEFILE': [
        'writeFormat = obspy.io.shapefile.core:_write_shapefile',
        ],
    'obspy.plugin.event.KML': [
        'writeFormat = obspy.io.kml.core:_write_kml',
        ],
    'obspy.plugin.event.IMS10BULLETIN': [
        'isFormat = obspy.io.iaspei.core:_is_ims10_bulletin',
        'readFormat = obspy.io.iaspei.core:_read_ims10_bulletin',
        ],
    'obspy.plugin.event.EVT': [
        'isFormat = obspy.io.sh.evt:_is_evt',
        'readFormat = obspy.io.sh.evt:_read_evt',
        ],
    'obspy.plugin.event.FOCMEC': [
        'isFormat = obspy.io.focmec.core:_is_focmec',
        'readFormat = obspy.io.focmec.core:_read_focmec',
        ],
    'obspy.plugin.event.HYPODDPHA': [
        'isFormat = obspy.io.hypodd.pha:_is_pha',
        'readFormat = obspy.io.hypodd.pha:_read_pha',
        ],
    'obspy.plugin.inventory': [
        'STATIONXML = obspy.io.stationxml.core',
        'INVENTORYXML = obspy.io.arclink.inventory',
        'SC3ML = obspy.io.seiscomp.inventory',
        'SACPZ = obspy.io.sac.sacpz',
        'CSS = obspy.io.css.station',
        'SHAPEFILE = obspy.io.shapefile.core',
        'STATIONTXT = obspy.io.stationtxt.core',
        'KML = obspy.io.kml.core',
        'SEED = obspy.io.xseed.core',
        'XSEED = obspy.io.xseed.core',
        'RESP = obspy.io.xseed.core',
        ],
    'obspy.plugin.inventory.STATIONXML': [
        'isFormat = obspy.io.stationxml.core:_is_stationxml',
        'readFormat = obspy.io.stationxml.core:_read_stationxml',
        'writeFormat = obspy.io.stationxml.core:_write_stationxml',
        ],
    'obspy.plugin.inventory.INVENTORYXML': [
        'isFormat = obspy.io.arclink.inventory:_is_inventory_xml',
        'readFormat = obspy.io.arclink.inventory:_read_inventory_xml',
        ],
    'obspy.plugin.inventory.SC3ML': [
        'isFormat = obspy.io.seiscomp.core:_is_sc3ml',
        'readFormat = obspy.io.seiscomp.inventory:_read_sc3ml',
        ],
    'obspy.plugin.inventory.SACPZ': [
        'writeFormat = obspy.io.sac.sacpz:_write_sacpz',
        ],
    'obspy.plugin.inventory.CSS': [
        'writeFormat = obspy.io.css.station:_write_css',
        ],
    'obspy.plugin.inventory.SHAPEFILE': [
        'writeFormat = obspy.io.shapefile.core:_write_shapefile',
        ],
    'obspy.plugin.inventory.STATIONTXT': [
        'isFormat = obspy.io.stationtxt.core:is_fdsn_station_text_file',
        'readFormat = '
        'obspy.io.stationtxt.core:read_fdsn_station_text_file',
        'writeFormat = obspy.io.stationtxt.core:_write_stationtxt',
        ],
    'obspy.plugin.inventory.KML': [
        'writeFormat = obspy.io.kml.core:_write_kml',
        ],
    'obspy.plugin.inventory.SEED': [
        'isFormat = obspy.io.xseed.core:_is_seed',
        'readFormat = obspy.io.xseed.core:_read_seed',
    ],
    'obspy.plugin.inventory.XSEED': [
        'isFormat = obspy.io.xseed.core:_is_xseed',
        'readFormat = obspy.io.xseed.core:_read_xseed',
    ],
    'obspy.plugin.inventory.RESP': [
        'isFormat = obspy.io.xseed.core:_is_resp',
        'readFormat = obspy.io.xseed.core:_read_resp',
    ],
    'obspy.plugin.detrend': [
        'linear = scipy.signal:detrend',
        'constant = scipy.signal:detrend',
        'demean = scipy.signal:detrend',
        'simple = obspy.signal.detrend:simple',
        'polynomial = obspy.signal.detrend:polynomial',
        'spline = obspy.signal.detrend:spline'
        ],
    'obspy.plugin.differentiate': [
        'gradient = numpy:gradient',
        ],
    'obspy.plugin.integrate': [
        'cumtrapz = '
        'obspy.signal.differentiate_and_integrate:integrate_cumtrapz',
        'spline = '
        'obspy.signal.differentiate_and_integrate:integrate_spline',
        ],
    'obspy.plugin.filter': [
        'bandpass = obspy.signal.filter:bandpass',
        'bandstop = obspy.signal.filter:bandstop',
        'lowpass = obspy.signal.filter:lowpass',
        'highpass = obspy.signal.filter:highpass',
        'lowpass_cheby_2 = obspy.signal.filter:lowpass_cheby_2',
        'lowpass_fir = obspy.signal.filter:lowpass_FIR',
        'remez_fir = obspy.signal.filter:remez_FIR',
        ],
    'obspy.plugin.interpolate': [
        'interpolate_1d = obspy.signal.interpolation:interpolate_1d',
        'weighted_average_slopes = '
        'obspy.signal.interpolation:weighted_average_slopes',
        'lanczos = obspy.signal.interpolation:lanczos_interpolation'
        ],
    'obspy.plugin.rotate': [
        'rotate_ne_rt = obspy.signal.rotate:rotate_ne_rt',
        'rotate_rt_ne = obspy.signal.rotate:rotate_rt_ne',
        'rotate_zne_lqt = obspy.signal.rotate:rotate_zne_lqt',
        'rotate_lqt_zne = obspy.signal.rotate:rotate_lqt_zne'
        ],
    'obspy.plugin.taper': [
        'cosine = obspy.signal.invsim:cosine_taper',
        'barthann = scipy.signal:barthann',
        'bartlett = scipy.signal:bartlett',
        'blackman = scipy.signal:blackman',
        'blackmanharris = scipy.signal:blackmanharris',
        'bohman = scipy.signal:bohman',
        'boxcar = scipy.signal:boxcar',
        'chebwin = scipy.signal:chebwin',
        'flattop = scipy.signal:flattop',
        'gaussian = scipy.signal:gaussian',
        'general_gaussian = scipy.signal:general_gaussian',
        'hamming = scipy.signal:hamming',
        'hann = scipy.signal:hann',
        'kaiser = scipy.signal:kaiser',
        'nuttall = scipy.signal:nuttall',
        'parzen = scipy.signal:parzen',
        'slepian = scipy.signal:slepian',
        'triang = scipy.signal:triang',
        ],
    'obspy.plugin.trigger': [
        'recstalta = obspy.signal.trigger:recursive_sta_lta',
        'carlstatrig = obspy.signal.trigger:carl_sta_trig',
        'classicstalta = obspy.signal.trigger:classic_sta_lta',
        'delayedstalta = obspy.signal.trigger:delayed_sta_lta',
        'zdetect = obspy.signal.trigger:z_detect',
        'recstaltapy = obspy.signal.trigger:recursive_sta_lta_py',
        'classicstaltapy = obspy.signal.trigger:classic_sta_lta_py',
        ],
    'obspy.db.feature': [
        'minmax_amplitude = obspy.db.feature:MinMaxAmplitudeFeature',
        'bandpass_preview = obspy.db.feature:BandpassPreviewFeature',
        ],
    }


def find_packages():
    """
    Simple function to find all modules under the current folder.
    """
    modules = []
    for dirpath, _, filenames in os.walk(os.path.join(SETUP_DIRECTORY,
                                                      "obspy")):
        if "__init__.py" in filenames:
            modules.append(os.path.relpath(dirpath, SETUP_DIRECTORY))
    return [_i.replace(os.sep, ".") for _i in modules]


# monkey patches for MS Visual Studio
if IS_MSVC:
    import distutils
    from distutils.msvc9compiler import MSVCCompiler

    # for Python 2.x only -> support library paths containing spaces
    if distutils.__version__.startswith('2.'):
        def _library_dir_option(self, dir):
            return '/LIBPATH:"%s"' % (dir)

        MSVCCompiler.library_dir_option = _library_dir_option

    # remove 'init' entry in exported symbols
    def _get_export_symbols(self, ext):
        return ext.export_symbols
    from distutils.command.build_ext import build_ext
    build_ext.get_export_symbols = _get_export_symbols


# helper function for collecting export symbols from .def files
def export_symbols(*path):
    lines = open(os.path.join(*path), 'r').readlines()[2:]
    return [s.strip() for s in lines if s.strip() != '']


# adds --with-system-libs command-line option if possible
def add_features():
    if 'setuptools' not in sys.modules or not hasattr(setuptools, 'Feature'):
        return {}

    class ExternalLibFeature(setuptools.Feature):
        def __init__(self, *args, **kwargs):
            self.name = kwargs['name']
            setuptools.Feature.__init__(self, *args, **kwargs)

        def include_in(self, dist):
            globals()[self.name] = True

        def exclude_from(self, dist):
            globals()[self.name] = False

    return {
        'system-libs': setuptools.Feature(
            'use of system C libraries',
            standard=False,
            require_features=('system-evalresp', 'system-libmseed')
        ),
        'system-evalresp': ExternalLibFeature(
            'use of system evalresp library',
            standard=False,
            name='EXTERNAL_EVALRESP'
        ),
        'system-libmseed': ExternalLibFeature(
            'use of system libmseed library',
            standard=False,
            name='EXTERNAL_LIBMSEED'
        )
    }


def configuration(parent_package="", top_path=None):
    """
    Config function mainly used to compile C code.
    """
    config = Configuration("", parent_package, top_path)

    # GSE2
    path = os.path.join("obspy", "io", "gse2", "src", "GSE_UTI")
    files = [os.path.join(path, "gse_functions.c")]
    # compiler specific options
    kwargs = {}
    if IS_MSVC:
        # get export symbols
        kwargs['export_symbols'] = export_symbols(path, 'gse_functions.def')
    config.add_extension(_get_lib_name("gse2", add_extension_suffix=False),
                         files, **kwargs)

    # LIBMSEED
    path = os.path.join("obspy", "io", "mseed", "src")
    files = [os.path.join(path, "obspy-readbuffer.c")]
    if not EXTERNAL_LIBMSEED:
        files += glob.glob(os.path.join(path, "libmseed", "*.c"))
    # compiler specific options
    kwargs = {}
    if IS_MSVC:
        # needed by libmseed lmplatform.h
        kwargs['define_macros'] = [('WIN32', '1')]
        # get export symbols
        kwargs['export_symbols'] = \
            export_symbols(path, 'libmseed', 'libmseed.def')
        kwargs['export_symbols'] += \
            export_symbols(path, 'obspy-readbuffer.def')
    if EXTERNAL_LIBMSEED:
        kwargs['libraries'] = ['mseed']
    config.add_extension(_get_lib_name("mseed", add_extension_suffix=False),
                         files, **kwargs)

    # SEGY
    path = os.path.join("obspy", "io", "segy", "src")
    files = [os.path.join(path, "ibm2ieee.c")]
    # compiler specific options
    kwargs = {}
    if IS_MSVC:
        # get export symbols
        kwargs['export_symbols'] = export_symbols(path, 'libsegy.def')
    config.add_extension(_get_lib_name("segy", add_extension_suffix=False),
                         files, **kwargs)

    # SIGNAL
    path = os.path.join("obspy", "signal", "src")
    files = glob.glob(os.path.join(path, "*.c"))
    # compiler specific options
    kwargs = {}
    if IS_MSVC:
        # get export symbols
        kwargs['export_symbols'] = export_symbols(path, 'libsignal.def')
    config.add_extension(_get_lib_name("signal", add_extension_suffix=False),
                         files, **kwargs)

    # EVALRESP
    path = os.path.join("obspy", "signal", "src")
    if EXTERNAL_EVALRESP:
        files = glob.glob(os.path.join(path, "evalresp", "_obspy*.c"))
    else:
        files = glob.glob(os.path.join(path, "evalresp", "*.c"))
    # compiler specific options
    kwargs = {}
    if IS_MSVC:
        # needed by evalresp evresp.h
        kwargs['define_macros'] = [('WIN32', '1')]
        # get export symbols
        kwargs['export_symbols'] = export_symbols(path, 'libevresp.def')
    if EXTERNAL_EVALRESP:
        kwargs['libraries'] = ['evresp']
    config.add_extension(_get_lib_name("evresp", add_extension_suffix=False),
                         files, **kwargs)

    # TAU
    path = os.path.join("obspy", "taup", "src")
    files = [os.path.join(path, "inner_tau_loops.c")]
    # compiler specific options
    kwargs = {}
    if IS_MSVC:
        # get export symbols
        kwargs['export_symbols'] = export_symbols(path, 'libtau.def')
    config.add_extension(_get_lib_name("tau", add_extension_suffix=False),
                         files, **kwargs)

    add_data_files(config)

    return config


def add_data_files(config):
    """
    Recursively include all non python files
    """
    # python files are included per default, we only include data files
    # here
    EXCLUDE_WILDCARDS = ['*.py', '*.pyc', '*.pyo', '*.pdf', '.git*']
    EXCLUDE_DIRS = ['src', '__pycache__']
    common_prefix = SETUP_DIRECTORY + os.path.sep
    for root, dirs, files in os.walk(os.path.join(SETUP_DIRECTORY, 'obspy')):
        root = root.replace(common_prefix, '')
        for name in files:
            if any(fnmatch.fnmatch(name, w) for w in EXCLUDE_WILDCARDS):
                continue
            config.add_data_files(os.path.join(root, name))
        for folder in EXCLUDE_DIRS:
            if folder in dirs:
                dirs.remove(folder)

    # Force include the contents of some directories.
    FORCE_INCLUDE_DIRS = [
        os.path.join(SETUP_DIRECTORY, 'obspy', 'io', 'mseed', 'src',
                     'libmseed', 'test')]

    for folder in FORCE_INCLUDE_DIRS:
        for root, _, files in os.walk(folder):
            for filename in files:
                config.add_data_files(
                    os.path.relpath(os.path.join(root, filename),
                                    SETUP_DIRECTORY))


# Auto-generate man pages from --help output
class Help2ManBuild(build):
    description = "Run help2man on scripts to produce man pages"

    def finalize_options(self):
        build.finalize_options(self)
        self.help2man = find_executable('help2man')
        if not self.help2man:
            raise DistutilsSetupError('Building man pages requires help2man.')

    def run(self):
        mandir = os.path.join(self.build_base, 'man')
        self.mkpath(mandir)

        from pkg_resources import EntryPoint
        for entrypoint in ENTRY_POINTS['console_scripts']:
            ep = EntryPoint.parse(entrypoint)
            if not ep.module_name.startswith('obspy'):
                continue

            output = os.path.join(mandir, ep.name + '.1')
            print('Generating %s ...' % (output))
            exec_command([self.help2man,
                          '--no-info', '--no-discard-stderr',
                          '--output', output,
                          '"%s -m %s"' % (sys.executable,
                                          ep.module_name)])


class Help2ManInstall(install):
    description = 'Install man pages generated by help2man'
    user_options = install.user_options + [
        ('manprefix=', None, 'MAN Prefix Path')
    ]

    def initialize_options(self):
        self.manprefix = None
        install.initialize_options(self)

    def finalize_options(self):
        if self.manprefix is None:
            self.manprefix = os.path.join('share', 'man')
        install.finalize_options(self)

    def run(self):
        if not self.skip_build:
            self.run_command('build_man')

        srcdir = os.path.join(self.build_base, 'man')
        mandir = os.path.join(self.install_base, self.manprefix, 'man1')
        if self.root is not None:
            mandir = change_root(self.root, mandir)
        self.mkpath(mandir)
        self.copy_tree(srcdir, mandir)


def setupPackage():
    # setup package
    setup(
        name='obspy',
        version=get_git_version(),
        description=DOCSTRING[1],
        long_description="\n".join(DOCSTRING[3:]),
        url="https://www.obspy.org",
        author='The ObsPy Development Team',
        author_email='devs@obspy.org',
        license='GNU Lesser General Public License, Version 3 (LGPLv3)',
        platforms='OS Independent',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: GNU Library or ' +
                'Lesser General Public License (LGPL)',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics'],
        keywords=KEYWORDS,
        packages=find_packages(),
        namespace_packages=[],
        zip_safe=False,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        features=add_features(),
        # this is needed for "easy_install obspy==dev"
        download_url=("https://github.com/obspy/obspy/zipball/master"
                      "#egg=obspy=dev"),
        include_package_data=True,
        entry_points=ENTRY_POINTS,
        ext_package='obspy.lib',
        cmdclass={
            'build_man': Help2ManBuild,
            'install_man': Help2ManInstall
        },
        configuration=configuration)


if __name__ == '__main__':
    # clean --all does not remove extensions automatically
    if 'clean' in sys.argv and '--all' in sys.argv:
        import shutil
        # delete complete build directory
        path = os.path.join(SETUP_DIRECTORY, 'build')
        try:
            shutil.rmtree(path)
        except Exception:
            pass
        # delete all shared libs from lib directory
        path = os.path.join(SETUP_DIRECTORY, 'obspy', 'lib')
        for filename in glob.glob(path + os.sep + '*.pyd'):
            try:
                os.remove(filename)
            except Exception:
                pass
        for filename in glob.glob(path + os.sep + '*.so'):
            try:
                os.remove(filename)
            except Exception:
                pass
        path = os.path.join(SETUP_DIRECTORY, 'obspy', 'taup', 'data', 'models')
        try:
            shutil.rmtree(path)
        except Exception:
            pass
    else:
        setupPackage()
