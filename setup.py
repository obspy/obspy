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

For more information visit http://www.obspy.org.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
# Importing setuptools monkeypatches some of distutils commands so things like
# 'python setup.py develop' work. Wrap in try/except so it is not an actual
# dependency. Inplace installation with pip works also without importing
# setuptools.
try:
    import setuptools  # @UnusedImport # NOQA
except:
    pass

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from numpy.distutils.ccompiler import get_default_compiler

import glob
import inspect
import fnmatch
import os
import platform
import sys


# Directory of the current file in the (hopefully) most reliable way
# possible, according to krischer
SETUP_DIRECTORY = os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe())))

# Import the version string.
UTIL_PATH = os.path.join(SETUP_DIRECTORY, "obspy", "core", "util")
sys.path.insert(0, UTIL_PATH)
from version import get_git_version  # @UnresolvedImport
sys.path.pop(0)

LOCAL_PATH = os.path.join(SETUP_DIRECTORY, "setup.py")
DOCSTRING = __doc__.split("\n")

# check for MSVC
if platform.system() == "Windows" and (
        'msvc' in sys.argv or '-c' not in sys.argv and get_default_compiler()
        == 'msvc'):
    IS_MSVC = True
else:
    IS_MSVC = False

# package specific settings
KEYWORDS = [
    'ArcLink', 'array', 'array analysis', 'ASC', 'beachball',
    'beamforming', 'cross correlation', 'database', 'dataless',
    'Dataless SEED', 'datamark', 'earthquakes', 'Earthworm', 'EIDA',
    'envelope', 'events', 'FDSN', 'features', 'filter', 'focal mechanism',
    'GSE1', 'GSE2', 'hob', 'iapsei-tau', 'imaging', 'instrument correction',
    'instrument simulation', 'IRIS', 'magnitude', 'MiniSEED', 'misfit',
    'mopad', 'MSEED', 'NERA', 'NERIES', 'observatory', 'ORFEUS', 'picker',
    'processing', 'PQLX', 'Q', 'real time', 'realtime', 'RESP',
    'response file', 'RT', 'SAC', 'SEED', 'SeedLink', 'SEG-2', 'SEG Y',
    'SEISAN', 'SeisHub', 'Seismic Handler', 'seismology', 'seismogram',
    'seismograms', 'signal', 'slink', 'spectrogram', 'StationXML', 'taper',
    'taup', 'travel time', 'trigger', 'VERCE', 'WAV', 'waveform', 'WaveServer',
    'WaveServerV', 'WebDC', 'web service', 'Winston', 'XML-SEED', 'XSEED']
INSTALL_REQUIRES = [
    'future',
    'numpy>1.0.0',
    'scipy',
    'matplotlib',
    'lxml',
    'sqlalchemy']
    #'suds']
    #'suds>=0.4.0']
EXTRAS_REQUIRE = {
    'tests': ['flake8>=2',
              'nose',
              'mock']}
ENTRY_POINTS = {
    'console_scripts': [
        'obspy-runtests = obspy.core.scripts.runtests:main',
        'obspy-reftek-rescue = obspy.core.scripts.reftekrescue:main',
        'obspy-print = obspy.core.scripts.print:main',
        'obspy-indexer = obspy.db.scripts.indexer:main',
        'obspy-scan = obspy.imaging.scripts.scan:main',
        'obspy-plot = obspy.imaging.scripts.plot:main',
        'obspy-mopad = obspy.imaging.scripts.mopad:main',
        'obspy-mseed-recordanalyzer = obspy.mseed.scripts.recordanalyzer:main',
        'obspy-dataless2xseed = obspy.xseed.scripts.dataless2xseed:main',
        'obspy-xseed2dataless = obspy.xseed.scripts.xseed2dataless:main',
        'obspy-dataless2resp = obspy.xseed.scripts.dataless2resp:main',
    ],
    'obspy.plugin.waveform': [
        'TSPAIR = obspy.core.ascii',
        'SLIST = obspy.core.ascii',
        'PICKLE = obspy.core.stream',
        'CSS = obspy.css.core',
        'DATAMARK = obspy.datamark.core',
        'GSE1 = obspy.gse2.core',
        'GSE2 = obspy.gse2.core',
        'MSEED = obspy.mseed.core',
        'SAC = obspy.sac.core',
        'SACXY = obspy.sac.core',
        'Y = obspy.y.core',
        'SEG2 = obspy.seg2.seg2',
        'SEGY = obspy.segy.core',
        'SU = obspy.segy.core',
        'SEISAN = obspy.seisan.core',
        'Q = obspy.sh.core',
        'SH_ASC = obspy.sh.core',
        'WAV = obspy.wav.core',
    ],
    'obspy.plugin.waveform.TSPAIR': [
        'isFormat = obspy.core.ascii:isTSPAIR',
        'readFormat = obspy.core.ascii:readTSPAIR',
        'writeFormat = obspy.core.ascii:writeTSPAIR',
    ],
    'obspy.plugin.waveform.SLIST': [
        'isFormat = obspy.core.ascii:isSLIST',
        'readFormat = obspy.core.ascii:readSLIST',
        'writeFormat = obspy.core.ascii:writeSLIST',
    ],
    'obspy.plugin.waveform.PICKLE': [
        'isFormat = obspy.core.stream:isPickle',
        'readFormat = obspy.core.stream:readPickle',
        'writeFormat = obspy.core.stream:writePickle',
    ],
    'obspy.plugin.waveform.CSS': [
        'isFormat = obspy.css.core:isCSS',
        'readFormat = obspy.css.core:readCSS',
    ],
    'obspy.plugin.waveform.DATAMARK': [
        'isFormat = obspy.datamark.core:isDATAMARK',
        'readFormat = obspy.datamark.core:readDATAMARK',
    ],
    'obspy.plugin.waveform.GSE1': [
        'isFormat = obspy.gse2.core:isGSE1',
        'readFormat = obspy.gse2.core:readGSE1',
    ],
    'obspy.plugin.waveform.GSE2': [
        'isFormat = obspy.gse2.core:isGSE2',
        'readFormat = obspy.gse2.core:readGSE2',
        'writeFormat = obspy.gse2.core:writeGSE2',
    ],
    'obspy.plugin.waveform.MSEED': [
        'isFormat = obspy.mseed.core:isMSEED',
        'readFormat = obspy.mseed.core:readMSEED',
        'writeFormat = obspy.mseed.core:writeMSEED',
    ],
    'obspy.plugin.waveform.SAC': [
        'isFormat = obspy.sac.core:isSAC',
        'readFormat = obspy.sac.core:readSAC',
        'writeFormat = obspy.sac.core:writeSAC',
    ],
    'obspy.plugin.waveform.SACXY': [
        'isFormat = obspy.sac.core:isSACXY',
        'readFormat = obspy.sac.core:readSACXY',
        'writeFormat = obspy.sac.core:writeSACXY',
    ],
    'obspy.plugin.waveform.SEG2': [
        'isFormat = obspy.seg2.seg2:isSEG2',
        'readFormat = obspy.seg2.seg2:readSEG2',
    ],
    'obspy.plugin.waveform.SEGY': [
        'isFormat = obspy.segy.core:isSEGY',
        'readFormat = obspy.segy.core:readSEGY',
        'writeFormat = obspy.segy.core:writeSEGY',
    ],
    'obspy.plugin.waveform.SU': [
        'isFormat = obspy.segy.core:isSU',
        'readFormat = obspy.segy.core:readSU',
        'writeFormat = obspy.segy.core:writeSU',
    ],
    'obspy.plugin.waveform.SEISAN': [
        'isFormat = obspy.seisan.core:isSEISAN',
        'readFormat = obspy.seisan.core:readSEISAN',
    ],
    'obspy.plugin.waveform.Q': [
        'isFormat = obspy.sh.core:isQ',
        'readFormat = obspy.sh.core:readQ',
        'writeFormat = obspy.sh.core:writeQ',
    ],
    'obspy.plugin.waveform.SH_ASC': [
        'isFormat = obspy.sh.core:isASC',
        'readFormat = obspy.sh.core:readASC',
        'writeFormat = obspy.sh.core:writeASC',
    ],
    'obspy.plugin.waveform.WAV': [
        'isFormat = obspy.wav.core:isWAV',
        'readFormat = obspy.wav.core:readWAV',
        'writeFormat = obspy.wav.core:writeWAV',
    ],
    'obspy.plugin.waveform.Y': [
        'isFormat = obspy.y.core:isY',
        'readFormat = obspy.y.core:readY',
    ],
    'obspy.plugin.event': [
        'QUAKEML = obspy.core.quakeml',
        'MCHEDR = obspy.pde.mchedr',
        'JSON = obspy.core.json.core',
    ],
    'obspy.plugin.event.QUAKEML': [
        'isFormat = obspy.core.quakeml:isQuakeML',
        'readFormat = obspy.core.quakeml:readQuakeML',
        'writeFormat = obspy.core.quakeml:writeQuakeML',
    ],
    'obspy.plugin.event.MCHEDR': [
        'isFormat = obspy.pde.mchedr:isMchedr',
        'readFormat = obspy.pde.mchedr:readMchedr',
    ],
    'obspy.plugin.event.JSON': [
        'writeFormat = obspy.core.json.core:writeJSON',
    ],
    'obspy.plugin.inventory': [
        'STATIONXML = obspy.station.stationxml',
    ],
    'obspy.plugin.inventory.STATIONXML': [
        'isFormat = obspy.station.stationxml:is_StationXML',
        'readFormat = obspy.station.stationxml:read_StationXML',
        'writeFormat = obspy.station.stationxml:write_StationXML',
    ],
    'obspy.plugin.detrend': [
        'linear = scipy.signal:detrend',
        'constant = scipy.signal:detrend',
        'demean = scipy.signal:detrend',
        'simple = obspy.signal.detrend:simple',
    ],
    'obspy.plugin.differentiate': [
        'gradient = numpy:gradient',
    ],
    'obspy.plugin.filter': [
        'bandpass = obspy.signal.filter:bandpass',
        'bandstop = obspy.signal.filter:bandstop',
        'lowpass = obspy.signal.filter:lowpass',
        'highpass = obspy.signal.filter:highpass',
        'lowpassCheby2 = obspy.signal.filter:lowpassCheby2',
        'lowpassFIR = obspy.signal.filter:lowpassFIR',
        'remezFIR = obspy.signal.filter:remezFIR',
    ],
    'obspy.plugin.integrate': [
        'trapz = scipy.integrate:trapz',
        'cumtrapz = scipy.integrate:cumtrapz',
        'simps = scipy.integrate:simps',
        'romb = scipy.integrate:romb',
    ],
    'obspy.plugin.rotate': [
        'rotate_NE_RT = obspy.signal:rotate_NE_RT',
        'rotate_RT_NE = obspy.signal:rotate_RT_NE',
        'rotate_ZNE_LQT = obspy.signal:rotate_ZNE_LQT',
        'rotate_LQT_ZNE = obspy.signal:rotate_LQT_ZNE'
    ],
    'obspy.plugin.taper': [
        'cosine = obspy.signal.invsim:cosTaper',
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
        'recstalta = obspy.signal.trigger:recSTALTA',
        'carlstatrig = obspy.signal.trigger:carlSTATrig',
        'classicstalta = obspy.signal.trigger:classicSTALTA',
        'delayedstalta = obspy.signal.trigger:delayedSTALTA',
        'zdetect = obspy.signal.trigger:zDetect',
        'recstaltapy = obspy.signal.trigger:recSTALTAPy',
        'classicstaltapy = obspy.signal.trigger:classicSTALTAPy',
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


def _get_lib_name(lib):
    """
    Helper function to get an architecture and Python version specific library
    filename.
    """
    return "lib%s_%s_%s_py%s" % (
        lib, platform.system(), platform.architecture()[0], "".join(
            [str(i) for i in platform.python_version_tuple()[:2]]))

# monkey patches for MS Visual Studio
if IS_MSVC:
    # support library paths containing spaces
    def _library_dir_option(self, dir):
        return '"/LIBPATH:%s"' % (dir)

    from distutils.msvc9compiler import MSVCCompiler
    MSVCCompiler.library_dir_option = _library_dir_option

    # remove 'init' entry in exported symbols
    def _get_export_symbols(self, ext):
        return ext.export_symbols
    from distutils.command.build_ext import build_ext
    build_ext.get_export_symbols = _get_export_symbols

    # tau shared library has to be compiled with gfortran directly
    def link(self, _target_desc, objects, output_filename,
             *args, **kwargs):  # @UnusedVariable
        # check if 'tau' library is linked
        if 'tau' not in output_filename:
            # otherwise just use the original link method
            return self.original_link(_target_desc, objects, output_filename,
                                      *args, **kwargs)
        if '32' in platform.architecture()[0]:
            taupargs = ["-m32"]
        else:
            taupargs = ["-m64"]
        # ignoring all f2py objects
        objects = objects[2:]
        self.spawn(['gfortran.exe'] +
                   ["-static-libgcc", "-static-libgfortran", "-shared"] +
                   taupargs + objects + ["-o", output_filename])

    MSVCCompiler.original_link = MSVCCompiler.link
    MSVCCompiler.link = link


# helper function for collecting export symbols from .def files
def export_symbols(*path):
    lines = open(os.path.join(*path), 'r').readlines()[2:]
    return [s.strip() for s in lines if s.strip() != '']


def configuration(parent_package="", top_path=None):
    """
    Config function mainly used to compile C and Fortran code.
    """
    config = Configuration("", parent_package, top_path)

    # GSE2
    path = os.path.join(SETUP_DIRECTORY, "obspy", "gse2", "src", "GSE_UTI")
    files = [os.path.join(path, "gse_functions.c")]
    # compiler specific options
    kwargs = {}
    if IS_MSVC:
        # get export symbols
        kwargs['export_symbols'] = export_symbols(path, 'gse_functions.def')
    config.add_extension(_get_lib_name("gse2"), files, **kwargs)

    # LIBMSEED
    path = os.path.join(SETUP_DIRECTORY, "obspy", "mseed", "src")
    files = glob.glob(os.path.join(path, "libmseed", "*.c"))
    files.append(os.path.join(path, "obspy-readbuffer.c"))
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
        # workaround Win32 and MSVC - see issue #64
        if '32' in platform.architecture()[0]:
            kwargs['extra_compile_args'] = ["/fp:strict"]
    config.add_extension(_get_lib_name("mseed"), files, **kwargs)

    # SEGY
    path = os.path.join(SETUP_DIRECTORY, "obspy", "segy", "src")
    files = [os.path.join(path, "ibm2ieee.c")]
    # compiler specific options
    kwargs = {}
    if IS_MSVC:
        # get export symbols
        kwargs['export_symbols'] = export_symbols(path, 'libsegy.def')
    config.add_extension(_get_lib_name("segy"), files, **kwargs)

    # SIGNAL
    path = os.path.join(SETUP_DIRECTORY, "obspy", "signal", "src")
    files = glob.glob(os.path.join(path, "*.c"))
    # compiler specific options
    kwargs = {}
    if IS_MSVC:
        # get export symbols
        kwargs['export_symbols'] = export_symbols(path, 'libsignal.def')
    config.add_extension(_get_lib_name("signal"), files, **kwargs)

    # EVALRESP
    path = os.path.join(SETUP_DIRECTORY, "obspy", "signal", "src")
    files = glob.glob(os.path.join(path, "evalresp", "*.c"))
    # compiler specific options
    kwargs = {}
    if IS_MSVC:
        # needed by evalresp evresp.h
        kwargs['define_macros'] = [('WIN32', '1')]
        # get export symbols
        kwargs['export_symbols'] = export_symbols(path, 'libevresp.def')
    config.add_extension(_get_lib_name("evresp"), files, **kwargs)

    # TAUP
    path = os.path.join(SETUP_DIRECTORY, "obspy", "taup", "src")
    libname = _get_lib_name("tau")
    files = glob.glob(os.path.join(path, "*.f"))
    # compiler specific options
    kwargs = {'libraries': []}
    # XXX: The build subdirectory is difficult to determine if installed
    # via pypi or other means. I could not find a reliable way of doing it.
    new_interface_path = os.path.join("build", libname + os.extsep + "pyf")
    interface_file = os.path.join(path, "_libtau.pyf")
    with open(interface_file, "r") as open_file:
        interface_file = open_file.read()
    # In the original .pyf file the library is called _libtau.
    interface_file = interface_file.replace("_libtau", libname)
    if not os.path.exists("build"):
        os.mkdir("build")
    with open(new_interface_path, "w") as open_file:
        open_file.write(interface_file)
    files.insert(0, new_interface_path)
    # we do not need this when linking with gcc, only when linking with
    # gfortran the option -lgcov is required
    if os.environ.get('OBSPY_C_COVERAGE', ""):
        kwargs['libraries'].append('gcov')
    config.add_extension(libname, files, **kwargs)

    add_data_files(config)

    return config


def add_data_files(config):
    """
    Recursively include all non python files
    """
    # python files are included per default, we only include data files
    # here
    EXCLUDE_WILDCARDS = ['*.py', '*.pyc', '*.pyo', '*.pdf']
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


def setupPackage():
    # setup package
    setup(
        name='obspy',
        version=get_git_version(),
        description=DOCSTRING[1],
        long_description="\n".join(DOCSTRING[3:]),
        url="http://www.obspy.org",
        author='The ObsPy Development Team',
        author_email='devs@obspy.org',
        license='GNU Lesser General Public License, Version 3 (LGPLv3)',
        platforms='OS Independent',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: GNU Library or ' +
                'Lesser General Public License (LGPL)',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics'],
        keywords=KEYWORDS,
        packages=find_packages(),
        namespace_packages=[],
        zip_safe=False,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        # this is needed for "easy_install obspy==dev"
        download_url=("https://github.com/obspy/obspy/zipball/master"
                      "#egg=obspy=dev"),
        include_package_data=True,
        entry_points=ENTRY_POINTS,
        ext_package='obspy.lib',
        configuration=configuration)


if __name__ == '__main__':
    # clean --all does not remove extensions automatically
    if 'clean' in sys.argv and '--all' in sys.argv:
        import shutil
        # delete complete build directory
        path = os.path.join(SETUP_DIRECTORY, 'build')
        try:
            shutil.rmtree(path)
        except:
            pass
        # delete all shared libs from lib directory
        path = os.path.join(SETUP_DIRECTORY, 'obspy', 'lib')
        for filename in glob.glob(path + os.sep + '*.pyd'):
            try:
                os.remove(filename)
            except:
                pass
        for filename in glob.glob(path + os.sep + '*.so'):
            try:
                os.remove(filename)
            except:
                pass
    setupPackage()
