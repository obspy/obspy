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

from distutils.ccompiler import get_default_compiler
from distutils.ccompiler import CCompiler
from distutils.errors import DistutilsExecError, CompileError
from distutils.unixccompiler import UnixCCompiler, _darwin_compiler_fixup
from setuptools import find_packages, setup
from setuptools.extension import Extension
import distribute_setup
import glob
import os
import platform
import shutil
import sys


# package specific settings
KEYWORDS = ['ArcLink', 'array', 'array analysis', 'ASC', 'beachball',
    'beamforming', 'cross correlation', 'database', 'dataless',
    'Dataless SEED', 'datamark', 'earthquakes', 'Earthworm', 'EIDA',
    'envelope', 'events', 'features', 'filter', 'focal mechanism', 'GSE1',
    'GSE2', 'hob', 'iapsei-tau', 'imaging', 'instrument correction',
    'instrument simulation', 'IRIS', 'magnitude', 'MiniSEED', 'misfit',
    'mopad', 'MSEED', 'NERA', 'NERIES', 'observatory', 'ORFEUS', 'picker',
    'processing', 'PQLX', 'Q', 'real time', 'realtime', 'RESP',
    'response file', 'RT', 'SAC', 'SEED', 'SeedLink', 'SEG-2', 'SEG Y',
    'SEISAN', 'SeisHub', 'Seismic Handler', 'seismology', 'seismogram',
    'seismograms', 'signal', 'slink', 'spectrogram', 'taper', 'taup',
    'travel time', 'trigger', 'VERCE', 'WAV', 'waveform', 'WaveServer',
    'WaveServerV', 'WebDC', 'Winston', 'XML-SEED', 'XSEED']
INSTALL_REQUIRES = [
    'numpy>1.0.0',
    'scipy',
    'lxml',
    'sqlalchemy',
    'suds>=0.4.0']
ENTRY_POINTS = {
    'console_scripts': [
        'obspy-flinn-engdahl = obspy.core.scripts.flinnengdahl:main',
        'obspy-runtests = obspy.core.scripts.runtests:main',
        'obspy-reftek-rescue = obspy.core.scripts.reftekrescue:main',
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
        'DATAMARK = obspy.datamark.core',
        'GSE1 = obspy.gse2.core',
        'GSE2 = obspy.gse2.core',
        'MSEED = obspy.mseed.core',
        'SAC = obspy.sac.core',
        'SACXY = obspy.sac.core',
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
        'writeFormat = obspy.seg2.seg2:writeSEG2',
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
    'obspy.plugin.event': [
        'QUAKEML = obspy.core.quakeml',
    ],
    'obspy.plugin.event.QUAKEML': [
        'isFormat = obspy.core.quakeml:isQuakeML',
        'readFormat = obspy.core.quakeml:readQuakeML',
        'writeFormat = obspy.core.quakeml:writeQuakeML',
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

# Here be dragons - beware!

UTIL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "obspy",
                                         "core", "util"))
sys.path.insert(0, UTIL_PATH)
from version import get_git_version as _getVersionString
sys.path.pop(0)

LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))
DOCSTRING = __doc__.split("\n")
IS_WINDOWS = platform.system() == "Windows"
IS_DEVELOP = 'develop' in sys.argv

NO_GFORTRAN_MSG = """

###############################################################################

ERROR: gfortran compiler not found! Retrying now without FORTRAN support.

All FORTRAN extensions in the following modules will *not* work:
  * obspy.taup

###############################################################################

"""

NO_CCOMPILER_MSG = """

###############################################################################

ERROR: C compiler not found! Retrying now without C support.

All C-extensions in following modules will *not* work:
  * obspy.gse2
  * obspy.mseed
  * obspy.segy
  * obspy.signal

###############################################################################

"""


if IS_WINDOWS:
    # ugly Monkey patch for MSVCCompiler & Mingw32CCompiler for Windows
    # using MinGW64 (http://mingw-w64.sourceforge.net/)
    from distutils.msvccompiler import MSVCCompiler
    from distutils.cygwinccompiler import Mingw32CCompiler
    MSVCCompiler._c_extensions.append(".f")

    def compile(self, sources, output_dir=None, **kwargs):  # @UnusedVariable
        # we check if 'taup' is in sources
        IS_FORTRAN = False
        for source in sources:
            if 'taup' in source:
                IS_FORTRAN = True
        if not IS_FORTRAN:
            # otherwise we just use the original compile method
            return self.original_compile(sources, output_dir=None, **kwargs)
        if output_dir:
            try:
                os.makedirs(output_dir)
            except OSError:
                pass
        if '32' in platform.architecture()[0]:
            # 32 bit gfortran compiler
            self.compiler_so = ["mingw32-gfortran.exe"]
        else:
            # 64 bit gfortran compiler
            self.compiler_so = ["x86_64-w64-mingw32-gfortran.exe"]
        objects = []
        for src in sources:
            file = os.path.splitext(src)[0]
            if output_dir:
                obj = os.path.join(output_dir, os.path.basename(file) + ".o")
            else:
                obj = file + ".o"
            try:
                self.spawn(self.compiler_so + \
                           ["-fno-underscoring", "-c"] + [src, '-o', obj])
            except DistutilsExecError:
                _, msg, _ = sys.exc_info()
                raise CompileError(msg)
            objects.append(obj)
        return objects

    def link(self, _target_desc, objects, output_filename,
             *args, **kwargs):  # @UnusedVariable
        # we check if 'taup' is in output_filename
        if 'taup' not in output_filename:
            # otherwise we just use the original link method
            return self.original_link(_target_desc, objects, output_filename,
                                      *args, **kwargs)
        try:
            os.makedirs(os.path.dirname(output_filename))
        except OSError:
            pass
        self.spawn(self.compiler_so + \
                   ["-static-libgcc", "-static-libgfortran", "-shared"] + \
                   objects + ["-o", output_filename])

    MSVCCompiler.original_compile = MSVCCompiler.compile
    MSVCCompiler.compile = compile
    MSVCCompiler.original_link = MSVCCompiler.link
    MSVCCompiler.link = link
    Mingw32CCompiler.original_compile = Mingw32CCompiler.compile
    Mingw32CCompiler.compile = compile
    Mingw32CCompiler.original_link = Mingw32CCompiler.link
    Mingw32CCompiler.link = link
else:
    # Monkey patch CCompiler for Unix, Linux and Mac
    # Pretend .f is a C extension and change corresponding compilation call
    CCompiler.language_map['.f'] = "c"
    # Monkey patch UnixCCompiler for Unix, Linux and MacOS
    UnixCCompiler.src_extensions.append(".f")

    def _compile(self, obj, src, *args, **kwargs):  # @UnusedVariable
        # we check if 'taup' is in sources
        IS_FORTRAN = False
        if 'taup' in src:
            IS_FORTRAN = True
        if not IS_FORTRAN:
            # otherwise we just use the original compile method
            UnixCCompiler.linker_so = None
            return self._original_compile(obj, src, *args, **kwargs)
        UnixCCompiler.linker_so = ["gfortran"]
        self.compiler_so = ["gfortran"]
        cc_args = ['-c', '-fno-underscoring']
        if sys.platform == 'darwin':
            self.compiler_so = _darwin_compiler_fixup(self.compiler_so,
                                                      cc_args)
        else:
            cc_args.append('-fPIC')
        try:
            self.spawn(self.compiler_so + [src, '-o', obj] + cc_args)
        except DistutilsExecError:
            _, msg, _ = sys.exc_info()
            raise CompileError(msg)
    UnixCCompiler._original_compile = UnixCCompiler._compile
    UnixCCompiler._compile = _compile


def convert2to3():
    """
    Convert source to Python 3.x syntax using lib2to3.
    """
    # create a new 2to3 directory for converted source files
    dst_path = os.path.join(LOCAL_PATH, '2to3')
    shutil.rmtree(dst_path, ignore_errors=True)

    # copy original tree into 2to3 folder ignoring some unneeded files
    def ignored_files(adir, filenames):  # @UnusedVariable
        return ['.svn', '2to3', 'debian', 'build', 'dist'] + \
               ['.git', '.gitignore'] + \
               [fn for fn in filenames if fn.startswith('distribute')] + \
               [fn for fn in filenames if fn.endswith('.egg-info')]
    shutil.copytree(LOCAL_PATH, dst_path, ignore=ignored_files)
    os.chdir(dst_path)
    sys.path.insert(0, dst_path)
    # run lib2to3 script on duplicated source
    from lib2to3.main import main
    print("Converting to Python3 via lib2to3...")
    main("lib2to3.fixes", ["-w", "-n", "--no-diffs", "obspy"])


# hack to prevent build_ext to append __init__ to the export symbols
class finallist(list):
    def append(self, object):
        return


class MyExtension(Extension):
    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)
        self.export_symbols = finallist(self.export_symbols)


def setupLibMSEED():
    """
    Prepare building of C extension libmseed.
    """
    macros = []
    extra_link_args = []
    extra_compile_args = []
    src_obspy = os.path.join('obspy', 'mseed', 'src') + os.sep
    src = os.path.join('obspy', 'mseed', 'src', 'libmseed') + os.sep
    # get symbols for libmseed
    lines = open(src + 'libmseed.def', 'r').readlines()[2:]
    symbols = [s.strip() for s in lines if s.strip() != '']
    # get symbols for obspy-readbuffer.c
    lines = open(src_obspy + 'obspy-readbuffer.def', 'r').readlines()[2:]
    symbols += [s.strip() for s in lines if s.strip() != '']

    # system specific settings
    if IS_WINDOWS:
        # needed by libmseed lmplatform.h
        macros.append(('WIN32', '1'))
        # disable some warnings for MSVC
        macros.append(('_CRT_SECURE_NO_WARNINGS', '1'))
        if 'msvc' in sys.argv or \
            ('-c' not in sys.argv and get_default_compiler() == 'msvc'):
            if platform.architecture()[0] == '32bit':
                # Workaround Win32 and MSVC - see issue #64
                extra_compile_args.append("/fp:strict")

    # create library name
    if IS_DEVELOP:
        lib_name = 'libmseed-%s-%s-py%s' % (
            platform.system(), platform.architecture()[0],
            ''.join([str(i) for i in platform.python_version_tuple()[:2]]))
    else:
        lib_name = 'libmseed'

    # setup C extension
    lib = MyExtension(lib_name,
                      define_macros=macros,
                      libraries=[],
                      sources=[src + 'fileutils.c', src + 'genutils.c',
                               src + 'gswap.c', src + 'lmplatform.c',
                               src + 'lookup.c', src + 'msrutils.c',
                               src + 'pack.c', src + 'packdata.c',
                               src + 'traceutils.c', src + 'tracelist.c',
                               src + 'unpack.c', src + 'unpackdata.c',
                               src + 'selection.c', src + 'logging.c',
                               src + 'parseutils.c',
                               src_obspy + 'obspy-readbuffer.c'],
                      export_symbols=symbols,
                      extra_link_args=extra_link_args,
                      extra_compile_args=extra_compile_args)
    return lib


def setupLibGSE2():
    """
    Prepare building of C extension libgse2.
    """
    macros = []
    src = os.path.join('obspy', 'gse2', 'src', 'GSE_UTI') + os.sep
    symbols = [s.strip()
               for s in open(src + 'gse_functions.def').readlines()[2:]
               if s.strip() != '']
    # system specific settings
    if IS_WINDOWS:
        # disable some warnings for MSVC
        macros.append(('_CRT_SECURE_NO_WARNINGS', '1'))
    # create library name
    if IS_DEVELOP:
        lib_name = 'libgse2-%s-%s-py%s' % (
            platform.system(), platform.architecture()[0],
            ''.join([str(i) for i in platform.python_version_tuple()[:2]]))
    else:
        lib_name = 'libgse2'
    # setup C extension
    lib = MyExtension(lib_name,
                      define_macros=macros,
                      libraries=[],
                      sources=[src + 'buf.c', src + 'gse_functions.c'],
                      export_symbols=symbols,
                      extra_link_args=[])
    return lib


def setupLibSignal():
    """
    Prepare building of C extension libsignal.
    """
    macros = []
    src = os.path.join('obspy', 'signal', 'src') + os.sep
    symbols = [s.strip() for s in open(src + 'libsignal.def').readlines()[2:]
               if s.strip() != '']
    # system specific settings
    if IS_WINDOWS:
        # disable some warnings for MSVC
        macros.append(('_CRT_SECURE_NO_WARNINGS', '1'))
    # create library name
    if IS_DEVELOP:
        lib_name = 'libsignal-%s-%s-py%s' % (
            platform.system(), platform.architecture()[0],
            ''.join([str(i) for i in platform.python_version_tuple()[:2]]))
    else:
        lib_name = 'libsignal'
    # setup C extension
    lib = MyExtension(lib_name,
                      define_macros=macros,
                      sources=[src + 'recstalta.c', src + 'xcorr.c',
                               src + 'coordtrans.c', src + 'pk_mbaer.c',
                               src + 'filt_util.c', src + 'arpicker.c',
                               src + 'bbfk.c', src + 'stalta.c'],
                      export_symbols=symbols)
    return lib


def setupLibEvalResp():
    """
    Prepare building of evalresp extension library.
    """
    macros = []
    src = os.path.join('obspy', 'signal', 'src') + os.sep
    src_evresp = os.path.join('obspy', 'signal', 'src', 'evalresp') + os.sep
    evresp_include_dir = src_evresp
    symbols = [s.strip() for s in open(src + 'libevresp.def').readlines()[2:]
               if s.strip() != '']
    # system specific settings
    if IS_WINDOWS:
        # needed by evalresp evresp.h
        macros.append(('WIN32', '1'))
        # disable some warnings for MSVC
        macros.append(('_CRT_SECURE_NO_WARNINGS', '1'))
    # create library name
    if IS_DEVELOP:
        lib_name = 'libevresp-%s-%s-py%s' % (
            platform.system(), platform.architecture()[0],
            ''.join([str(i) for i in platform.python_version_tuple()[:2]]))
    else:
        lib_name = 'libevresp'
    # setup C extension
    lib = MyExtension(lib_name,
                      define_macros=macros,
                      include_dirs=[evresp_include_dir],
                      sources=glob.glob(os.path.join(src_evresp, '*.c')),
                      export_symbols=symbols)
    return lib


def setupLibSEGY():
    """
    Prepare building of C extension libsegy.
    """
    macros = []
    src = os.path.join('obspy', 'segy', 'src') + os.sep
    symbols = [s.strip() for s in open(src + 'libsegy.def').readlines()[2:]
               if s.strip() != '']
    # system specific settings
    if IS_WINDOWS:
        # disable some warnings for MSVC
        macros.append(('_CRT_SECURE_NO_WARNINGS', '1'))
    # create library name
    if IS_DEVELOP:
        lib_name = 'libsegy-%s-%s-py%s' % (
            platform.system(), platform.architecture()[0],
            ''.join([str(i) for i in platform.python_version_tuple()[:2]]))
    else:
        lib_name = 'libsegy'
    # setup C extension
    lib = MyExtension(lib_name,
                      define_macros=macros,
                      include_dirs=[],
                      sources=[src + 'ibm2ieee.c'],
                      # The following two lines are needed for OpenMP which is
                      # currently not working.
                      #extra_compile_args = ['-fopenmp'],
                      #extra_link_args=['-lgomp'],
                      export_symbols=symbols)
    return lib


def setupLibTauP():
    """
    Prepare building of Fortran extensions.
    """
    # create library name
    if IS_DEVELOP:
        lib_name = 'libtaup-%s-%s-py%s' % (
            platform.system(), platform.architecture()[0],
            ''.join([str(i) for i in platform.python_version_tuple()[:2]]))
    else:
        lib_name = 'libtaup'
    # setup Fortran extension
    src = os.path.join('obspy', 'taup', 'src') + os.sep
    lib = MyExtension(lib_name,
                      libraries=['gfortran'],
                      sources=[src + 'emdlv.f', src + 'libtau.f',
                               src + 'ttimes_subrout.f'])
    return lib


def setupPackage(gfortran=True, ccompiler=True):
    # automatically install distribute if the user does not have it installed
    distribute_setup.use_setuptools()
    # use lib2to3 for Python 3.x
    if sys.version_info[0] == 3:
        convert2to3()
    # external modules
    ext_modules = []
    if ccompiler:
        ext_modules += [setupLibMSEED(), setupLibGSE2(), setupLibSignal(),
                        setupLibEvalResp(), setupLibSEGY()]
    if gfortran:
        ext_modules.append(setupLibTauP())
    kwargs = {}
    if ext_modules:
        kwargs['ext_package'] = 'obspy.lib'
        kwargs['ext_modules'] = ext_modules
    # remove clean from second call of this function so nothing gets rebuild
    if not gfortran or not ccompiler:
        argvs = sys.argv[1:]
        if 'clean' in argvs:
            # get index of clean command
            i0 = argvs.index('clean')
            # backup everything in front of clean
            temp = argvs[:i0]
            # search and remove options after clean starting with a dash
            rest = argvs[(i0 + 1):]
            for i, arg in enumerate(rest):
                if arg.startswith('-'):
                    continue
                # append everything after clean at its options to backup
                temp += rest[i:]
                break
            # set setup command line arguments
            kwargs['script_args'] = temp
    # setup package
    setup(
        name='obspy',
        version=_getVersionString(),
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
            'License :: OSI Approved :: GNU Library or ' + \
                'Lesser General Public License (LGPL)',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics'],
        keywords=KEYWORDS,
        packages=find_packages(exclude=['distribute_setup']),
        namespace_packages=[],
        zip_safe=False,
        install_requires=INSTALL_REQUIRES,
        download_url="https://github.com/obspy/obspy/zipball/master" + \
            "#egg=obspy=dev",  # this is needed for "easy_install obspy==dev"
        include_package_data=True,
        entry_points=ENTRY_POINTS,
        use_2to3=True,
        **kwargs
    )
    # cleanup after using lib2to3 for Python 3.x
    if sys.version_info[0] == 3:
        os.chdir(LOCAL_PATH)


if __name__ == '__main__':
    gfortran = True
    ccompiler = True
    # the following construct will retry building even if the C or gfortran
    # compilers are missing - it will warn the user if something went wrong
    while True:
        try:
            setupPackage(gfortran=gfortran, ccompiler=ccompiler)
        except SystemExit, e:
            if 'gfortran' in str(e):
                if not gfortran:
                    break
                # retry
                print NO_GFORTRAN_MSG
                gfortran = False
                continue
            elif 'gcc' in str(e):
                if not ccompiler:
                    break
                # retry
                print NO_CCOMPILER_MSG
                ccompiler = False
                # gcc is also needed for gfortran on non windows system
                if not IS_WINDOWS:
                    print NO_GFORTRAN_MSG
                    gfortran = False
                continue
            else:
                raise
        except ValueError, e:
            # Windows specific exception if MSVC compiler is missing
            if IS_WINDOWS and 'path' in str(e):
                if not ccompiler:
                    break
                # retry
                print NO_CCOMPILER_MSG
                ccompiler = False
                continue
            else:
                raise
        else:
            # no exception - everything seems to be fine - exit
            break
    # print any error message again for better visibility
    if not gfortran:
        print NO_GFORTRAN_MSG
    if not ccompiler:
        print NO_CCOMPILER_MSG
