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

from numpy.distutils.core import DistutilsSetupError, setup
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
from headers import (  # @UnresolvedImport
    KEYWORDS, INSTALL_REQUIRES, EXTRAS_REQUIRE, ENTRY_POINTS)
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
EXTERNAL_LIBS = False


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
    if 'setuptools' not in sys.modules:
        return {}

    class ExternalLibFeature(setuptools.Feature):
        def include_in(self, dist):
            global EXTERNAL_LIBS
            EXTERNAL_LIBS = True

        def exclude_from(self, dist):
            global EXTERNAL_LIBS
            EXTERNAL_LIBS = False

    return {
        'system-libs': ExternalLibFeature(
            'use of system C libraries',
            standard=False,
            EXTERNAL_LIBS=True
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
    if not EXTERNAL_LIBS:
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
    if EXTERNAL_LIBS:
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
    if EXTERNAL_LIBS:
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
    if EXTERNAL_LIBS:
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
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
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
