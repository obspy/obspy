#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Travel time calculation tool.

The obspy.taup package contains Python wrappers for iaspei-tau - a travel time
library by Arthur Snoke (http://www.iris.edu/software/downloads/processing/).
The library iaspei-tau is written in Fortran and interfaced via Python ctypes.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see Beyreuther et al. 2010, Megies et al. 2011).
The goal of the ObsPy project is to facilitate rapid application development
for seismology.

For more information visit http://www.obspy.org.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    Unknown
"""
from distutils.ccompiler import CCompiler
from distutils.errors import DistutilsExecError, CompileError
from distutils.unixccompiler import UnixCCompiler, _darwin_compiler_fixup
from setuptools import find_packages, setup
from setuptools.extension import Extension
import os
import platform
import shutil
import sys


LOCAL_PATH = os.path.abspath(os.path.dirname(__file__))
DOCSTRING = __doc__.split("\n")

# package specific
VERSION = open(os.path.join(LOCAL_PATH, 'obspy', 'taup', 'VERSION.txt')).read()
NAME = 'obspy.taup'
AUTHOR = 'The ObsPy Development Team'
AUTHOR_EMAIL = 'devs@obspy.org'
LICENSE = 'GNU Lesser General Public License, Version 3 (LGPLv3)'
KEYWORDS = ['ObsPy', 'seismology', 'taup']
INSTALL_REQUIRES = ['obspy.core']
ENTRY_POINTS = {}


if platform.system() != "Windows":
    # Monkey patch CCompiler for Unix, Linux and Windows
    # Pretend that .f is a C extension and change corresponding compilation call
    CCompiler.language_map['.f'] = "c"
    # Monkey patch UnixCCompiler for Unix, Linux and MacOS
    UnixCCompiler.src_extensions.append(".f")
    def _compile(self, obj, src, *args, **kwargs): #@UnusedVariable
            self.compiler_so = ["gfortran"]
            cc_args = ['-c', '-fno-underscoring']
            if sys.platform == 'darwin':
                self.compiler_so = _darwin_compiler_fixup(self.compiler_so,
                                                          cc_args)
            else:
                cc_args.append('-fPIC')
            try:
                self.spawn(self.compiler_so + [src, '-o', obj] + cc_args)
            except DistutilsExecError, msg:
                raise CompileError, msg
    UnixCCompiler._compile = _compile
else:
    # Monkey patch MSVCCompiler & Mingw32CCompiler for Windows
    # using MinGW64 (http://mingw-w64.sourceforge.net/)
    from distutils.msvccompiler import MSVCCompiler
    from distutils.cygwinccompiler import Mingw32CCompiler
    MSVCCompiler._c_extensions.append(".f")

    def compile(self, sources, output_dir=None, **kwargs): #@UnusedVariable
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
                self.spawn(self.compiler_so + ["-fno-underscoring", "-c"] + \
                           [src, '-o', obj])
            except DistutilsExecError, msg:
                raise CompileError, msg
            objects.append(obj)
        return objects

    def link(self, _target_desc, objects, output_filename,
             *args, **kwargs): #@UnusedVariable
        try:
            os.makedirs(os.path.dirname(output_filename))
        except OSError:
            pass
        self.spawn(self.compiler_so + \
                   ["-static-libgcc", "-static-libgfortran", "-shared"] + \
                   objects + ["-o", output_filename])

    MSVCCompiler.compile = compile
    MSVCCompiler.link = link
    Mingw32CCompiler.compile = compile
    Mingw32CCompiler.link = link


def setupLibTauP():
    """
    Prepare building of Fortran extensions.
    """
    # hack to prevent build_ext to append __init__ to the export symbols
    class finallist(list):
        def append(self, object):
            return

    class MyExtension(Extension):
        def __init__(self, *args, **kwargs):
            Extension.__init__(self, *args, **kwargs)
            self.export_symbols = finallist(self.export_symbols)

    # create library name
    if 'develop' in sys.argv:
        lib_name = 'libtaup-%s-%s-py%s' % (
            platform.system(), platform.architecture()[0],
            ''.join([str(i) for i in platform.python_version_tuple()[:2]]))
    else:
        lib_name = 'libtaup'
    # setup Fortran extension
    src = os.path.join('obspy', 'taup', 'src') + os.sep
    lib = MyExtension(lib_name,
                      libraries=['gfortran'],
                      sources=[src + 'emdlv.f' , src + 'libtau.f',
                               src + 'ttimes_subrout.f'])
    return lib


def convert2to3():
    """
    Convert source to Python 3.x syntax using lib2to3.
    """
    # create a new 2to3 directory for converted source files
    dst_path = os.path.join(LOCAL_PATH, '2to3')
    shutil.rmtree(dst_path, ignore_errors=True)
    # copy original tree into 2to3 folder ignoring some unneeded files
    def ignored_files(adir, filenames): #@UnusedVariable
        return ['.svn', '2to3', 'debian', 'build', 'dist'] + \
               [fn for fn in filenames if fn.startswith('distribute')] + \
               [fn for fn in filenames if fn.endswith('.egg-info')]
    shutil.copytree(LOCAL_PATH, dst_path, ignore=ignored_files)
    os.chdir(dst_path)
    sys.path.insert(0, dst_path)
    # run lib2to3 script on duplicated source
    from lib2to3.main import main
    print("Converting to Python3 via lib2to3...")
    main("lib2to3.fixes", ["-w", "-n", "--no-diffs", "obspy"])


def getVersion():
    # fetch version
    file = os.path.join(LOCAL_PATH, 'obspy', NAME.split('.')[1], 'VERSION.txt')
    return open(file).read()


def setupPackage():
    # use lib2to3 for Python 3.x
    if sys.version_info[0] == 3:
        convert2to3()
    # setup package
    setup(
        name=NAME,
        version=getVersion(),
        description=DOCSTRING[1],
        long_description="\n".join(DOCSTRING[3:]),
        url="http://www.obspy.org",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        license=LICENSE,
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
        namespace_packages=['obspy'],
        zip_safe=False,
        install_requires=INSTALL_REQUIRES,
        download_url="https://svn.obspy.org/trunk/%s#egg=%s-dev" % (NAME, NAME),
        include_package_data=True,
        test_suite="%s.tests.suite" % (NAME),
        entry_points=ENTRY_POINTS,
        ext_package='obspy.taup.lib',
        ext_modules=[setupLibTauP()],
        use_2to3=True,
    )
    # cleanup after using lib2to3 for Python 3.x
    if sys.version_info[0] == 3:
        os.chdir(LOCAL_PATH)


if __name__ == '__main__':
    setupPackage()
