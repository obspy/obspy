# -*- coding: utf-8 -*-
"""
Testing utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy.core.util.misc import get_untracked_files_from_git
from obspy.core.util.base import getMatplotlibVersion, NamedTemporaryFile
import fnmatch
import inspect
import sys
import os
import glob
import unittest
import doctest
import StringIO
import shutil
import warnings


def add_unittests(testsuite, module_name):
    """
    Function to add all available unittests of the module with given name
    (e.g. "obspy.core") to the given unittest TestSuite.
    All submodules in the "tests" directory whose names are starting with
    ``test_`` are added.

    :type testsuite: unittest.TestSuite
    :param testsuite: testsuite to which the tests should be added
    :type module_name: str
    :param module_name: name of the module of which the tests should be added

    .. rubric:: Example

    >>> import unittest
    >>> suite = unittest.TestSuite()
    >>> add_unittests(suite, "obspy.core")
    """
    MODULE_NAME = module_name
    MODULE_TESTS = __import__(MODULE_NAME + ".tests", fromlist="obspy")

    filename_pattern = os.path.join(MODULE_TESTS.__path__[0], "test_*.py")
    files = glob.glob(filename_pattern)
    names = (os.path.basename(file).split(".")[0] for file in files)
    module_names = (".".join([MODULE_NAME, "tests", name]) for name in names)
    for module_name in module_names:
        module = __import__(module_name, fromlist="obspy")
        testsuite.addTest(module.suite())


def add_doctests(testsuite, module_name):
    """
    Function to add all available doctests of the module with given name
    (e.g. "obspy.core") to the given unittest TestSuite.
    All submodules in the module's root directory are added.
    Occurring errors are shown as warnings.

    :type testsuite: unittest.TestSuite
    :param testsuite: testsuite to which the tests should be added
    :type module_name: str
    :param module_name: name of the module of which the tests should be added

    .. rubric:: Example

    >>> import unittest
    >>> suite = unittest.TestSuite()
    >>> add_doctests(suite, "obspy.core")
    """
    MODULE_NAME = module_name
    MODULE = __import__(MODULE_NAME, fromlist="obspy")
    MODULE_PATH = MODULE.__path__[0]
    MODULE_PATH_LEN = len(MODULE_PATH)

    for root, _dirs, files in os.walk(MODULE_PATH):
        # skip directories without __init__.py
        if not '__init__.py' in files:
            continue
        # skip tests directories
        if root.endswith('tests'):
            continue
        # skip scripts directories
        if root.endswith('scripts'):
            continue
        # skip lib directories
        if root.endswith('lib'):
            continue
        # loop over all files
        for file in files:
            # skip if not python source file
            if not file.endswith('.py'):
                continue
            # get module name
            parts = root[MODULE_PATH_LEN:].split(os.sep)[1:]
            module_name = ".".join([MODULE_NAME] + parts + [file[:-3]])
            try:
                module = __import__(module_name, fromlist="obspy")
                testsuite.addTest(doctest.DocTestSuite(module))
            except ValueError:
                pass


def checkForMatplotlibCompareImages():
    try:
        # trying to stay inside 80 char line
        import matplotlib.testing.compare as _compare
        compare_images = _compare.compare_images  # NOQA @UnusedVariable
    except:
        return False
    # matplotlib's (< 1.2) compare_images() uses PIL internally
    if getMatplotlibVersion() < [1, 2, 0]:
        try:
            import PIL  # NOQA @UnusedImport
        except ImportError:
            return False
    return True


HAS_COMPARE_IMAGE = checkForMatplotlibCompareImages()


class ImageComparisonException(unittest.TestCase.failureException):
    pass


class ImageComparison(NamedTemporaryFile):
    """
    Handles the comparison against a baseline image in an image test.

    :type image_path: str
    :param image_path: Path to directory where the baseline image is located
    :type image_name: str
    :param image_name: Filename (with suffix, without directory path) of the
        baseline image
    :type reltol: float (optional)
    :param reltol: Multiplier that is applied to the default tolerance
        value (i.e. 10 means a 10 times harder to pass test tolerance).

    The class should be used with Python's "with" statement. When setting up,
    the matplotlib rcdefaults are set to ensure consistent image testing.
    After the plotting is completed, the :meth:`ImageComparison.compare`
    method is called automatically at the end of the "with" block, comparing
    against the previously specified baseline image. This raises an exception
    (if the test fails) with the message string from
    :func:`matplotlib.testing.compare.compare_images`. Afterwards all
    temporary files are deleted automatically.

    .. note::
        If images created during the testrun should be kept after the test, set
        environment variable `OBSPY_KEEP_IMAGES` to any value before executing
        the test (e.g. with `$ OBSPY_KEEP_IMAGES= obspy-runtests` or `$
        OBSPY_KEEP_IMAGES= python test_sometest.py`). For `obspy-runtests` the
        option "--keep-images" can also be used instead of setting an
        environment variable. Created images and diffs for failing tests are
        then stored in a subfolder "testrun" under the baseline image's
        directory.

    .. rubric:: Example

    >>> from obspy import read
    >>> with ImageComparison("/my/baseline/folder", 'plot.png') as ic:
    ...     st = read()  # doctest: +SKIP
    ...     st.plot(outfile=ic.name)  # doctest: +SKIP
    ...     # image is compared against baseline image automatically
    """
    def __init__(self, image_path, image_name, reltol=1, *args, **kwargs):
        self.suffix = "." + image_name.split(".")[-1]
        super(ImageComparison, self).__init__(suffix=self.suffix, *args,
                                              **kwargs)
        self.image_name = image_name
        self.baseline_image = os.path.join(image_path, image_name)
        self.keep_output = "OBSPY_KEEP_IMAGES" in os.environ
        self.output_path = os.path.join(image_path, "testrun")
        self.diff_filename = "-failed-diff.".join(self.name.rsplit(".", 1))
        self.tol = get_matplotlib_defaul_tolerance() * reltol

    def __enter__(self):
        """
        Set matplotlib defaults.
        """
        from matplotlib import get_backend, rcParams, rcdefaults
        import locale

        try:
            locale.setlocale(locale.LC_ALL, str('en_US.UTF-8'))
        except:
            try:
                locale.setlocale(locale.LC_ALL,
                                 str('English_United States.1252'))
            except:
                msg = "Could not set locale to English/United States. " + \
                      "Some date-related tests may fail"
                warnings.warn(msg)

        if get_backend().upper() != 'AGG':
            import matplotlib
            try:
                matplotlib.use('AGG', warn=False)
            except TypeError:
                msg = "Image comparison requires matplotlib backend 'AGG'"
                warnings.warn(msg)

        # set matplotlib builtin default settings for testing
        rcdefaults()
        rcParams['font.family'] = 'Bitstream Vera Sans'
        try:
            rcParams['text.hinting'] = False
        except KeyError:
            warnings.warn("could not set rcParams['text.hinting']")
        try:
            rcParams['text.hinting_factor'] = 8
        except KeyError:
            warnings.warn("could not set rcParams['text.hinting_factor']")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # @UnusedVariable
        """
        Remove tempfiles and store created images if OBSPY_KEEP_IMAGES
        environment variable is set.
        """
        try:
            # only compare images if no exception occured in the with
            # statement. this avoids masking previously occured exceptions (as
            # an exception may occur in compare()). otherwise we only clean up
            # and the exception gets re-raised at the end of __exit__.
            if exc_type is None:
                self.compare()
        finally:
            import matplotlib.pyplot as plt
            self.close()
            plt.close()
            if self.keep_output:
                self._copy_tempfiles()
            os.remove(self.name)
            if os.path.exists(self.diff_filename):
                os.remove(self.diff_filename)

    def compare(self, reltol=1):  # @UnusedVariable
        """
        Run :func:`matplotlib.testing.compare.compare_images` and raise an
        unittest.TestCase.failureException with the message string given by
        matplotlib if the comparison exceeds the allowed tolerance.
        """
        from matplotlib.testing.compare import compare_images
        if os.stat(self.name).st_size == 0:
            msg = "Empty output image file."
            raise ImageComparisonException(msg)
        msg = compare_images(self.baseline_image, self.name, tol=self.tol)
        if msg:
            raise ImageComparisonException(msg)

    def _copy_tempfiles(self):
        """
        Copies created images from tempfiles to a subfolder of baseline images.
        """
        directory = self.output_path
        if os.path.exists(directory) and not os.path.isdir(directory):
            msg = "Could not keep output image, target directory exists:" + \
                  directory
            warnings.warn(msg)
            return
        if not os.path.exists(directory):
            os.mkdir(directory)
        if os.path.isfile(self.diff_filename):
            diff_filename_new = \
                "-failed-diff.".join(self.image_name.rsplit(".", 1))
            shutil.copy(self.diff_filename, os.path.join(directory,
                                                         diff_filename_new))
        shutil.copy(self.name, os.path.join(directory, self.image_name))


def get_matplotlib_defaul_tolerance():
    """
    The two test images ("ok", "fail") result in the following rms values:
    matplotlib v1.3.x (git rev. 26b18e2): 0.8 and 9.0
    matplotlib v1.2.1: 1.7e-3 and 3.6e-3
    """
    if getMatplotlibVersion() < [1, 3, 0]:
        return 2e-3
    else:
        return 1


FLAKE8_EXCLUDE_FILES = [
    "*/__init__.py",
    ]

try:
    import flake8
except ImportError:
    HAS_FLAKE8 = False
else:
    # Only accept flake8 version >= 2.0
    HAS_FLAKE8 = flake8.__version__ >= '2'


def check_flake8():
    if not HAS_FLAKE8:
        raise Exception('flake8 is required to check code formatting')
    import flake8.main
    from flake8.engine import get_style_guide

    test_dir = os.path.abspath(inspect.getfile(inspect.currentframe()))
    obspy_dir = os.path.dirname(os.path.dirname(os.path.dirname(test_dir)))
    untracked_files = get_untracked_files_from_git() or []
    files = []
    for dirpath, _, filenames in os.walk(obspy_dir):
        filenames = [_i for _i in filenames if
                     os.path.splitext(_i)[-1] == os.path.extsep + "py"]
        if not filenames:
            continue
        for py_file in filenames:
            py_file = os.path.join(dirpath, py_file)
            # ignore untracked files
            if os.path.abspath(py_file) in untracked_files:
                continue

            # Check files that do not match any exclusion pattern
            for exclude_pattern in FLAKE8_EXCLUDE_FILES:
                if fnmatch.fnmatch(py_file, exclude_pattern):
                    break
            else:
                files.append(py_file)
    flake8_style = get_style_guide(parse_argv=False,
                                   config_file=flake8.main.DEFAULT_CONFIG)
    sys.stdout = StringIO.StringIO()
    report = flake8_style.check_files(files)
    sys.stdout.seek(0)
    message = sys.stdout.read()
    sys.stdout = sys.__stdout__
    return report, message


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
