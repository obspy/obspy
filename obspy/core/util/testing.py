# -*- coding: utf-8 -*-
"""
Testing utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import PY2, native_str

import difflib
import doctest
import fnmatch
import glob
import inspect
import io
import os
import re
import shutil
import unittest
import warnings

from lxml import etree
import numpy as np

from obspy.core.util.base import NamedTemporaryFile, get_matplotlib_version
from obspy.core.util.misc import CatchOutput, get_untracked_files_from_git, \
    MatplotlibBackend


MATPLOTLIB_VERSION = get_matplotlib_version()


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
    MODULE_TESTS = __import__(MODULE_NAME + ".tests",
                              fromlist=[native_str("obspy")])
    filename_pattern = os.path.join(MODULE_TESTS.__path__[0], "test_*.py")
    files = glob.glob(filename_pattern)
    names = (os.path.basename(file).split(".")[0] for file in files)
    module_names = (".".join([MODULE_NAME, "tests", name]) for name in names)
    for module_name in module_names:
        module = __import__(module_name,
                            fromlist=[native_str("obspy")])
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
    MODULE = __import__(MODULE_NAME, fromlist=[native_str("obspy")])
    MODULE_PATH = MODULE.__path__[0]
    MODULE_PATH_LEN = len(MODULE_PATH)

    for root, _dirs, files in os.walk(MODULE_PATH):
        # skip directories without __init__.py
        if '__init__.py' not in files:
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
                module = __import__(module_name,
                                    fromlist=[native_str("obspy")])
                testsuite.addTest(doctest.DocTestSuite(module))
            except ValueError:
                pass


def write_png(arr, filename):
    """
    Custom write_png() function. matplotlib < 1.3 cannot write RGBA png files.

    Modified from http://stackoverflow.com/a/19174800/1657047
    """
    import zlib
    import struct

    buf = arr[::-1, :, :].tostring()
    height, width, _ = arr.shape

    # reverse the vertical line order and add null bytes at the start
    width_byte_4 = width * 4
    raw_data = b''.join(
        b'\x00' + buf[span:span + width_byte_4]
        for span in range((height - 1) * width * 4, -1, - width_byte_4))

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack(native_str("!I"), len(data)) +
                chunk_head +
                struct.pack(native_str("!I"),
                            0xFFFFFFFF & zlib.crc32(chunk_head)))

    with open(filename, "wb") as fh:
        fh.write(b''.join([
            b'\x89PNG\r\n\x1a\n',
            png_pack(b'IHDR', struct.pack(native_str("!2I5B"),
                     width, height, 8, 6, 0, 0, 0)),
            png_pack(b'IDAT', zlib.compress(raw_data, 9)),
            png_pack(b'IEND', b'')]))


def compare_images(expected, actual, tol):
    """
    Custom version of :func:`matplotlib.testing.compare.compare_images`.
    This enable ObsPy to have the same image comparison metric across
    matplotlib versions. Furthermore nose is no longer a test dependency of
    ObsPy.

    In contrast to the matplotlib version this one only works with png
    files. Fully transparent pixels will have their color set to white as
    the RGB values of fully transparent pixels change depending on the
    matplotlib version.

    Additionally this version uses a straight RMSE definition instead of the
    binned one of matplotlib.

    :param expected: The filename of the expected png file.
    :type expected: str
    :param actual: The filename of the actual png file.
    :type actual: str
    :param tol: The tolerance (a color value difference, where 255 is the
        maximal difference). The test fails if the average pixel difference
        is greater than this value.
    :type tol: float
    """
    import matplotlib.image

    if not os.path.exists(actual):
        msg = "Output image %s does not exist." % actual
        raise Exception(msg)

    if os.stat(actual).st_size == 0:
        msg = "Output image file %s is empty." % actual
        raise Exception(msg)

    if not os.path.exists(expected):
        raise IOError('Baseline image %r does not exist.' % expected)

    # Open the images. Will be opened as RGBA as float32 ranging from 0 to 1.
    expected_image = matplotlib.image.imread(native_str(expected))
    actual_image = matplotlib.image.imread(native_str(actual))
    if expected_image.shape != actual_image.shape:
        raise ImageComparisonException(
            "The shape of the received image %s is not equal to the expected "
            "shape %s." % (str(actual_image.shape),
                           str(expected_image.shape)))

    # Set the "color" of fully transparent pixels to white. This avoids the
    # issue of different "colors" for transparent pixels.
    expected_image[expected_image[..., 3] <= 0.0035] = [1.0, 1.0, 1.0, 0.0]
    actual_image[actual_image[..., 3] <= 0.0035] = [1.0, 1.0, 1.0, 0.0]

    # This deviates a bit from the matplotlib version and just calculates
    # the root mean square error of all pixel values without any other fancy
    # considerations. It also uses the alpha channel of the images. Scaled
    # by 255.
    rms = np.sqrt(np.sum((255.0 * (expected_image - actual_image)) ** 2) /
                  float(expected_image.size))

    base, ext = os.path.splitext(actual)
    diff_image = '%s-%s%s' % (base, 'failed-diff', ext)

    if rms <= tol:
        if os.path.exists(diff_image):
            os.unlink(diff_image)
        return None

    # Save diff image, expand differences in luminance domain
    absDiffImage = np.abs(expected_image - actual_image)
    absDiffImage *= 10.0
    save_image_np = np.clip(absDiffImage, 0.0, 1.0)
    # Hard-code the alpha channel to fully solid
    save_image_np[:, :, 3] = 1.0

    write_png(np.uint8(save_image_np * 255.0), diff_image)

    return dict(rms=rms, expected=str(expected), actual=str(actual),
                diff=str(diff_image), tol=tol)


class ImageComparisonException(unittest.TestCase.failureException):
    pass


class ImageComparison(NamedTemporaryFile):
    """
    Handles the comparison against a baseline image in an image test.

    .. note::
        Baseline images are created using matplotlib version `1.3.1`.

    :type image_path: str
    :param image_path: Path to directory where the baseline image is located
    :type image_name: str
    :param image_name: Filename (with suffix, without directory path) of the
        baseline image
    :type reltol: float, optional
    :param reltol: Multiplier that is applied to the default tolerance
        value (i.e. 10 means a 10 times harder to pass test tolerance).
    :type adjust_tolerance: bool, optional
    :param adjust_tolerance: Adjust the tolerance based on the matplotlib
        version. Can optionally be turned off to simply compare two images.

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
        To only keep failed images and the corresponding diff image,
        additionally set environment variable `OBSPY_KEEP_ONLY_FAILED_IMAGES`
        to any value before executing the test.

    .. rubric:: Example

    >>> from obspy import read
    >>> with ImageComparison("/my/baseline/folder", 'plot.png') as ic:
    ...     st = read()  # doctest: +SKIP
    ...     st.plot(outfile=ic.name)  # doctest: +SKIP
    ...     # image is compared against baseline image automatically
    """
    def __init__(self, image_path, image_name, reltol=1,
                 adjust_tolerance=True, *args, **kwargs):
        self.suffix = "." + image_name.split(".")[-1]
        super(ImageComparison, self).__init__(suffix=self.suffix, *args,
                                              **kwargs)
        self.image_name = image_name
        self.baseline_image = os.path.join(image_path, image_name)
        self.keep_output = "OBSPY_KEEP_IMAGES" in os.environ
        self.keep_only_failed = "OBSPY_KEEP_ONLY_FAILED_IMAGES" in os.environ
        self.output_path = os.path.join(image_path, "testrun")
        self.diff_filename = "-failed-diff.".join(self.name.rsplit(".", 1))
        self.tol = reltol * 3.0

        # Higher tolerance for older matplotlib versions. This is pretty
        # high but the pictures are at least guaranteed to be generated and
        # look (roughly!) similar. Otherwise testing is just a pain and
        # frankly not worth the effort!
        if adjust_tolerance:
            if MATPLOTLIB_VERSION < [1, 3, 0]:
                self.tol *= 30

    def __enter__(self):
        """
        Set matplotlib defaults.
        """
        MatplotlibBackend.switch_backend("AGG", sloppy=False)

        from matplotlib import font_manager, rcParams, rcdefaults
        import locale

        try:
            locale.setlocale(locale.LC_ALL, native_str('en_US.UTF-8'))
        except:
            try:
                locale.setlocale(locale.LC_ALL,
                                 native_str('English_United States.1252'))
            except:
                msg = "Could not set locale to English/United States. " + \
                      "Some date-related tests may fail"
                warnings.warn(msg)

        # set matplotlib builtin default settings for testing
        rcdefaults()
        rcParams['font.family'] = 'Bitstream Vera Sans'
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', 'findfont:.*')
            font_manager.findfont('Bitstream Vera Sans')
        if w:
            warnings.warn('Unable to find the Bitstream Vera Sans font. '
                          'Plotting tests will likely fail.')
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
        msg = ""
        try:
            # only compare images if no exception occurred in the with
            # statement. this avoids masking previously occurred exceptions (as
            # an exception may occur in compare()). otherwise we only clean up
            # and the exception gets re-raised at the end of __exit__.
            if exc_type is None:
                msg = self.compare()
        # we can still upload images if comparison fails on two different sized
        # images
        except ValueError as e:
            failed = True
            if "operands could not be broadcast together" in msg:
                msg = str(e) + "\n"
                upload_links = self._upload_images()
                msg += ("\tExpected:  {expected}\n"
                        "\tActual:    {actual}\n"
                        "\tDiff:      {diff}\n").format(**upload_links)
                raise ImageComparisonException(msg)
            raise
        # simply reraise on any other unhandled exceptions
        except:
            failed = True
            raise
        # if image comparison not raises by itself, the test failed if we get a
        # message back or the test passed if we get an empty message
        else:
            if msg:
                failed = True
            else:
                failed = False
            if failed:
                # base message on deviation of baseline and actual image
                msg = ("Image comparision failed.\n"
                       "\tRMS:       {rms}\n"
                       "\tTolerance: {tol}\n").format(**msg)
                # optionally, copy failed images from /tmp and append
                # the local paths
                if self.keep_output:
                    ff = self._get_final_filenames()
                    msg += ("\tExpected:  {expected}\n"
                            "\tActual:    {actual}\n"
                            "\tDiff:      {diff}\n").format(**ff)
                # try to upload to imgur, if successful append links to message
                upload_result = self._upload_images()
                if isinstance(upload_result, dict):
                    msg += ("\tExpected:  {expected}\n"
                            "\tActual:    {actual}\n"
                            "\tDiff:      {diff}\n").format(**upload_result)
                else:
                    msg += upload_result
                msg = msg.rstrip()
                raise ImageComparisonException(msg)
        # finally clean up after the image test, whether failed or not.
        # if specified move generated output to source tree
        finally:
            import matplotlib.pyplot as plt
            self.close()  # flush internal buffer
            self._fileobj.close()
            plt.close()
            if self.keep_output:
                if failed or not self.keep_only_failed:
                    self._copy_tempfiles()
            # delete temporary files
            os.remove(self.name)
            if os.path.exists(self.diff_filename):
                os.remove(self.diff_filename)

    def compare(self):
        """
        Run a custom version
        of :func:`matplotlib.testing.compare.compare_images` and raise an
        unittest.TestCase.failureException with the message string given by
        matplotlib if the comparison exceeds the allowed tolerance.
        """
        if os.stat(self.name).st_size == 0:
            msg = "Empty output image file."
            raise ImageComparisonException(msg)

        return compare_images(self.baseline_image, self.name, tol=self.tol)

    def _get_final_filenames(self):
        """
        Helper function returning the
        :return:
        """
        directory = self.output_path
        filename_new = os.path.join(directory, self.image_name)
        diff_filename_new = "-failed-diff.".join(
            self.image_name.rsplit(".", 1))
        diff_filename_new = os.path.join(directory, diff_filename_new)
        return {"actual": os.path.abspath(filename_new),
                "expected": os.path.abspath(self.baseline_image),
                "diff": os.path.abspath(diff_filename_new)}

    def _copy_tempfiles(self):
        """
        Copies created images from tempfiles to a subfolder of baseline images.
        """
        try:
            filenames = self._get_final_filenames()
            directory = self.output_path
            if os.path.exists(directory):
                if not os.path.isdir(directory):
                    msg = ("Could not keep output image, target directory "
                           "exists but is no directory: %s") % directory
                    warnings.warn(msg)
                    return
            else:
                os.mkdir(directory)
            if os.path.isfile(self.diff_filename):
                shutil.copy(self.diff_filename, filenames["diff"])
            shutil.copy(self.name, filenames["actual"])
        except Exception as e:
            msg = ("Failed to copy images from temporary directory "
                   "(caught %s: %s).") % (e.__class__.__name__, str(e))
            warnings.warn(msg)

    def _upload_images(self):
        """
        Uploads images to imgur.

        :returns: ``dict`` with links to uploaded images or ``str`` with
            message if upload failed
        """
        try:
            import pyimgur
            import requests
        except Exception as e:
            msg = ("Upload to imgur failed (caught %s: %s).")
            return msg % (e.__class__.__name__, str(e))
        # try to get imgur client id from environment
        imgur_clientid = \
            os.environ.get("OBSPY_IMGUR_CLIENTID") or "53b182544dc5d89"
        # upload images and return urls
        links = {}
        try:
            imgur = pyimgur.Imgur(imgur_clientid)
            if os.path.exists(self.baseline_image):
                up = imgur.upload_image(self.baseline_image, title=self.name)
                links['expected'] = up.link
            if os.path.exists(self.name):
                up = imgur.upload_image(self.name, title=self.name)
                links['actual'] = up.link
            if os.path.exists(self.diff_filename):
                up = imgur.upload_image(self.diff_filename,
                                        title=self.diff_filename)
                links['diff'] = up.link
        except (requests.exceptions.SSLError, Exception) as e:
            msg = ("Upload to imgur failed (caught %s: %s).")
            return msg % (e.__class__.__name__, str(e))
        return links


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

# List of flake8 error codes to ignore. Keep it as small as possible - there
# usually is little reason to fight flake8.
FLAKE8_IGNORE_CODES = [
    # E402 module level import not at top of file
    # This is really annoying when using the standard library import hooks
    # from the future package.
    "E402"
]


def check_flake8():
    if not HAS_FLAKE8:
        raise Exception('flake8 is required to check code formatting')

    # pyflakes autodetection of PY2 does not work with the future library.
    # Therefore, overwrite the pyflakes autodetection manually
    if PY2:
        import pyflakes.checker  # @UnusedImport
        pyflakes.checker.PY2 = True
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

            # exclude *.py files in obspy/lib
            try:
                tmp_dir, _ = os.path.split(py_file)
                _, tmp_dir = os.path.split(tmp_dir)
                if tmp_dir == "lib":
                    continue
            except:
                pass
            # Check files that do not match any exclusion pattern
            for exclude_pattern in FLAKE8_EXCLUDE_FILES:
                if fnmatch.fnmatch(py_file, exclude_pattern):
                    break
            else:
                files.append(py_file)
    flake8_style = get_style_guide(parse_argv=False,
                                   config_file=flake8.main.DEFAULT_CONFIG)
    flake8_style.options.ignore = tuple(set(
        flake8_style.options.ignore).union(set(FLAKE8_IGNORE_CODES)))

    with CatchOutput() as out:
        files = [native_str(f) for f in files]
        report = flake8_style.check_files(files)

    return report, out.stdout


# this dictionary contains the locations of checker routines that determine
# whether the module's tests can be executed or not (e.g. because test server
# is unreachable, necessary ports are blocked, etc.).
# A checker routine should return either an empty string (tests can and will
# be executed) or a message explaining why tests can not be executed (all
# tests of corresponding module will be skipped).
MODULE_TEST_SKIP_CHECKS = {
    'seishub':
        'obspy.clients.seishub.tests.test_client._check_server_availability',
    }


def compare_xml_strings(doc1, doc2):
    """
    Simple helper function to compare two XML strings.

    :type doc1: str
    :type doc2: str
    """
    # Compat py2k and py3k
    try:
        doc1 = doc1.encode()
        doc2 = doc2.encode()
    except:
        pass
    obj1 = etree.fromstring(doc1).getroottree()
    obj2 = etree.fromstring(doc2).getroottree()

    buf = io.BytesIO()
    obj1.write_c14n(buf)
    buf.seek(0, 0)
    str1 = buf.read().decode()
    str1 = [_i.strip() for _i in str1.splitlines()]

    buf = io.BytesIO()
    obj2.write_c14n(buf)
    buf.seek(0, 0)
    str2 = buf.read().decode()
    str2 = [_i.strip() for _i in str2.splitlines()]

    unified_diff = difflib.unified_diff(str1, str2)

    err_msg = "\n".join(unified_diff)
    if err_msg:  # pragma: no cover
        raise AssertionError("Strings are not equal.\n" + err_msg)


def remove_unique_IDs(xml_string, remove_creation_time=False):
    """
    Removes unique ID parts of e.g. 'publicID="..."' attributes from xml
    strings.

    :type xml_string: str
    :param xml_string: xml string to process
    :type remove_creation_time: bool
    :param xml_string: controls whether to remove 'creationTime' tags or not.
    :rtype: str
    """
    prefixes = ["id", "publicID", "pickID", "originID", "preferredOriginID",
                "preferredMagnitudeID", "preferredFocalMechanismID",
                "referenceSystemID", "methodID", "earthModelID",
                "triggeringOriginID", "derivedOriginID", "momentMagnitudeID",
                "greensFunctionID", "filterID", "amplitudeID",
                "stationMagnitudeID", "earthModelID", "slownessMethodID",
                "pickReference", "amplitudeReference"]
    if remove_creation_time:
        prefixes.append("creationTime")
    for prefix in prefixes:
        xml_string = re.sub("%s='.*?'" % prefix, '%s=""' % prefix, xml_string)
        xml_string = re.sub('%s=".*?"' % prefix, '%s=""' % prefix, xml_string)
        xml_string = re.sub("<%s>.*?</%s>" % (prefix, prefix),
                            '<%s/>' % prefix, xml_string)
    return xml_string


def get_all_py_files():
    """
    Return a list with full absolute paths to all .py files in ObsPy file tree.

    :rtype: list of str
    """
    util_dir = os.path.abspath(inspect.getfile(inspect.currentframe()))
    obspy_dir = os.path.dirname(os.path.dirname(os.path.dirname(util_dir)))
    py_files = set()
    # Walk the obspy directory structure
    for dirpath, _, filenames in os.walk(obspy_dir):
        py_files.update([os.path.abspath(os.path.join(dirpath, i)) for i in
                         filenames if i.endswith(".py")])
    return sorted(py_files)


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
