# -*- coding: utf-8 -*-
"""
Testing utilities for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

import difflib
import doctest
import glob
import inspect
import io
import os
import re
import shutil
import unittest
import warnings
from distutils.version import LooseVersion

import numpy as np
from lxml import etree

from obspy.core.util.base import NamedTemporaryFile, MATPLOTLIB_VERSION
from obspy.core.util.misc import MatplotlibBackend

# this dictionary contains the locations of checker routines that determine
# whether the module's tests can be executed or not (e.g. because test server
# is unreachable, necessary ports are blocked, etc.).
# A checker routine should return either an empty string (tests can and will
# be executed) or a message explaining why tests can not be executed (all
# tests of corresponding module will be skipped).
MODULE_TEST_SKIP_CHECKS = {
    'clients.seishub':
        'obspy.clients.seishub.tests.test_client._check_server_availability'}


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
    module_tests = __import__(module_name + ".tests",
                              fromlist=[native_str("obspy")])
    filename_pattern = os.path.join(module_tests.__path__[0], "test_*.py")
    files = glob.glob(filename_pattern)
    names = (os.path.basename(file).split(".")[0] for file in files)
    module_names = (".".join([module_name, "tests", name]) for name in names)
    for _module_name in module_names:
        _module = __import__(_module_name,
                             fromlist=[native_str("obspy")])
        testsuite.addTest(_module.suite())


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
    module = __import__(module_name, fromlist=[native_str("obspy")])
    module_path = module.__path__[0]
    module_path_len = len(module_path)

    for root, _dirs, files in os.walk(module_path):
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
            parts = root[module_path_len:].split(os.sep)[1:]
            _module_name = ".".join([module_name] + parts + [file[:-3]])
            try:
                _module = __import__(_module_name,
                                     fromlist=[native_str("obspy")])
                testsuite.addTest(doctest.DocTestSuite(_module))
            except ValueError:
                pass


def write_png(arr, filename):
    """
    Custom write_png() function. matplotlib < 1.3 cannot write RGBA png files.

    Modified from https://stackoverflow.com/a/19174800
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
    abs_diff_image = np.abs(expected_image - actual_image)
    abs_diff_image *= 10.0
    save_image_np = np.clip(abs_diff_image, 0.0, 1.0)
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
    :type plt_close_all_enter: bool
    :param plt_close_all_enter: Whether to close all open figures when entering
        context (:func:`matplotlib.pyplot.close` with "all" as first argument.
    :type plt_close_all_exit: bool
    :param plt_close_all_exit: Whether to call :func:`matplotlib.pyplot.close`
        with "all" as first argument (close all figures) or no arguments (close
        active figure). Has no effect if ``plt_close=False``.
    :type style: str
    :param style: The Matplotlib style to use to generate the figure. When
        using matplotlib 1.5 or newer, the default will be ``'classic'`` to
        ensure compatibility with older releases. On older releases, the
        default will leave the style as is. You may wish to set it to
        ``'default'`` to enable the new style from Matplotlib 2.0, or some
        alternate style, which will work back to Matplotlib 1.4.0.
    :type no_uploads: bool
    :param no_uploads: If set to ``True`` no uploads to imgur are attempted, no
        matter what (e.g. any options to ``obspy-runtests`` that would normally
        cause an upload attempt). This can be used to forcibly deactivate
        upload attempts in image tests that are expected to fail.

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
                 adjust_tolerance=True, plt_close_all_enter=True,
                 plt_close_all_exit=True, style=None, no_uploads=False, *args,
                 **kwargs):
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
        self.plt_close_all_enter = plt_close_all_enter
        self.plt_close_all_exit = plt_close_all_exit
        self.no_uploads = no_uploads

        if (MATPLOTLIB_VERSION < [1, 4, 0] or
                (MATPLOTLIB_VERSION[:2] == [1, 4] and style is None)):
            # No good style support.
            self.style = None
        else:
            import matplotlib.style as mstyle
            self.style = mstyle.context(style or 'classic')

        # Adjust the tolerance based on the matplotlib version. This works
        # well enough and otherwise testing is just a pain.
        #
        # The test images were generated with matplotlib tag 291091c6eb267
        # which is after https://github.com/matplotlib/matplotlib/issues/7905
        # has been fixed.
        #
        # Thus test images should accurate for matplotlib >= 2.0.1 anf
        # fairly accurate for matplotlib 1.5.x.
        if adjust_tolerance:
            # Really old versions.
            if MATPLOTLIB_VERSION < [1, 3, 0]:
                self.tol *= 30
            # 1.3 + 1.4 have slightly different text positioning mostly.
            elif [1, 3, 0] <= MATPLOTLIB_VERSION < [1, 5, 0]:
                self.tol *= 15
            # A few plots with mpl 1.5 have ticks and axis slightl shifted.
            # This is especially true for ticks with exponential numbers.
            # Thus the tolerance also has to be a bit higher here.
            elif [1, 5, 0] <= MATPLOTLIB_VERSION < [2, 0, 0]:
                self.tol *= 5.0
            # Matplotlib 2.0.0 has a bug with the tick placement. This is
            # fixed in 2.0.1 but the tolerance for 2.0.0 has to be much
            # higher. 12 is an empiric value. The tick placement potentially
            # influences the axis locations and then the misfit is really
            # quite high.
            elif [2, 0, 0] <= MATPLOTLIB_VERSION < [2, 0, 1]:
                self.tol *= 12
            # Some section waveform plots made on 2.2.2 have offset ticks on
            # 2.0.2, so up tolerance a bit (see #2493)
            elif MATPLOTLIB_VERSION < [2, 1]:
                self.tol *= 5

            # One last pass depending on the freetype version.
            # XXX: Should eventually be handled differently!
            try:
                from matplotlib import ft2font
            except ImportError:
                pass
            else:
                if hasattr(ft2font, "__freetype_version__"):
                    if (LooseVersion(ft2font.__freetype_version__) >=
                            LooseVersion("2.8.0")):
                        self.tol *= 10

    def __enter__(self):
        """
        Set matplotlib defaults.
        """
        MatplotlibBackend.switch_backend("AGG", sloppy=False)

        from matplotlib import font_manager, rcParams, rcdefaults
        import locale

        try:
            locale.setlocale(locale.LC_ALL, native_str('en_US.UTF-8'))
        except Exception:
            try:
                locale.setlocale(locale.LC_ALL,
                                 native_str('English_United States.1252'))
            except Exception:
                msg = "Could not set locale to English/United States. " + \
                      "Some date-related tests may fail"
                warnings.warn(msg)

        # set matplotlib builtin default settings for testing
        rcdefaults()
        if self.style is not None:
            self.style.__enter__()
        if MATPLOTLIB_VERSION >= [2, 0, 0]:
            default_font = 'DejaVu Sans'
        else:
            default_font = 'Bitstream Vera Sans'
        rcParams['font.family'] = default_font
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', 'findfont:.*')
            font_manager.findfont(default_font)
            if w:
                warnings.warn('Unable to find the ' + default_font + ' font. '
                              'Plotting tests will likely fail.')
        try:
            rcParams['text.hinting'] = False
        except KeyError:
            warnings.warn("could not set rcParams['text.hinting']")
        try:
            rcParams['text.hinting_factor'] = 8
        except KeyError:
            warnings.warn("could not set rcParams['text.hinting_factor']")

        if self.plt_close_all_enter:
            import matplotlib.pyplot as plt
            try:
                plt.close("all")
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
        except ImageComparisonException as e:
            failed = True
            if "is not equal to the expected shape" in msg:
                msg = str(e) + "\n"
                upload_result = self._upload_images()
                if isinstance(upload_result, dict):
                    msg += ("\tFile:      {}\n"
                            "\tExpected:  {expected}\n"
                            "\tActual:    {actual}\n"
                            "\tDiff:      {diff}\n").format(self.image_name,
                                                            **upload_result)
                else:
                    msg += upload_result
                raise ImageComparisonException(msg)
            raise
        # we can still upload actual image if baseline image does not exist
        except IOError as e:
            failed = True
            if "Baseline image" in msg and "does not exist." in msg:
                msg = str(e) + "\n"
                upload_result = self._upload_images()
                if isinstance(upload_result, dict):
                    msg += ("\tFile:      {}\n"
                            "\tExpected:  ---\n"
                            "\tActual:    {actual}\n"
                            "\tDiff:      ---\n").format(self.image_name,
                                                         **upload_result)
                else:
                    msg += upload_result
                raise ImageComparisonException(msg)
            raise
        # simply reraise on any other unhandled exceptions
        except Exception:
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
                msg = ("Image comparison failed.\n"
                       "\tFile:      {}\n"
                       "\tRMS:       {rms}\n"
                       "\tTolerance: {tol}\n").format(self.image_name, **msg)
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
            self.close()  # flush internal buffer
            self._fileobj.close()
            if self.plt_close_all_exit:
                import matplotlib.pyplot as plt
                try:
                    plt.close("all")
                except Exception:
                    pass
            if self.style is not None:
                self.style.__exit__(exc_type, exc_val, exc_tb)
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
        Uploads images to imgur unless explicitly deactivated with option
        `no_uploads` (to speed up tests that are expected to fail).

        :returns: ``dict`` with links to uploaded images or ``str`` with
            message if upload failed
        """
        if self.no_uploads:
            return "Upload to imgur deactivated with option 'no_uploads'."
        try:
            import pyimgur
            # try to get imgur client id from environment
            imgur_clientid = \
                os.environ.get("OBSPY_IMGUR_CLIENTID") or "53b182544dc5d89"
            imgur_client_secret = \
                os.environ.get("OBSPY_IMGUR_CLIENT_SECRET", None)
            imgur_client_refresh_token = \
                os.environ.get("OBSPY_IMGUR_REFRESH_TOKEN", None)
            # upload images and return urls
            links = {}
            imgur = pyimgur.Imgur(imgur_clientid,
                                  client_secret=imgur_client_secret,
                                  refresh_token=imgur_client_refresh_token)
            if imgur_client_secret and imgur_client_refresh_token:
                try:
                    imgur.refresh_access_token()
                except Exception as e:
                    msg = ('Refreshing access token for Imgur API failed '
                           '(caught {}: {!s}).)').format(e.__class__.__name__,
                                                         e)
                    warnings.warn(msg)
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
        except Exception as e:
            msg = ("Upload to imgur failed (caught %s: %s).")
            return msg % (e.__class__.__name__, str(e))
        return links


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
    except Exception:
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


def remove_unique_ids(xml_string, remove_creation_time=False):
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


class WarningsCapture(object):
    """
    Try hard to capture all warnings.

    Aims to be a reliable drop-in replacement for built-in
    warnings.catch_warnings() context manager.

    Based on pytest's _DeprecatedCallContext context manager.
    """
    def __enter__(self):
        self.captured_warnings = []
        self._old_warn = warnings.warn
        self._old_warn_explicit = warnings.warn_explicit
        warnings.warn_explicit = self._warn_explicit
        warnings.warn = self._warn
        return self

    def _warn_explicit(self, message, category, *args, **kwargs):
        self.captured_warnings.append(
            warnings.WarningMessage(message=category(message),
                                    category=category,
                                    filename="", lineno=0))

    def _warn(self, message, category=Warning, *args, **kwargs):
        if isinstance(message, Warning):
            self.captured_warnings.append(
                warnings.WarningMessage(
                    message=category(message), category=category or Warning,
                    filename="", lineno=0))
        else:
            self.captured_warnings.append(
                warnings.WarningMessage(
                    message=category(message), category=category,
                    filename="", lineno=0))

    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.warn_explicit = self._old_warn_explicit
        warnings.warn = self._old_warn

    def __len__(self):
        return len(self.captured_warnings)

    def __getitem__(self, key):
        return self.captured_warnings[key]


def create_diverse_catalog():
    """
    Create a catalog with a single event that has many features.

    Uses most the event related classes.
    """

    # imports are here in order to avoid circular import issues
    import obspy.core.event as ev
    from obspy import UTCDateTime, Catalog
    # local dict for storing state
    state = dict(time=UTCDateTime('2016-05-04T12:00:01'))

    def _create_event():
        event = ev.Event(
            event_type='mining explosion',
            event_descriptions=[_get_event_description()],
            picks=[_create_pick()],
            origins=[_create_origins()],
            station_magnitudes=[_get_station_mag()],
            magnitudes=[_create_magnitudes()],
            amplitudes=[_get_amplitudes()],
            focal_mechanisms=[_get_focal_mechanisms()],
        )
        # set preferred origin, focal mechanism, magnitude
        preferred_objects = dict(
            origin=event.origins[-1].resource_id,
            focal_mechanism=event.focal_mechanisms[-1].resource_id,
            magnitude=event.magnitudes[-1].resource_id,
        )
        for item, value in preferred_objects.items():
            setattr(event, 'preferred_' + item + '_id', value)

        event.scope_resource_ids()
        return event

    def _create_pick():
        # setup some of the classes
        creation = ev.CreationInfo(
            agency='SwanCo',
            author='Indago',
            creation_time=UTCDateTime(),
            version='10.10',
            author_url=ev.ResourceIdentifier('smi:local/me.com'),
        )

        pick = ev.Pick(
            time=state['time'],
            comments=[ev.Comment(x) for x in 'BOB'],
            evaluation_mode='manual',
            evaluation_status='final',
            creation_info=creation,
            phase_hint='P',
            polarity='positive',
            onset='emergent',
            back_azimith_errors={"uncertainty": 10},
            slowness_method_id=ev.ResourceIdentifier('smi:local/slow'),
            backazimuth=122.1,
            horizontal_slowness=12,
            method_id=ev.ResourceIdentifier(),
            horizontal_slowness_errors={'uncertainty': 12},
            filter_id=ev.ResourceIdentifier(),
            waveform_id=ev.WaveformStreamID('UU', 'FOO', '--', 'HHZ'),
        )
        state['pick_id'] = pick.resource_id
        return pick

    def _create_origins():
        ori = ev.Origin(
            resource_id=ev.ResourceIdentifier('smi:local/First'),
            time=UTCDateTime('2016-05-04T12:00:00'),
            time_errors={'uncertainty': .01},
            longitude=-111.12525,
            longitude_errors={'uncertainty': .020},
            latitude=47.48589325,
            latitude_errors={'uncertainty': .021},
            depth=2.123,
            depth_errors={'uncertainty': 1.22},
            depth_type='from location',
            time_fixed=False,
            epicenter_fixed=False,
            reference_system_id=ev.ResourceIdentifier(),
            method_id=ev.ResourceIdentifier(),
            earth_model_id=ev.ResourceIdentifier(),
            arrivals=[_get_arrival()],
            composite_times=[_get_composite_times()],
            quality=_get_origin_quality(),
            origin_type='hypocenter',
            origin_uncertainty=_get_origin_uncertainty(),
            region='US',
            evaluation_mode='manual',
            evaluation_status='final',
        )
        state['origin_id'] = ori.resource_id
        return ori

    def _get_arrival():
        return ev.Arrival(
            resource_id=ev.ResourceIdentifier('smi:local/Ar1'),
            pick_id=state['pick_id'],
            phase='P',
            time_correction=.2,
            azimuth=12,
            distance=10,
            takeoff_angle=15,
            takeoff_angle_errors={'uncertainty': 10.2},
            time_residual=.02,
            horizontal_slowness_residual=12.2,
            backazimuth_residual=12.2,
            time_weight=.23,
            horizontal_slowness_weight=12,
            backazimuth_weight=12,
            earth_model_id=ev.ResourceIdentifier(),
            commens=[ev.Comment(x) for x in 'Nothing'],
        )

    def _get_composite_times():
        return ev.CompositeTime(
            year=2016,
            year_errors={'uncertainty': 0},
            month=5,
            month_errors={'uncertainty': 0},
            day=4,
            day_errors={'uncertainty': 0},
            hour=0,
            hour_errors={'uncertainty': 0},
            minute=0,
            minute_errors={'uncertainty': 0},
            second=0,
            second_errors={'uncertainty': .01}
        )

    def _get_origin_quality():
        return ev.OriginQuality(
            associate_phase_count=1,
            used_phase_count=1,
            associated_station_count=1,
            used_station_count=1,
            depth_phase_count=1,
            standard_error=.02,
            azimuthal_gap=.12,
            ground_truth_level='GT0',
        )

    def _get_origin_uncertainty():
        return ev.OriginUncertainty(
            horizontal_uncertainty=1.2,
            min_horizontal_uncertainty=.12,
            max_horizontal_uncertainty=2.2,
            confidence_ellipsoid=_get_confidence_ellipsoid(),
            preferred_description="uncertainty ellipse",
        )

    def _get_confidence_ellipsoid():
        return ev.ConfidenceEllipsoid(
            semi_major_axis_length=12,
            semi_minor_axis_length=12,
            major_axis_plunge=12,
            major_axis_rotation=12,
        )

    def _create_magnitudes():
        return ev.Magnitude(
            resource_id=ev.ResourceIdentifier(),
            mag=5.5,
            mag_errors={'uncertainty': .01},
            magnitude_type='Mw',
            origin_id=state['origin_id'],
            station_count=1,
            station_magnitude_contributions=[_get_station_mag_contrib()],
        )

    def _get_station_mag():
        station_mag = ev.StationMagnitude(
            mag=2.24,
        )
        state['station_mag_id'] = station_mag.resource_id
        return station_mag

    def _get_station_mag_contrib():
        return ev.StationMagnitudeContribution(
            station_magnitude_id=state['station_mag_id'],
        )

    def _get_event_description():
        return ev.EventDescription(
            text='some text about the EQ',
            type='earthquake name',
        )

    def _get_amplitudes():
        return ev.Amplitude(
            generic_amplitude=.0012,
            type='A',
            unit='m',
            period=1,
            time_window=_get_timewindow(),
            pick_id=state['pick_id'],
            scalling_time=state['time'],
            mangitude_hint='ML',
            scaling_time_errors=ev.QuantityError(uncertainty=42.0),
        )

    def _get_timewindow():
        return ev.TimeWindow(
            begin=1.2,
            end=2.2,
            reference=UTCDateTime('2016-05-04T12:00:00'),
        )

    def _get_focal_mechanisms():
        return ev.FocalMechanism(
            nodal_planes=_get_nodal_planes(),
            principal_axis=_get_principal_axis(),
            azimuthal_gap=12,
            station_polarity_count=12,
            misfit=.12,
            station_distribution_ratio=.12,
            moment_tensor=_get_moment_tensor(),
        )

    def _get_nodal_planes():
        return ev.NodalPlanes(
            nodal_plane_1=ev.NodalPlane(strike=12, dip=2, rake=12),
            nodal_plane_2=ev.NodalPlane(strike=12, dip=2, rake=12),
            preferred_plane=2,
        )

    def _get_principal_axis():
        return ev.PrincipalAxes(
            t_axis=15,
            p_axis=15,
            n_axis=15,
        )

    def _get_moment_tensor():
        return ev.MomentTensor(
            scalar_moment=12213,
            tensor=_get_tensor(),
            variance=12.23,
            variance_reduction=98,
            double_couple=.22,
            clvd=.55,
            iso=.33,
            source_time_function=_get_source_time_function(),
            data_used=[_get_data_used()],
            method_id=ev.ResourceIdentifier(),
            inversion_type='general',
        )

    def _get_tensor():
        return ev.Tensor(
            m_rr=12,
            m_rr_errors={'uncertainty': .01},
            m_tt=12,
            m_pp=12,
            m_rt=12,
            m_rp=12,
            m_tp=12,
        )

    def _get_source_time_function():
        return ev.SourceTimeFunction(
            type='triangle',
            duration=.12,
            rise_time=.33,
            decay_time=.23,
        )

    def _get_data_used():
        return ev.DataUsed(
            wave_type='body waves',
            station_count=12,
            component_count=12,
            shortest_period=1,
            longest_period=20,
        )

    events = [_create_event()]
    return Catalog(events=events)


def setup_context_testcase(test_case, cm):
    """
    Use a contextmanager to set up a unittest test case.

    Inspired by Ned Batchelder's recipe found here: goo.gl/8TBJ7s.

    :param test_case:
        An instance of unittest.TestCase
    :param cm:
        Any instances which implements the context manager protocol,
        ie its class definition implements __enter__ and __exit__ methods.
    """
    val = cm.__enter__()
    test_case.addCleanup(cm.__exit__, None, None, None)
    return val


def streams_almost_equal(st1, st2, default_stats=True, rtol=1e-05, atol=1e-08,
                         equal_nan=True):
    """
    Return True if two streams are almost equal.

    :param st1: The first :class:`~obspy.core.stream.Stream` object.
    :param st2: The second :class:`~obspy.core.stream.Stream` object.
    :param default_stats:
        If True only compare the default stats on the traces, such as seed
        identification codes, start/end times, sampling_rates, etc. If
        False also compare extra stats attributes such as processing and
        format specific information.
    :param rtol: The relative tolerance parameter passed to
        :func:`~numpy.allclose` for comparing time series.
    :param atol: The absolute tolerance parameter passed to
        :func:`~numpy.allclose` for comparing time series.
    :param equal_nan:
        If ``True`` NaNs are evaluated equal when comparing the time
        series.
    :return: bool

    .. rubric:: Example

    1) Changes to the non-default parameters of the
        :class:`~obspy.core.trace.Stats` objects of the stream's contained
        :class:`~obspy.core.trace.Trace` objects will cause the streams to
        be considered unequal, but they will be considered almost equal.

        >>> from obspy import read
        >>> st1 = read()
        >>> st2 = read()
        >>> # The traces should, of course, be equal.
        >>> assert st1 == st2
        >>> # Perform detrending on st1 twice so processing stats differ.
        >>> st1 = st1.detrend('linear')
        >>> st1 = st1.detrend('linear')
        >>> st2 = st2.detrend('linear')
        >>> # The traces are no longer equal, but are almost equal.
        >>> assert st1 != st2
        >>> assert streams_almost_equal(st1, st2)


    2) Slight differences in each trace's data will cause the streams
        to be considered unequal, but they will be almost equal if the
        differences don't exceed the limits set by the ``rtol`` and
        ``atol`` parameters.

        >>> from obspy import read
        >>> st1 = read()
        >>> st2 = read()
        >>> # Perturb the trace data in st2 slightly.
        >>> for tr in st2:
        ...     tr.data *= (1 + 1e-6)
        >>> # The streams are no longer equal.
        >>> assert st1 != st2
        >>> # But they are almost equal.
        >>> assert streams_almost_equal(st1, st2)
        >>> # Unless, of course, there is a large change.
        >>> st1[0].data *= 10
        >>> assert not streams_almost_equal(st1, st2)
    """
    from obspy.core.stream import Stream
    # Return False if both objects are not streams or not the same length.
    are_streams = isinstance(st1, Stream) and isinstance(st2, Stream)
    if not are_streams or not len(st1) == len(st2):
        return False
    # Kwargs to pass trace_almost_equal.
    tr_kwargs = dict(default_stats=default_stats, rtol=rtol, atol=atol,
                     equal_nan=equal_nan)
    # Ensure the streams are sorted (as done with the __equal__ method)
    st1_sorted = st1.select()
    st1_sorted.sort()
    st2_sorted = st2.select()
    st2_sorted.sort()
    # Iterate over sorted trace pairs and determine if they are almost equal.
    for tr1, tr2 in zip(st1_sorted, st2_sorted):
        if not traces_almost_equal(tr1, tr2, **tr_kwargs):
            return False  # If any are not almost equal return None.
    return True


def traces_almost_equal(tr1, tr2, default_stats=True, rtol=1e-05, atol=1e-08,
                        equal_nan=True):
    """
    Return True if the two traces are almost equal.

    :param tr1: The first :class:`~obspy.core.trace.Trace` object.
    :param tr2: The second :class:`~obspy.core.trace.Trace` object.
    :param default_stats:
        If True only compare the default stats on the traces, such as seed
        identification codes, start/end times, sampling_rates, etc. If
        False also compare extra stats attributes such as processing and
        format specific information.
    :param rtol: The relative tolerance parameter passed to
        :func:`~numpy.allclose` for comparing time series.
    :param atol: The absolute tolerance parameter passed to
        :func:`~numpy.allclose` for comparing time series.
    :param equal_nan:
        If ``True`` NaNs are evaluated equal when comparing the time
        series.
    :return: bool
    """
    from obspy.core.trace import Trace
    # If other isnt  a trace, or data is not the same len return False.
    if not isinstance(tr2, Trace) or len(tr1.data) != len(tr2.data):
        return False
    # First compare the array values
    try:  # Use equal_nan if available
        all_close = np.allclose(tr1.data, tr2.data, rtol=rtol,
                                atol=atol, equal_nan=equal_nan)
    except TypeError:
        # This happens on very old versions of numpy. Essentially
        # we just need to handle NaN detection on our own, if equal_nan.
        is_close = np.isclose(tr1.data, tr2.data, rtol=rtol, atol=atol)
        if equal_nan:
            isnan = np.isnan(tr1.data) & np.isnan(tr2.data)
        else:
            isnan = np.zeros(tr1.data.shape).astype(bool)
        all_close = np.all(isnan | is_close)
    # Then compare the stats objects
    stats1 = _make_stats_dict(tr1, default_stats)
    stats2 = _make_stats_dict(tr2, default_stats)
    return all_close and stats1 == stats2


def _make_stats_dict(tr, default_stats):
    """
    Return a dict of stats from trace optionally including processing.
    """
    from obspy.core.trace import Stats
    if not default_stats:
        return dict(tr.stats)
    return {i: tr.stats[i] for i in Stats.defaults}


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
