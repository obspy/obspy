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
from distutils.version import LooseVersion
import doctest
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


if __name__ == '__main__':
    doctest.testmod(exclude_empty=True)
