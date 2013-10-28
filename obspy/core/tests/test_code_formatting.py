# -*- coding: utf-8 -*-

from obspy.core.util.misc import get_untracked_files_from_git
from obspy.core.util.decorator import skipIf
import fnmatch
import inspect
import os
import sys
import unittest
import StringIO

EXCLUDE_FILES = [
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
    test_dir = os.path.abspath(inspect.getfile(inspect.currentframe()))
    obspy_dir = os.path.dirname(os.path.dirname(os.path.dirname(test_dir)))
    error_count = 0
    file_count = 0
    untracked_files = get_untracked_files_from_git() or []
    sys.stdout = StringIO.StringIO()
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
            for exclude_pattern in EXCLUDE_FILES:
                if fnmatch.fnmatch(py_file, exclude_pattern):
                    break
            else:
                file_count += 1
                if flake8.main.check_file(py_file):
                    error_count += 1
    sys.stdout.seek(0)
    message = sys.stdout.read()
    sys.stdout = sys.__stdout__
    return error_count, message, file_count


class CodeFormattingTestCase(unittest.TestCase):
    """
    Test codebase for compliance with the flake8 tool.
    """
    @skipIf('OBSPY_NO_FLAKE8' in os.environ, 'flake8 check disabled')
    def test_flake8(self):
        """
        Test codebase for compliance with the flake8 tool.
        """
        error_count, message, file_count = check_flake8()
        self.assertTrue(file_count > 10)
        self.assertEqual(error_count, 0, message)


def suite():
    return unittest.makeSuite(CodeFormattingTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
