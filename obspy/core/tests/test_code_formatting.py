# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import codecs
import fnmatch
import inspect
import os
import re
import unittest

from obspy.core.util.decorator import skipIf
from obspy.core.util.testing import check_flake8


class CodeFormattingTestCase(unittest.TestCase):
    """
    Test codebase for compliance with the flake8 tool.
    """

    @skipIf('OBSPY_NO_FLAKE8' in os.environ, 'flake8 check disabled')
    def test_flake8(self):
        """
        Test codebase for compliance with the flake8 tool.
        """
        report, message = check_flake8()
        file_count = report.counters["files"]
        error_count = report.get_count()
        self.assertTrue(file_count > 10)
        self.assertEqual(error_count, 0, message)


class FutureUsageTestCase(unittest.TestCase):
    def test_future_imports_in_every_file(self):
        """
        Tests that every single Python file includes the appropriate future
        headers to enforce consistent behavior.
        """
        test_dir = os.path.abspath(inspect.getfile(inspect.currentframe()))
        obspy_dir = os.path.dirname(os.path.dirname(os.path.dirname(test_dir)))

        # There are currently only three exceptions. Two files are imported
        # during installation and thus cannot contain future imports. The
        # third file is the compatibility layer which naturally also does
        # not want to import future.
        exceptions = [
            os.path.join('core', 'util', 'libnames.py'),
            os.path.join('core', 'util', 'version.py'),
            os.path.join('core', 'compatibility.py'),
            os.path.join('lib', '*'),
        ]
        exceptions = [os.path.join(obspy_dir, i) for i in exceptions]

        def _match_exceptions(filename):
            for pattern in exceptions:
                if fnmatch.fnmatch(filename, pattern):
                    return True
            return False

        future_import_line = (
            "from __future__ import (absolute_import, division, "
            "print_function, unicode_literals)")
        builtins_line = "from future.builtins import *  # NOQA"

        future_imports_pattern = re.compile(
            r"^from __future__ import \(absolute_import,\s*"
            r"division,\s*print_function,\s*unicode_literals\)$",
            flags=re.MULTILINE)

        builtin_pattern = re.compile(
            r"^from future\.builtins import \*  # NOQA",
            flags=re.MULTILINE)

        failures = []
        # Walk the obspy directory.
        for dirpath, _, filenames in os.walk(obspy_dir):
            # Find all Python files.
            filenames = [os.path.abspath(os.path.join(dirpath, i)) for i in
                         filenames if i.endswith(".py")]
            for filename in filenames:
                if _match_exceptions(filename):
                    continue
                with codecs.open(filename, "r", encoding="utf-8") as fh:
                    content = fh.read()

                    if re.search(future_imports_pattern, content) is None:
                        failures.append("File '%s' misses imports: %s" %
                                        (filename, future_import_line))

                    if re.search(builtin_pattern, content) is None:
                        failures.append("File '%s' misses imports: %s" %
                                        (filename, builtins_line))
        self.assertEqual(len(failures), 0, "\n" + "\n".join(failures))


def suite():

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CodeFormattingTestCase, 'test'))
    suite.addTest(unittest.makeSuite(FutureUsageTestCase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
