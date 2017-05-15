# -*- coding: utf-8 -*-
"""
Integration with ObsPy's core classes.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import io
import re

from .parser import is_xseed


def _is_seed(filename):
    """
    Determine if the file is (dataless) SEED file.

    No comprehensive check - it only checks the initial record sequence 
    number and the very first blockette.

    :type filename: str
    :param filename: Path/filename of a local file to be checked.
    :rtype: bool
    :returns: `True` if file seems to be a RESP file, `False` otherwise.
    """
    try:
        if hasattr(filename, "read") and hasattr(filename, "seek") and \
                hasattr(filename, "tell"):
            pos = filename.tell()
            try:
                buf = filename.read(128)
            finally:
                filename.seek(pos, 0)
        else:
            with io.open(filename, "rb") as fh:
                buf = fh.read(128)
    except IOError:
        return False

    # Minimum record size.
    if len(buf) < 128:
        return False

    if buf[:8] != b"000001V ":
        return False

    if buf[8: 8 + 3] not in [b"010", b"008", b"005"]:
        return False

    return True


def _is_xseed(filename):
    """
    Determine if the file is an XML-SEED file.

    Does not do any schema validation but only check the root tag.

    :type filename: str
    :param filename: Path/filename of a local file to be checked.
    :rtype: bool
    :returns: `True` if file seems to be a RESP file, `False` otherwise.
    """
    return is_xseed(filename)


def _is_resp(filename):
    """
    Check if a file at the specified location appears to be a RESP file.

    :type filename: str
    :param filename: Path/filename of a local file to be checked.
    :rtype: bool
    :returns: `True` if file seems to be a RESP file, `False` otherwise.
    """
    try:
        with open(filename, "rb") as fh:
            try:
                # lookup the first line that does not start with a hash sign
                while True:
                    # use splitlines to correctly detect e.g. mac formatted
                    # files on Linux
                    lines = fh.readline().splitlines()
                    # end of file without finding an appropriate line
                    if not lines:
                        return False
                    # check each line after splitting them
                    for line in lines:
                        if line.decode().startswith("#"):
                            continue
                        # do the regex check on the first non-comment line
                        if re.match(r'[bB]0[1-6][0-9]F[0-9]{2} ',
                                    line.decode()):
                            return True
                        return False
            except UnicodeDecodeError:
                return False
    except IOError:
        return False


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)