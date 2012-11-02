# -*- coding: utf-8 -*-
"""
Decorator used in ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.util.base import NamedTemporaryFile
import functools
import os
import unittest
import warnings


def deprecated(func, warning_msg=None):
    """
    This is a decorator which can be used to mark functions as deprecated.

    It will result in a warning being emitted when the function is used.
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        if 'deprecated' in str(func.__doc__).lower():
            msg = func.__doc__
        elif warning_msg:
            msg = warning_msg
        else:
            msg = "Call to deprecated function %s." % func.__name__
        warnings.warn(msg, category=DeprecationWarning)
        return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


def deprecated_keywords(keywords):
    """
    Decorator for marking keywords as deprecated.

    :type keywords: dict
    :param keywords: old/new keyword names as key/value pairs.
    """
    def fdec(func):
        fname = func.func_name
        msg = "Deprecated keyword %s in %s() call - please use %s instead."
        msg2 = "Deprecated keyword %s in %s() call - ignoring."

        @functools.wraps(func)
        def echo_func(*args, **kwargs):
            for kw in kwargs.keys():
                if kw in keywords:
                    nkw = keywords[kw]
                    if nkw is None:
                        warnings.warn(msg2 % (kw, fname),
                                      category=DeprecationWarning)
                    else:
                        warnings.warn(msg % (kw, fname, nkw),
                                      category=DeprecationWarning)
                        kwargs[nkw] = kwargs[kw]
                    del(kwargs[kw])
            return func(*args, **kwargs)
        return echo_func

    return fdec


def skip(reason):
    """
    Unconditionally skip a test.
    """
    def decorator(test_item):
        if not (isinstance(test_item, type) and issubclass(test_item,
                                                           unittest.TestCase)):
            @functools.wraps(test_item)
            def skip_wrapper(*args, **kwargs):  # @UnusedVariable
                return

            test_item = skip_wrapper

        test_item.__unittest_skip__ = True
        test_item.__unittest_skip_why__ = reason
        return test_item
    return decorator


def skipIf(condition, reason):
    """
    Skip a test if the condition is true.
    """
    if condition:
        return skip(reason)

    def _id(obj):
        return obj

    return _id


def uncompressFile(func):
    """
    Decorator used for temporary uncompressing file if .gz or .bz2 archive.
    """
    def wrapped_func(filename, *args, **kwargs):
        if not isinstance(filename, basestring):
            return func(filename, *args, **kwargs)
        elif not os.path.exists(filename):
            msg = "File not found '%s'" % (filename)
            raise IOError(msg)
        # check if we got a compressed file
        unpacked_data = None
        if filename.endswith('.bz2'):
            # bzip2
            try:
                import bz2
                unpacked_data = bz2.decompress(open(filename, 'rb').read())
            except:
                pass
        elif filename.endswith('.gz'):
            # gzip
            try:
                import gzip
                unpacked_data = gzip.open(filename, 'rb').read()
            except:
                pass
        if unpacked_data:
            # we unpacked something without errors - create temporary file
            tempfile = NamedTemporaryFile()
            tempfile._fileobj.write(unpacked_data)
            tempfile.close()
            # call wrapped function
            try:
                result = func(tempfile.name, *args, **kwargs)
            except:
                # clean up unpacking procedure
                if unpacked_data:
                    tempfile.close()
                    os.remove(tempfile.name)
                raise
            # clean up unpacking procedure
            if unpacked_data:
                tempfile.close()
                os.remove(tempfile.name)
        else:
            # call wrapped function with original filename
            result = func(filename, *args, **kwargs)
        return result
    return wrapped_func


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
