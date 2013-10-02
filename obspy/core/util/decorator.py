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
import numpy as np
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
        # check if we got a compressed file or archive
        obj_list = []
        if filename.endswith('.tar') or filename.endswith('.tgz') or \
                filename.endswith('.tar.gz') or filename.endswith('.tar.bz2'):
            # tarfile module
            try:
                import tarfile
                if not tarfile.is_tarfile(filename):
                    raise
                # reading with transparent compression
                tar = tarfile.open(filename, 'r|*')
                for tarinfo in tar:
                    # only handle regular files
                    if not tarinfo.isfile():
                        continue
                    data = tar.extractfile(tarinfo).read()
                    obj_list.append(data)
                tar.close()
            except:
                pass
        elif filename.endswith('.zip'):
            # zipfile module
            try:
                import zipfile
                if not zipfile.is_zipfile(filename):
                    raise
                zip = zipfile.ZipFile(filename)
                obj_list = [zip.read(name) for name in zip.namelist()]
            except:
                pass
        elif filename.endswith('.bz2'):
            # bz2 module
            try:
                import bz2
                obj_list.append(bz2.decompress(open(filename, 'rb').read()))
            except:
                pass
        elif filename.endswith('.gz'):
            # gzip module
            try:
                import gzip
                obj_list.append(gzip.open(filename, 'rb').read())
            except:
                pass
        # handle results
        if obj_list:
            # write results to temporary files
            result = None
            for obj in obj_list:
                with NamedTemporaryFile() as tempfile:
                    tempfile._fileobj.write(obj)
                    stream = func(tempfile.name, *args, **kwargs)
                    # just add other stream objects to first stream
                    if result is None:
                        result = stream
                    else:
                        result += stream
        else:
            # no compressions
            result = func(filename, *args, **kwargs)
        return result
    return wrapped_func


def raiseIfMasked(func):
    """
    Raises if the first argument (self in case of methods) is a Trace with
    masked values or a Stream containing a Trace with masked values.
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        arrays = []
        # first arg seems to be a Stream
        if hasattr(args[0], "traces"):
            arrays = [tr.data for tr in args[0]]
        # first arg seems to be a Trace
        if hasattr(args[0], "data") and isinstance(args[0].data, np.ndarray):
            arrays = [args[0].data]
        for arr in arrays:
            if np.ma.is_masked(arr):
                msg = "Trace with masked values found. This is not " + \
                      "supported for this operation. Try the split() " + \
                      "method on Trace/Stream to produce a Stream with " + \
                      "unmasked Traces."
                raise NotImplementedError(msg)
        return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
