# -*- coding: utf-8 -*-
"""
Py3k compatibility module
"""
from future.utils import PY2

import inspect
import io
import sys

import numpy as np

# optional dependencies
try:
    if PY2:
        import mock  # NOQA
    else:
        from unittest import mock  # NOQA
except:
    pass

if PY2:
    from string import maketrans
else:
    maketrans = bytes.maketrans


# NumPy does not offer the frombuffer method under Python 3 and instead
# relies on the built-in memoryview object.
if PY2:
    def frombuffer(data, dtype):
        # For compatibility with NumPy 1.4
        if isinstance(dtype, unicode):  # noqa
            dtype = str(dtype)
        if data:
            return np.frombuffer(data, dtype=dtype).copy()
        else:
            return np.array([], dtype=dtype)
else:
    def frombuffer(data, dtype):
        return np.array(memoryview(data)).view(dtype).copy()  # NOQA


def is_text_buffer(obj):
    """
    Helper function determining if the passed object is an object that can
    read and write text or not.

    :param obj: The object to be tested.
    :return: True/False
    """
    # Default open()'ed files and StringIO (in Python 2) don't inherit from any
    # of the io classes thus we only test the methods of the objects which
    # in Python 2 should be safe enough.
    if PY2 and not isinstance(obj, io.BufferedIOBase) and \
            not isinstance(obj, io.TextIOBase):
        if hasattr(obj, "read") and hasattr(obj, "write") \
                and hasattr(obj, "seek") and hasattr(obj, "tell"):
            return True
        return False

    return isinstance(obj, io.TextIOBase)


def is_bytes_buffer(obj):
    """
    Helper function determining if the passed object is an object that can
    read and write bytes or not.

    :param obj: The object to be tested.
    :return: True/False
    """
    # Default open()'ed files and StringIO (in Python 2) don't inherit from any
    # of the io classes thus we only test the methods of the objects which
    # in Python 2 should be safe enough.
    if PY2 and not isinstance(obj, io.BufferedIOBase) and \
            not isinstance(obj, io.TextIOBase):
        if hasattr(obj, "read") and hasattr(obj, "write") \
                and hasattr(obj, "seek") and hasattr(obj, "tell"):
            return True
        return False

    return isinstance(obj, io.BufferedIOBase)


def round_away(number):
    """
    Simple function that rounds a number to the nearest integer. If the number
    is halfway between two integers, it will round away from zero. Of course
    only works up machine precision. This should hopefully behave like the
    round() function in Python 2.

    This is potentially desired behavior in the trim functions but some more
    thought should be poured into it.

    The np.round() function rounds towards the even nearest even number in case
    of half-way splits.

    >>> round_away(2.5)
    3
    >>> round_away(-2.5)
    -3

    >>> round_away(10.5)
    11
    >>> round_away(-10.5)
    -11

    >>> round_away(11.0)
    11
    >>> round_away(-11.0)
    -11
    """

    floor = np.floor(number)
    ceil = np.ceil(number)
    if (floor != ceil) and (abs(number - floor) == abs(ceil - number)):
        return int(int(number) + int(np.sign(number)))
    else:
        return int(np.round(number))


# If not Python 2.6 return the getcallargs function.
if not (sys.version_info[0] == 2 and sys.version_info[1] == 6):
    getcallargs = inspect.getcallargs
else:
    # Otherwise redefine it here. This is a copy from the Python 2.7 stdlib
    # source code with only minor modifications.
    def getcallargs(func, *positional, **named):
        """Get the mapping of arguments to values.

        A dict is returned, with keys the function argument names (including
        the names of the * and ** arguments, if any), and values the respective
        bound values from 'positional' and 'named'."""
        args, varargs, varkw, defaults = inspect.getargspec(func)
        f_name = func.__name__
        arg2value = {}

        # The following closures are basically because of tuple parameter
        # unpacking.
        assigned_tuple_params = []

        def assign(arg, value):
            if isinstance(arg, str):
                arg2value[arg] = value
            else:
                assigned_tuple_params.append(arg)
                value = iter(value)
                for i, subarg in enumerate(arg):
                    try:
                        subvalue = next(value)
                    except StopIteration:
                        raise ValueError('need more than %d %s to unpack' %
                                         (i, 'values' if i > 1 else 'value'))
                    assign(subarg, subvalue)
                try:
                    next(value)
                except StopIteration:
                    pass
                else:
                    raise ValueError('too many values to unpack')

        def is_assigned(arg):
            if isinstance(arg, str):
                return arg in arg2value
            return arg in assigned_tuple_params
        if inspect.ismethod(func) and func.im_self is not None:
            # implicit 'self' (or 'cls' for classmethods) argument
            positional = (func.im_self,) + positional
        num_pos = len(positional)
        num_total = num_pos + len(named)
        num_args = len(args)
        num_defaults = len(defaults) if defaults else 0
        for arg, value in zip(args, positional):
            assign(arg, value)
        if varargs:
            if num_pos > num_args:
                assign(varargs, positional[-(num_pos-num_args):])
            else:
                assign(varargs, ())
        elif 0 < num_args < num_pos:
            raise TypeError('%s() takes %s %d %s (%d given)' % (
                f_name, 'at most' if defaults else 'exactly', num_args,
                'arguments' if num_args > 1 else 'argument', num_total))
        elif num_args == 0 and num_total:
            if varkw:
                if num_pos:
                    # XXX: We should use num_pos, but Python also uses
                    # num_total:
                    raise TypeError('%s() takes exactly 0 arguments '
                                    '(%d given)' % (f_name, num_total))
            else:
                raise TypeError('%s() takes no arguments (%d given)' %
                                (f_name, num_total))
        for arg in args:
            if isinstance(arg, str) and arg in named:
                if is_assigned(arg):
                    raise TypeError("%s() got multiple values for keyword "
                                    "argument '%s'" % (f_name, arg))
                else:
                    assign(arg, named.pop(arg))
        if defaults:    # fill in any missing values with the defaults
            for arg, value in zip(args[-num_defaults:], defaults):
                if not is_assigned(arg):
                    assign(arg, value)
        if varkw:
            assign(varkw, named)
        elif named:
            unexpected = next(iter(named))
            if isinstance(unexpected, str):
                unexpected = unexpected.encode(sys.getdefaultencoding(),
                                               'replace')
            raise TypeError("%s() got an unexpected keyword argument '%s'" %
                            (f_name, unexpected))
        unassigned = num_args - len([arg for arg in args if is_assigned(arg)])
        if unassigned:
            num_required = num_args - num_defaults
            raise TypeError('%s() takes %s %d %s (%d given)' % (
                f_name, 'at least' if defaults else 'exactly', num_required,
                'arguments' if num_required > 1 else 'argument', num_total))
        return arg2value
