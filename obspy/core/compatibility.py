# -*- coding: utf-8 -*-
"""
ObsPy's compatibility layer.

Includes things to easy dealing with Python version differences as well as
making it work with various versions of our dependencies.
"""
import collections
import importlib
import json

import numpy as np


# Importing the ABCs from collections will no longer work with Python 3.8.
try:
    collections_abc = collections.abc  # NOQA
except AttributeError:
    # Python 3.4 compat, see https://bugs.python.org/msg212284
    # some older Linux distribution (like Debian jessie) are still in LTS,
    # so be nice, this doesn't hurt and can be removed again later on
    collections_abc = importlib.import_module("collections.abc")  # NOQA


# NumPy does not offer the from_buffer method under Python 3 and instead
# relies on the built-in memoryview object.
def from_buffer(data, dtype):
    try:
        data = data.encode()
    except Exception:
        pass
    return np.array(memoryview(data)).view(dtype).copy()  # NOQA


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


def get_json_from_response(r):
    """
    Get a JSON response in a way that also works for very old request
    versions.

    :type r: :class:`requests.Response
    :param r: The server's response.
    """
    if hasattr(r, "json"):
        if isinstance(r.json, dict):
            return r.json
        return r.json()

    c = r.content
    try:
        c = c.decode()
    except Exception:
        pass
    return json.loads(c)


def get_text_from_response(r):
    """
    Get a text response in a way that also works for very old request versions.

    :type r: :class:`requests.Response
    :param r: The server's response.
    """
    if hasattr(r, "text"):
        return r.text

    c = r.content
    try:
        c = c.decode()
    except Exception:
        pass
    return c


def get_reason_from_response(r):
    """
    Get the status text.

    :type r: :class:`requests.Response
    :param r: The server's response.
    """
    # Very old requests version might not have the reason attribute.
    if hasattr(r, "reason"):
        c = r.reason
    else:  # pragma: no cover
        c = r.raw.reason

    if hasattr(c, "encode"):
        c = c.encode()

    return c
