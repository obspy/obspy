# -*- coding: utf-8 -*-
"""
AttribDict class for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import collections
import copy


class AttribDict(collections.MutableMapping):
    """
    A class which behaves like a dictionary.

    :type data: dict, optional
    :param data: Dictionary with initial keywords.

    .. rubric:: Basic Usage

    You may use the following syntax to change or access data in this class.

    >>> stats = AttribDict()
    >>> stats.network = 'BW'
    >>> stats['station'] = 'ROTZ'
    >>> print(stats.get('network'))
    BW
    >>> print(stats['network'])
    BW
    >>> print(stats.station)
    ROTZ
    >>> x = stats.keys()
    >>> x = sorted(x)
    >>> print(x[0], x[1])
    network station
    """
    defaults = {}
    readonly = []

    def __init__(self, *args, **kwargs):
        """
        An AttribDict can be initialized in two ways. It can be given an
        existing dictionary as a simple argument or alternatively all keyword
        arguments will become (key, value) pairs.

        >>> attrib_dict_1 = AttribDict({"a":1, "b":2})
        >>> attrib_dict_2 = AttribDict(a=1, b=2)
        >>> attrib_dict_1  #doctest: +SKIP
        AttribDict({'a': 1, 'b': 2})
        >>> assert(attrib_dict_1 == attrib_dict_2)
        """
        # set default values directly
        self.__dict__.update(self.defaults)
        # use overwritable update method to set arguments
        self.update(dict(*args, **kwargs))

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.__dict__)

    def __getitem__(self, name, default=None):
        try:
            return self.__dict__[name]
        except KeyError:
            # check if we got any default value given at class level
            if name in self.defaults:
                return self.defaults[name]
            # if both are missing check for a given default value
            if default is None:
                raise
            return default

    def __setitem__(self, key, value):
        if key in self.readonly:
            msg = 'Attribute "%s" in %s object is read only!'
            raise AttributeError(msg % (key, self.__class__.__name__))

        if isinstance(value, collections.Mapping) and \
           not isinstance(value, AttribDict):
            self.__dict__[key] = AttribDict(value)
        else:
            self.__dict__[key] = value

    def __delitem__(self, name):
        del self.__dict__[name]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, adict):
        # set default values
        self.__dict__.update(self.defaults)
        # update with pickle dictionary
        self.update(adict)

    def __getattr__(self, name, default=None):
        """
        Py3k hasattr() expects an AttributeError no KeyError to be
        raised if the attribute is not found.
        """
        try:
            return self.__getitem__(name, default)
        except KeyError as e:
            raise AttributeError(e.args[0])

    __setattr__ = __setitem__
    __delattr__ = __delitem__

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, *args, **kwargs):  # @UnusedVariable
        ad = self.__class__()
        ad.update(copy.deepcopy(self.__dict__))
        return ad

    def update(self, adict={}):
        for (key, value) in adict.items():
            if key in self.readonly:
                continue
            self.__setitem__(key, value)

    def _pretty_str(self, priorized_keys=[], min_label_length=16):
        """
        Return better readable string representation of AttribDict object.

        :type priorized_keys: list of str, optional
        :param priorized_keys: Keywords of current AttribDict which will be
            shown before all other keywords. Those keywords must exists
            otherwise an exception will be raised. Defaults to empty list.
        :type min_label_length: int, optional
        :param min_label_length: Minimum label length for keywords. Defaults
            to ``16``.
        :return: String representation of current AttribDict object.
        """
        keys = list(self.keys())
        # determine longest key name for alignment of all items
        try:
            i = max(max([len(k) for k in keys]), min_label_length)
        except ValueError:
            # no keys
            return ""
        pattern = "%%%ds: %%s" % (i)
        # check if keys exist
        other_keys = [k for k in keys if k not in priorized_keys]
        # priorized keys first + all other keys
        keys = priorized_keys + sorted(other_keys)
        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
