# -*- coding: utf-8 -*-
"""
Various types used in ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

# try native OrderDict implementations first (Python >= 2.7.x)
try:
    from collections import OrderedDict
except ImportError:
    # Copyright (c) 2009 Raymond Hettinger
    #
    # Permission is hereby granted, free of charge, to any person
    # obtaining a copy of this software and associated documentation files
    # (the "Software"), to deal in the Software without restriction,
    # including without limitation the rights to use, copy, modify, merge,
    # publish, distribute, sublicense, and/or sell copies of the Software,
    # and to permit persons to whom the Software is furnished to do so,
    # subject to the following conditions:
    #
    #     The above copyright notice and this permission notice shall be
    #     included in all copies or substantial portions of the Software.
    #
    #     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    #     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    #     OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    #     NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    #     HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    #     WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    #     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    #     OTHER DEALINGS IN THE SOFTWARE.
    from UserDict import DictMixin

    class OrderedDict(dict, DictMixin):
        """
        Dictionary that remembers insertion order.
        """
        def __init__(self, *args, **kwds):
            if len(args) > 1:
                msg = 'expected at most 1 arguments, got %d'
                raise TypeError(msg % len(args))
            try:
                self.__end
            except AttributeError:
                self.clear()
            self.update(*args, **kwds)

        def clear(self):
            self.__end = end = []
            end += [None, end, end]      # sentinel node for doubly linked list
            self.__map = {}              # key --> [key, prev, next]
            dict.clear(self)

        def __setitem__(self, key, value):
            if key not in self:
                end = self.__end
                curr = end[1]
                curr[2] = end[1] = self.__map[key] = [key, curr, end]
            dict.__setitem__(self, key, value)

        def __delitem__(self, key):
            dict.__delitem__(self, key)
            key, prev, next = self.__map.pop(key)
            prev[2] = next
            next[1] = prev

        def __iter__(self):
            end = self.__end
            curr = end[2]
            while curr is not end:
                yield curr[0]
                curr = curr[2]

        def __reversed__(self):
            end = self.__end
            curr = end[1]
            while curr is not end:
                yield curr[0]
                curr = curr[1]

        def popitem(self, last=True):
            if not self:
                raise KeyError('dictionary is empty')
            if last:
                key = reversed(self).next()
            else:
                key = iter(self).next()
            value = self.pop(key)
            return key, value

        def __reduce__(self):
            items = [[k, self[k]] for k in self]
            tmp = self.__map, self.__end
            del self.__map, self.__end
            inst_dict = vars(self).copy()
            self.__map, self.__end = tmp
            if inst_dict:
                return (self.__class__, (items,), inst_dict)
            return self.__class__, (items,)

        def keys(self):
            return list(self)

        setdefault = DictMixin.setdefault
        update = DictMixin.update
        pop = DictMixin.pop
        values = DictMixin.values
        items = DictMixin.items
        iterkeys = DictMixin.iterkeys
        itervalues = DictMixin.itervalues
        iteritems = DictMixin.iteritems

        def __repr__(self):
            if not self:
                return '%s()' % (self.__class__.__name__,)
            return '%s(%r)' % (self.__class__.__name__, self.items())

        def copy(self):
            return self.__class__(self)

        @classmethod
        def fromkeys(cls, iterable, value=None):
            d = cls()
            for key in iterable:
                d[key] = value
            return d

        def __eq__(self, other):
            if isinstance(other, OrderedDict):
                if len(self) != len(other):
                    return False
                for p, q in  zip(self.items(), other.items()):
                    if p != q:
                        return False
                return True
            return dict.__eq__(self, other)

        def __ne__(self, other):
            return not self == other


class Enum(object):
    """
    Enumerated type (enum) implementation for Python.

    :type enums: list of str

    .. rubric:: Example

    >>> from obspy.core.util import Enum
    >>> units = Enum(["m", "s", "m/s", "m/(s*s)", "m*s", "other"])

    There are different ways to access the correct enum values:

        >>> units.get('m/s')
        'm/s'
        >>> units['S']
        's'
        >>> units.OTHER
        'other'
        >>> units[3]
        'm/(s*s)'
        >>> units.xxx  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        KeyError: 'xxx'

    Changing enum values will not work:

        >>> units.m = 5  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        NotImplementedError
        >>> units['m'] = 'xxx'  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        NotImplementedError

    Calling with a value will either return the mapped enum value or ``None``:

        >>> units("M*s")
        'm*s'
        >>> units('xxx')
        >>> units(5)
        'other'
    """
    # marker needed for for usage within ABC classes
    __isabstractmethod__ = False

    def __init__(self, enums):
        self.__enums = OrderedDict(zip([str(e).lower() for e in enums], enums))

    def __call__(self, enum):
        try:
            return self.get(enum)
        except:
            return None

    def get(self, key):
        if isinstance(key, int):
            return self.__enums.values()[key]
        return self.__enums.__getitem__(key.lower())

    __getattr__ = get
    __getitem__ = get

    def set(self, name, value):
        if name == '_Enum__enums':
            self.__dict__[name] = value
            return
        raise NotImplementedError

    __setattr__ = set
    __setitem__ = set

    def __contains__(self, value):
        return value.lower() in self.__enums

    def values(self):
        return self.__enums.values()

    def keys(self):
        return self.__enums.keys()

    def items(self):
        return self.__enums.items()

    def iteritems(self):
        return self.__enums.iteritems()

    def __str__(self):
        """
        >>> enum = Enum(["c", "a", "b"])
        >>> print enum
        Enum(["c", "a", "b"])
        """
        keys = self.__enums.keys()
        return "Enum([%s])" % ", ".join(['"%s"' % _i for _i in keys])


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
