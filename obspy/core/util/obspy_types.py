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
                for p, q in zip(self.items(), other.items()):
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
    :type replace: dict, optional
    :param replace: Dictionary of keys which are replaced by values.

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

    The following enum allows replacing certain entries:

        >>> units2 = Enum(["m", "s", "m/s", "m/(s*s)", "m*s", "other"],
        ...               replace={'meter': 'm'})
        >>> units2('m')
        'm'
        >>> units2('meter')
        'm'
    """
    # marker needed for for usage within ABC classes
    __isabstractmethod__ = False

    def __init__(self, enums, replace={}):
        self.__enums = OrderedDict(zip([str(e).lower() for e in enums], enums))
        self.__replace = replace

    def __call__(self, enum):
        try:
            return self.get(enum)
        except:
            return None

    def get(self, key):
        if isinstance(key, int):
            return self.__enums.values()[key]
        if key in self._Enum__replace:
            return self._Enum__replace[key.lower()]
        return self.__enums.__getitem__(key.lower())

    __getattr__ = get
    __getitem__ = get

    def __setattr__(self, name, value):
        if name == '_Enum__enums':
            self.__dict__[name] = value
            return
        elif name == '_Enum__replace':
            super(Enum, self).__setattr__(name, value)
            return
        raise NotImplementedError

    __setitem__ = __setattr__

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


class CustomComplex(complex):
    """
    Helper class to inherit from and which stores a complex number that is
    extendable.
    """
    def __new__(cls, *args):
        return super(CustomComplex, cls).__new__(cls, *args)

    def __init__(self, *args):
        pass

    def __add__(self, other):
        new = self.__class__(complex(self) + other)
        new.__dict__.update(**self.__dict__)
        return new

    def __iadd__(self, other):
        new = self.__class__(complex(self) + other)
        new.__dict__.update(**self.__dict__)
        self = new

    def __mul__(self, other):
        new = self.__class__(complex(self) * other)
        new.__dict__.update(**self.__dict__)
        return new

    def __imul__(self, other):
        new = self.__class__(complex(self) * other)
        new.__dict__.update(**self.__dict__)
        self = new


class CustomFloat(float):
    """
    Helper class to inherit from and which stores a float number that is
    extendable.
    """
    def __new__(cls, *args):
        return super(CustomFloat, cls).__new__(cls, *args)

    def __init__(self, *args):
        pass

    def __add__(self, other):
        new = self.__class__(float(self) + other)
        new.__dict__.update(**self.__dict__)
        return new

    def __iadd__(self, other):
        new = self.__class__(float(self) + other)
        new.__dict__.update(**self.__dict__)
        self = new

    def __mul__(self, other):
        new = self.__class__(float(self) * other)
        new.__dict__.update(**self.__dict__)
        return new

    def __imul__(self, other):
        new = self.__class__(float(self) * other)
        new.__dict__.update(**self.__dict__)
        self = new


class FloatWithUncertainties(CustomFloat):
    """
    Helper class to inherit from and which stores a float with a given valid
    range, upper/lower uncertainties and eventual additional attributes.
    """
    _minimum = float("-inf")
    _maximum = float("inf")

    def __new__(cls, value, **kwargs):
        if not cls._minimum <= float(value) <= cls._maximum:
            msg = "value %s out of bounds (%s, %s)"
            msg = msg % (value, cls._minimum, cls._maximum)
            raise ValueError(msg)
        return super(FloatWithUncertainties, cls).__new__(cls, value)

    def __init__(self, value, lower_uncertainty=None, upper_uncertainty=None):
        # set uncertainties, if initialized with similar type
        if isinstance(value, FloatWithUncertainties):
            if lower_uncertainty is None:
                lower_uncertainty = value.lower_uncertainty
            if upper_uncertainty is None:
                upper_uncertainty = value.upper_uncertainty
        # set/override uncertainties, if explicitly specified
        self.lower_uncertainty = lower_uncertainty
        self.upper_uncertainty = upper_uncertainty


class FloatWithUncertaintiesFixedUnit(FloatWithUncertainties):
    """
    Float value that has lower and upper uncertainties and a fixed unit
    associated with it. Helper class to inherit from setting a custom value for
    the fixed unit (set unit in derived class as class attribute).

    :type value: float
    :param value: Actual float value.
    :type lower_uncertainty: float
    :param lower_uncertainty: Lower uncertainty (aka minusError)
    :type upper_uncertainty: float
    :param upper_uncertainty: Upper uncertainty (aka plusError)
    :type unit: str (read only)
    :param unit: Unit for physical interpretation of the float value.
    """
    _unit = ""

    def __init__(self, value, lower_uncertainty=None, upper_uncertainty=None):
        super(FloatWithUncertaintiesFixedUnit, self).__init__(
            value, lower_uncertainty=lower_uncertainty,
            upper_uncertainty=upper_uncertainty)

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        msg = "Unit is fixed for this object class."
        raise ValueError(msg)


class FloatWithUncertaintiesAndUnit(FloatWithUncertainties):
    """
    Float value that has lower and upper uncertainties and a unit associated
    with it.

    :type value: float
    :param value: Actual float value.
    :type lower_uncertainty: float
    :param lower_uncertainty: Lower uncertainty (aka minusError)
    :type upper_uncertainty: float
    :param upper_uncertainty: Upper uncertainty (aka plusError)
    :type unit: str
    :param unit: Unit for physical interpretation of the float value.
    """
    def __init__(self, value, lower_uncertainty=None, upper_uncertainty=None,
                 unit=None):
        super(FloatWithUncertaintiesAndUnit, self).__init__(
            value, lower_uncertainty=lower_uncertainty,
            upper_uncertainty=upper_uncertainty)
        self.unit = unit

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value


class _ComplexUncertainty(complex):
    """
    Complex class which can accept a python None as an argument and map it to
    a float value for storage.
    """
    _none = float("-inf")

    @classmethod
    def _encode(cls, arg):
        if arg is None:
            return cls._none
        return arg

    @classmethod
    def _decode(cls, arg):
        if arg == cls._none:
            return None
        return arg

    def __new__(cls, *args):
        cargs = [cls._encode(a) for a in args]
        if len(args) < 1:
            cargs.append(cls._none)
        if len(args) < 2:
            if args[0] is None:
                cargs.append(cls._none)
            else:
                cargs.append(0)
        return super(_ComplexUncertainty, cls).__new__(cls, *cargs)

    @property
    def real(self):
        _real = super(_ComplexUncertainty, self).real
        return self._decode(_real)

    @property
    def imag(self):
        _imag = super(_ComplexUncertainty, self).imag
        return self._decode(_imag)


class ComplexWithUncertainties(CustomComplex):
    """
    Complex class which can store uncertainties.

    Accepts FloatWithUncertainties and returns FloatWithUncertainties from
    property methods.
    """
    _lower_uncertainty = None
    _upper_uncertainty = None

    @staticmethod
    def _attr(obj, attr):
        try:
            return getattr(obj, attr)
        except AttributeError:
            return None

    @staticmethod
    def _uncertainty(value):
        if isinstance(value, tuple) or isinstance(value, list):
            u = _ComplexUncertainty(*value)
        else:
            u = _ComplexUncertainty(value)
        if u.real is None and u.imag is None:
            return None
        return u

    @property
    def lower_uncertainty(self):
        return self._lower_uncertainty

    @lower_uncertainty.setter
    def lower_uncertainty(self, value):
        self._lower_uncertainty = self._uncertainty(value)

    @property
    def upper_uncertainty(self):
        return self._upper_uncertainty

    @upper_uncertainty.setter
    def upper_uncertainty(self, value):
        self._upper_uncertainty = self._uncertainty(value)

    def __new__(cls, *args, **kwargs):
        return super(ComplexWithUncertainties, cls).__new__(cls, *args)

    def __init__(self, *args, **kwargs):
        """
        Complex type with optional keywords:

        :type lower_uncertainty: complex
        :param lower_uncertainty: Lower uncertainty (aka minusError)
        :type upper_uncertainty: complex
        :param upper_uncertainty: Upper uncertainty (aka plusError)

        """
        real_upper = None
        imag_upper = None
        real_lower = None
        imag_lower = None
        if len(args) >= 1:
            if isinstance(args[0], self.__class__):
                self.upper_uncertainty = args[0].upper_uncertainty
                self.lower_uncertainty = args[0].lower_uncertainty
            elif isinstance(args[0], FloatWithUncertainties):
                real_upper = args[0].upper_uncertainty
                real_lower = args[0].lower_uncertainty
        if len(args) >= 2 and isinstance(args[1], FloatWithUncertainties):
            imag_upper = args[1].upper_uncertainty
            imag_lower = args[1].lower_uncertainty
        if self.upper_uncertainty is None:
            self.upper_uncertainty = real_upper, imag_upper
        if self.lower_uncertainty is None:
            self.lower_uncertainty = real_lower, imag_lower
        if "lower_uncertainty" in kwargs:
            self.lower_uncertainty = kwargs['lower_uncertainty']
        if "upper_uncertainty" in kwargs:
            self.upper_uncertainty = kwargs['upper_uncertainty']

    @property
    def real(self):
        _real = super(ComplexWithUncertainties, self).real
        _lower = self._attr(self.lower_uncertainty, 'real')
        _upper = self._attr(self.upper_uncertainty, 'real')
        return FloatWithUncertainties(_real, lower_uncertainty=_lower,
                                      upper_uncertainty=_upper)

    @property
    def imag(self):
        _imag = super(ComplexWithUncertainties, self).imag
        _lower = self._attr(self.lower_uncertainty, 'imag')
        _upper = self._attr(self.upper_uncertainty, 'imag')
        return FloatWithUncertainties(_imag, lower_uncertainty=_lower,
                                      upper_uncertainty=_upper)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
