# -*- coding: utf-8 -*-
"""
AttribDict class for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import warnings


class AttribDict(dict, object):
    """
    A class which behaves like a dictionary.

    :type data: dict, optional
    :param data: Dictionary with initial keywords.

    .. rubric:: Basic Usage

    You may use the following syntax to change or access data in this class.

    >>> stats = AttribDict()
    >>> stats.network = 'BW'
    >>> stats['station'] = 'ROTZ'
    >>> stats.get('network')
    'BW'
    >>> stats['network']
    'BW'
    >>> stats.station
    'ROTZ'
    >>> x = stats.keys()
    >>> x = sorted(x)
    >>> x[0:3]
    ['network', 'station']
    """
    readonly = []
    priorized_keys = []

    def __init__(self, *args, **kwargs):
        """
        An AttribDict can be initialized in two ways. It can be given an
        existing dictionary as a simple argument or alternatively all keyword
        arguments will become (key, value) pairs.

        >>> attrib_dict_1 = AttribDict({"a":1, "b":2})
        >>> attrib_dict_2 = AttribDict(a=1, b=2)
        >>> print attrib_dict_1
        AttribDict({'a': 1, 'b': 2})
        >>> assert(attrib_dict_1 == attrib_dict_2)
        """
        # Deprecated support of the data={} kwarg.
        if kwargs.get("data") is not None and \
                isinstance(kwargs["data"], dict):
            kwargs.update(kwargs["data"])
            del kwargs["data"]
            msg = "The 'data' kwarg will be deprecated soon. Please use " + \
                  "either AttribDict(data_dict) or pass the kwargs directly."
            warnings.warn(msg, category=DeprecationWarning)
        # Args is allowed to be exactly one dictionary which will then be
        # appended to the kwarg dictionary.
        if len(args) == 1 and isinstance(args[0], dict):
            kwargs.update(args[0])
        dict.__init__(kwargs)
        self.update(kwargs)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

    def __setitem__(self, key, value):
        super(AttribDict, self).__setattr__(key, value)
        super(AttribDict, self).__setitem__(key, value)

    def __getitem__(self, name):
        if name in self.readonly:
            return self.__dict__[name]
        return super(AttribDict, self).__getitem__(name)

    def __delitem__(self, name):
        super(AttribDict, self).__delattr__(name)
        return super(AttribDict, self).__delitem__(name)

    def clear(self):
        self.__dict__ = {}
        return super(AttribDict, self).clear()

    def pop(self, name, default={}):
        value = super(AttribDict, self).pop(name, default)
        del self.__dict__[name]
        return value

    def popitem(self):
        (name, value) = super(AttribDict, self).popitem()
        super(AttribDict, self).__delattr__(name)
        return (name, value)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, pickle_dict):
        self.update(pickle_dict)

    __getattr__ = __getitem__
    __setattr__ = __setitem__
    __delattr__ = __delitem__

    def copy(self):
        return self.__class__(self.__dict__.copy())

    def __deepcopy__(self, *args, **kwargs):  # @UnusedVariable
        st = self.__class__()
        st.update(self)
        return st

    def update(self, adict={}):
        for (key, value) in adict.iteritems():
            if key in self.readonly:
                continue
            self[key] = value

    def setdefault(self, key, value=None):
        """
        If key is in the dictionary, return its value. If not, insert key with
        a value of default and return default. Default defaults to None.
        """
        try:
            return self.__dict__[key]
        except KeyError:
            super(AttribDict, self).__setattr__(key, value)
            super(AttribDict, self).__setitem__(key, value)
        return value

    def _pretty_str(self, priorized_keys=[], min_label_length=16):
        """
        Return better readable string representation of AttribDict object.

        :type priorized_keys: List of str, optional
        :param priorized_keys: Keywords of current AttribtDict which will be
            shown before all other keywords. Those keywords must exists
            otherwise an exception will be raised. Defaults to empty list.
        :type min_label_length: int, optional
        :param min_label_length: Minimum label length for keywords. Defaults
            to ``16``.
        :return: String representation of current AttribDict object.
        """
        keys = self.keys()
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


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
