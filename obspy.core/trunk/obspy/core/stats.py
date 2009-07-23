# -*- coding: utf-8 -*-


class Stats(dict, object):
    """
    A stats class which behaves like a dictionary.
    
    You may the following syntax to change or access data in this class:
      >>> stats = Stats()
      >>> stats.network = 'BW'
      >>> stats['station'] = 'ROTZ'
      >>> stats.get('network')
      'BW'
      >>> stats['network']
      'BW'
      >>> stats.station
      'ROTZ'
      >>> x = stats.keys()
      >>> x.sort()
      >>> x[0:3]
      ['network', 'station']
    """

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

    def __setitem__(self, key, value):
        super(Stats, self).__setattr__(key, value)
        return super(Stats, self).__setitem__(key, value)

    def __getitem__(self, name):
        return super(Stats, self).__getitem__(name)

    def __delitem__(self, name):
        super(Stats, self).__delattr__(name)
        return super(Stats, self).__delitem__(name)

    __getattr__ = __getitem__
    __setattr__ = __setitem__
    __delattr__ = __delitem__

    def copy(self, init={}):
        return Stats(init)

    def __deepcopy__(self, *args, **kwargs):
        st = Stats()
        st.update(self)
        return st


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
