# -*- coding: utf-8 -*-

import warnings

def raise_locked_warning():
    warnings.warn('Object is locked. Attributes cannot be changed.')

class UniqueList(list):
    """
    A list that will not allow duplicate entries. The list can also be locked
    to deny any further changes.

    XXX: Still a lot of functions missing. Only the most basic ones are used so
    far.
    """
    def __init__(self, *args):
        """
        Make sure no duplicate entries are added at the init method.
        """
        length = len(args)
        if not length:
            list.__init__(self)
        else:
            # Call list. It will raise the correct error!
            if length != 1:
                list.__init__(self, *args)
            items = list(args[0])
            new_items = []
            for item in items:
                if item not in new_items:
                    new_items.append(item)
            list.__init__(self, new_items)
        # Set locked attribute to False.
        self.__locked = False

    def __setslice__(self, i, j, sequence):
        """
        Modify setslice. In order to not produce any duplicates it will only be
        allowed if none of the items in sequence is anywhere outside of [i:j].
        """
        if self.__locked:
            raise_locked_warning()
            return
        # Check if any item in sequence is in any part outside of the slice to
        # be replaced.
        out_of_sequence = []
        if i > 0:
            out_of_sequence.extend(self[:i])
        if j < len(self) + 1:
            out_of_sequence.extend(self[j:])
        if out_of_sequence:
            for item in sequence:
                if item in out_of_sequence:
                    return
        # Check for duplicates in sequence.
        new_sequence = []
        for item in sequence:
            if item in new_sequence:
                return
            new_sequence.append(item)
        # Call the parent method.
        list.__setslice__(self, i, j, sequence)

    def __setitem__(self, key, value):
        """
        Setitem only works if the item is not in the other entries.
        """
        if self.__locked:
            raise_locked_warning()
            return
        # Lengthy if to assure 
        if type(key) == int and key >= 0 and key < len(self):
            pass
        if value not in self:
            list.__setitem__(self, key, value)

    def __repr__(self):
        return 'UniqueList(%s)' % list.__repr__(self)

    def append(self, item):
        """
        The append method.
        """
        if self.__locked:
            raise_locked_warning()
            return
        if item not in self:
            list.append(self, item)

    def extend(self, items):
        """
        The extend method.
        """
        if self.__locked:
            raise_locked_warning()
            return
        non_duplicate_items = []
        # Item needs to be iterable.
        try:
            iter(items)
        except:
            msg = 'object is not iterable'
            raise TypeError(msg)
        for item in items:
            if item not in self:
                non_duplicate_items.append(item)
        list.extend(self, non_duplicate_items)

    def count(self, value):
        """
        Useless here.
        XXX: Can the method be removed from the inherited class?
        """
        if value in self:
            return 1
        return 0

    def _lock(self):
        """
        Locks the object.
        """
        self.__locked = True

    def _unlock(self):
        """
        Unlocks the object.
        """
        # The base class needs to be called explicitly.
        object.__setattr__(self, '__locked', False)

def str_or_None(object):
    """
    Returns str(object) or None if object is None.
    This is slighty different from str() which returns "None" in this case.

    This could also be done in one line directly in the code via...
    (object is None) and None or str(object)
    ...but this would maybe deteriorate readability.

    We use it in the code in a similar manner as dict.get() which returns a
    value or None.
    """
    if object is None:
        return None
    return str(object)


def set_attribute(obj, etree, key, path, fct=None, do_pass=False):
    """
    Set attribute of obj with first value found in path of etree. Optionally
    run function fct on found value before assignment.
    If do_pass do pass.
    """
    try:
        value = etree.xpath(path)[0].text
        if value is None:
            return
        if fct and value is not None:
            value = fct(value)
    except IndexError:
        if do_pass:
            return
        else:
            value = None
    setattr(obj, key, value)
