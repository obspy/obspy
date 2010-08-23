# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime
from util import raise_locked_warning


class Pick(object):
    """
    Class that stores all information about one pick.
    """
    def __init__(self):
        # Status variable that determine whether the object and its picks can
        # be edited.
        # The base class needs to be called explicitly.
        object.__setattr__(self, '_Pick__locked', False)

    def __setattr__(self, name, value):
        """
        Attribute access only when the object is not locked.
        """
        if self.__locked:
            raise_locked_warning()
            return
        # Call the base class and set the value.
        object.__setattr__(self, name, value)

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
        object.__setattr__(self, '_Pick__locked', False)
