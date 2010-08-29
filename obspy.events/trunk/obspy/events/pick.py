# -*- coding: utf-8 -*-

from obspy.core import UTCDateTime, AttribDict
from util import raise_locked_warning


class Pick(AttribDict):
    """
    Class that stores all information about one pick.
    """
    def __init__(self):
        # Status variable that determine whether the AttribDict and its picks can
        # be edited.
        # The base class needs to be called explicitly.
        AttribDict.__setattr__(self, '_Pick__locked', False)

    def __setattr__(self, name, value):
        """
        Attribute access only when the AttribDict is not locked.
        """
        if self.__locked:
            raise_locked_warning()
            return
        # Call the base class and set the value.
        AttribDict.__setattr__(self, name, value)

    def _lock(self):
        """
        Locks the AttribDict.
        """
        self.__locked = True

    def _unlock(self):
        """
        Unlocks the AttribDict.
        """
        # The base class needs to be called explicitly.
        AttribDict.__setattr__(self, '_Pick__locked', False)
