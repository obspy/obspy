# -*- coding: utf-8 -*-


def isQ(filename):
    """
    Checks whether a file is Q or not. Returns True or False.

    Parameters
    ----------

    filename : string
        Name of the Q file to be read.
    """
    return False


def readQ(filename, headonly=False):
    """
    Reads a Q file and returns an ObsPy Stream object.

    Parameters
    ----------

    filename : string
        Q file to be read.
    headonly : bool, optional
        If set to True, read only the head. This is most useful for
        scanning available data in huge (temporary) data sets.

    Returns
    -------
    stream : :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.
    """
    raise NotImplementedError


def writeQ(stream, filename):
    """
    Writes a Q file from given ObsPy Stream object.

    Parameters
    ----------

    stream : :class:`~obspy.core.stream.Stream`
        A ObsPy Stream object.
    filename : string
        Name of Q file to be written.
    """
    raise NotImplementedError
