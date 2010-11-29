import os
import tempfile

def NamedTemporaryFile(dir=None, suffix='.tmp'):
    """
    Weak replacement for the Python class :class:`tempfile.NamedTemporaryFile`.

    This class will work also with Windows Vista's UAC.

    Duplicated from obspy.core so that obspy.segy is independant from
    obspy.core.

    .. warning::
        The calling program is responsible to close the returned file pointer
        after usage.
    """

    class NamedTemporaryFile(object):

        def __init__(self, fd, fname):
            self._fileobj = os.fdopen(fd, 'w+b')
            self.name = fname

        def __getattr__(self, attr):
            return getattr(self._fileobj, attr)
    return NamedTemporaryFile(*tempfile.mkstemp(dir=dir, suffix=suffix))
