import unittest

from obspy.core.util import add_doctests, add_unittests
from obspy.core.util.misc import MatplotlibBackend


# this code is needed to run the tests without any X11 or any other
# display, e.g. via a SSH connection. Import it only once, else a nasty
# warning occurs.
# see also: http://matplotlib.org/faq/howto_faq.html
MatplotlibBackend("AGG", sloppy=False)

MODULE_NAME = "obspy.imaging"


def suite():
    suite = unittest.TestSuite()
    add_doctests(suite, MODULE_NAME)
    add_unittests(suite, MODULE_NAME)
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
