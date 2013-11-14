# -*- coding: utf-8 -*-

import warnings
import unittest
from obspy.core.util import add_doctests, add_unittests


MODULE_NAME = "obspy.signal"


def suite():
    suite = unittest.TestSuite()
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings(
            'ignore', 'Module obspy.signal.psd is deprecated! '
            'Use obspy.signal.spectral_estimation instead or import directly '
            '"from obspy.signal import ...".', category=DeprecationWarning)
        add_doctests(suite, MODULE_NAME)
        add_unittests(suite, MODULE_NAME)
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
