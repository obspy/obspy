# -*- coding: utf-8 -*-
"""
The obspy.realtime.rttrace test suite.
"""
from obspy.core import Trace
from obspy.realtime import RtTrace
import unittest
import numpy as np
from obspy.realtime.rtmemory import RtMemory


class RtTraceTestCase(unittest.TestCase):

    def test_eq(self):
        """
        Testing __eq__ method.
        """
        tr = Trace()
        tr2 = RtTrace()
        tr3 = RtTrace()
        # RtTrace should never be equal with Trace objects
        self.assertFalse(tr2 == tr)
        self.assertFalse(tr2.__eq__(tr))
        self.assertTrue(tr2 == tr3)
        self.assertTrue(tr2.__eq__(tr3))

    def test_ne(self):
        """
        Testing __ne__ method.
        """
        tr = Trace()
        tr2 = RtTrace()
        tr3 = RtTrace()
        # RtTrace should never be equal with Trace objects
        self.assertTrue(tr2 != tr)
        self.assertTrue(tr2.__ne__(tr))
        self.assertFalse(tr2 != tr3)
        self.assertFalse(tr2.__ne__(tr3))

    def test_registerRtProcess(self):
        """
        Testing registerRtProcess method.
        """
        tr = RtTrace()
        # 1 - function call
        tr.registerRtProcess(np.abs)
        self.assertEqual(tr.processing, [(np.abs, {}, None)])
        # 2 - predefined RT processing algorithm
        tr.registerRtProcess('integrate', test=1, muh='maeh')
        self.assertEqual(tr.processing[1][0], 'integrate')
        self.assertEqual(tr.processing[1][1], {'test': 1, 'muh': 'maeh'})
        self.assertTrue(isinstance(tr.processing[1][2][0], RtMemory))
        # 3 - contained name of predefined RT processing algorithm
        tr.registerRtProcess('in')
        self.assertEqual(tr.processing[2][0], 'integrate')
        tr.registerRtProcess('integ')
        self.assertEqual(tr.processing[3][0], 'integrate')
        tr.registerRtProcess('integr')
        self.assertEqual(tr.processing[4][0], 'integrate')
        # 4 - unknown functions
        self.assertRaises(NotImplementedError, tr.registerRtProcess, 'xyz')
        # 5 - module instead of function
        self.assertRaises(NotImplementedError, tr.registerRtProcess, np)
        # check number off all processing steps within RtTrace
        self.assertEqual(len(tr.processing), 5)
        # check tr.stats.processing
        self.assertEqual(len(tr.stats.processing), 5)
        self.assertTrue(tr.stats.processing[0].startswith("realtime_process"))
        self.assertTrue('absolute' in tr.stats.processing[0])
        for i in range(1, 5):
            self.assertTrue('integrate' in tr.stats.processing[i])
        # check kwargs
        self.assertTrue("maeh" in tr.stats.processing[1])


def suite():
    return unittest.makeSuite(RtTraceTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
