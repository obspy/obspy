# -*- coding: utf-8 -*-

import unittest, obspy, time

def suite():
    """
    The obspy test suite.
    """
    suite = unittest.TestSuite()
    for module in ['gse2','mseed','sac','picker','arclink','xseed','filter']:
        try:
            __import__('obspy.%s.tests' % module)
            exec "suite.addTests(obspy.%s.tests.suite())" % module
        except:
            print "Cannot import test suite of module obspy.%s" % module
            time.sleep(0.5)
            continue
    return suite

def runTests():
    """
    This function runs all available tests in obspy
    """
    unittest.TextTestRunner(verbosity=2).run(suite())
    #unittest.main(defaultTest='suite')

if __name__ == '__main__':
    runTests()
