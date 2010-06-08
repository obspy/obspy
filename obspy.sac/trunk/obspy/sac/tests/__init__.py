# -*- coding: utf-8 -*-

from obspy.sac import sacio
from obspy.sac.tests import test_sacio, test_core
import unittest, doctest
import shutil
import os
from obspy.core.util import NamedTemporaryFile
tempdir = os.path.dirname(NamedTemporaryFile().name)
curdir = os.getcwd()
fn1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data','test.sac')
fn2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data','testxy.sac')

def setUp(dummy):
    os.chdir(tempdir)
    shutil.copy(fn1,os.getcwd())
    shutil.copy(fn2,os.getcwd())
    
def tearDown(dummy):
    os.chdir(curdir)
    fn1_loc = os.path.join(tempdir,'test.sac')
    fn2_loc = os.path.join(tempdir,'testxy.sac')
    fn3_loc = os.path.join(tempdir,'test2.sac')
    if os.path.isfile(fn1_loc) and fn1_loc != fn1:
        os.remove(fn1_loc)
    if os.path.isfile(fn2_loc) and fn2_loc != fn2:
        os.remove(fn2_loc)
    if os.path.isfile(fn3_loc):
        os.remove(fn3_loc)
    
def suite():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocTestSuite(sacio,setUp=setUp,tearDown=tearDown))
    suite.addTest(test_sacio.suite())
    suite.addTest(test_core.suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
