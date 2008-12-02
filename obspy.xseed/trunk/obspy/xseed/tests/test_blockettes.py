# -*- coding: utf-8 -*-

import unittest
import doctest
import os


def additional_tests():
    suite = unittest.TestSuite()
    docpath = 'doctests' + os.path.sep
    suite.addTest(doctest.DocFileSuite(docpath + 'blockette010.txt'))
    suite.addTest(doctest.DocFileSuite(docpath + 'blockette011.txt'))
    suite.addTest(doctest.DocFileSuite(docpath + 'blockette012.txt'))
    suite.addTest(doctest.DocFileSuite(docpath + 'blockette030.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette031.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette032.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette033.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette034.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette050.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette051.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette052.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette053.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette054.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette057.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette058.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette059.txt'))
#    suite.addTest(doctest.DocFileSuite(docpath + 'blockette061.txt'))
    return suite


def suite():
    return additional_tests()


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
