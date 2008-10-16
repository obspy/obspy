# -*- coding: utf-8 -*-

import unittest
import doctest

def additional_tests():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocFileSuite('blockette010.txt'))
    suite.addTest(doctest.DocFileSuite('blockette011.txt'))
    suite.addTest(doctest.DocFileSuite('blockette012.txt'))
    suite.addTest(doctest.DocFileSuite('blockette030.txt'))
    suite.addTest(doctest.DocFileSuite('blockette031.txt'))
    suite.addTest(doctest.DocFileSuite('blockette033.txt'))
    suite.addTest(doctest.DocFileSuite('blockette034.txt'))
    suite.addTest(doctest.DocFileSuite('blockette050.txt'))
    suite.addTest(doctest.DocFileSuite('blockette051.txt'))
    suite.addTest(doctest.DocFileSuite('blockette052.txt'))
    suite.addTest(doctest.DocFileSuite('blockette053.txt'))
    suite.addTest(doctest.DocFileSuite('blockette054.txt'))
    suite.addTest(doctest.DocFileSuite('blockette057.txt'))
    suite.addTest(doctest.DocFileSuite('blockette058.txt'))
    suite.addTest(doctest.DocFileSuite('blockette061.txt'))
    return suite


def suite():
    return additional_tests()


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
