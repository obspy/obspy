# -*- coding: utf-8 -*-

import doctest
import unittest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocFileSuite('test_blockette010.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette011.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette030.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette033.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette034.txt'))
    suite.addTest(doctest.DocFileSuite('test_blockette050.txt'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')