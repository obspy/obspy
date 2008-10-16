# -*- coding: utf-8 -*-

import unittest
import doctest


class BlocketteTestSuite(unittest.TestSuite):
    
    def __init__(self, *args, **kwargs):
        unittest.TestSuite.__init__(self, *args, **kwargs)
        self.addTest(doctest.DocFileSuite('blockette010.txt'))
        self.addTest(doctest.DocFileSuite('blockette011.txt'))
        self.addTest(doctest.DocFileSuite('blockette012.txt'))
        self.addTest(doctest.DocFileSuite('blockette030.txt'))
        self.addTest(doctest.DocFileSuite('blockette031.txt'))
        self.addTest(doctest.DocFileSuite('blockette033.txt'))
        self.addTest(doctest.DocFileSuite('blockette034.txt'))
        self.addTest(doctest.DocFileSuite('blockette050.txt'))
        self.addTest(doctest.DocFileSuite('blockette051.txt'))
        self.addTest(doctest.DocFileSuite('blockette052.txt'))
        self.addTest(doctest.DocFileSuite('blockette053.txt'))
        self.addTest(doctest.DocFileSuite('blockette054.txt'))
        self.addTest(doctest.DocFileSuite('blockette057.txt'))
        self.addTest(doctest.DocFileSuite('blockette058.txt'))
        self.addTest(doctest.DocFileSuite('blockette061.txt'))


def suite():
    return BlocketteTestSuite()


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
