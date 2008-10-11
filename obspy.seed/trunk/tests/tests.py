# -*- coding: utf-8 -*-

import doctest
import unittest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(doctest.DocFileSuite('test_blockette.txt'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')