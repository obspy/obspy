# -*- coding: utf-8 -*-

import doctest
import os
import unittest


def suite():
    suite = unittest.TestSuite()
    docpath = 'doctests' + os.path.sep
    suite.addTest(doctest.DocFileSuite(docpath + 'test.txt'))
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
