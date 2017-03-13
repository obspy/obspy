# -*- coding: utf-8 -*-

import unittest


def load_tests(loader, tests, pattern):  # @UnusedVariable
    return loader.discover('.')


if __name__ == '__main__':
    unittest.main()
