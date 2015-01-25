#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *

import inspect
import os
import unittest

from taupy.TauP_Create import TauP_Create

# to get ./data:
#data_dir = os.path.join(os.path.dirname(os.path.abspath(
#    inspect.getfile(inspect.currentframe()))), "data")


class TestTauPCreate(unittest.TestCase):
    def test_taupcreate(self):
        """
        Simple test for TauP_Create only tests if executed without error.
        """
        #TauP_Create.main(modelFilename='iasp91.tvel')
        pass
        # This is tested in test_tauPyModel, so commentd out here to save time.


if __name__ == '__main__':
    unittest.main(buffer=True)
