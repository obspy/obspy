#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import os
import unittest

from taupy.TauP_Create import TauP_Create

# to get ./data:
data_dir = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


class TestTauPCreate(unittest.TestCase):
    def test_taupcreate(self):
        TauP_Create.main(modelFilename='iasp91.tvel')


if __name__ == '__main__':
    unittest.main(buffer=True)
