#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
from unittest import TestCase


class TestTauPyModel(TestCase):
    def test_create_taup_model(self):
        """
        See if the create model function in the tau interface runs smoothly.
        """
        from obspy.taup import tau
        try:
            os.remove("ak135.taup")
        except FileNotFoundError:
            pass
        tau.TauPyModel("ak135", taup_model_path=".")
        os.remove("ak135.taup")
