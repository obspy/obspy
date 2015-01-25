#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *
from unittest import TestCase

__author__ = 'nicolas'
import os


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
        ak135 = tau.TauPyModel("ak135", taup_model_path=".")
        os.remove("ak135.taup")
