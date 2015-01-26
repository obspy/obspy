#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import inspect
import os
import sys


ROOT = os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe())))


def _get_model_filename(model_name):
    """
    Get the pickled filename of a model. Depends on the Python version.

    :param model_name: The model name.
    """
    model_dir = os.path.join(ROOT, "data", "models")
    model_name = os.path.splitext(os.path.basename(model_name))[0]
    filename = os.path.join(
        model_dir, model_name +
        ("__py%i%i__tvel" % sys.version_info[:2]) + os.path.extsep + "pickle")
    return filename


def parse_phase_list(phase_list):
    """
    Takes a list of phases, returns a list of individual phases. Performs e.g.
    replacing e.g. ``"ttall"`` with the relevant phases.
    """
    phase_names = []
    for phaseName in phase_list:
        phase_names += get_phase_names(phaseName)
    # Remove duplicates.
    return list(set(phase_names))


def get_phase_names(phase_name):
    """
    Called by parse_phase_list to replace e.g. ttall with the relevant phases.
    """
    names = []
    if(phase_name.lower() == "ttp"
       or phase_name.lower() == "tts"
       or phase_name.lower() == "ttbasic"
       or phase_name.lower() == "tts+"
       or phase_name.lower() == "ttp+"
       or phase_name.lower() == "ttall"):
        if(phase_name.lower() == "ttp"
           or phase_name.lower() == "ttp+"
           or phase_name.lower() == "ttbasic"
           or phase_name.lower() == "ttall"):
            names.append("p")
            names.append("P")
            names.append("Pn")
            names.append("Pdiff")
            names.append("PKP")
            names.append("PKiKP")
            names.append("PKIKP")
        if(phase_name.lower() == "tts"
           or phase_name.lower() == "tts+"
           or phase_name.lower() == "ttbasic"
           or phase_name.lower() == "ttall"):
            names.append("s")
            names.append("S")
            names.append("Sn")
            names.append("Sdiff")
            names.append("SKS")
            names.append("SKIKS")
        if(phase_name.lower() == "ttp+"
           or phase_name.lower() == "ttbasic"
           or phase_name.lower() == "ttall"):
            names.append("PcP")
            names.append("pP")
            names.append("pPdiff")
            names.append("pPKP")
            names.append("pPKIKP")
            names.append("pPKiKP")
            names.append("sP")
            names.append("sPdiff")
            names.append("sPKP")
            names.append("sPKIKP")
            names.append("sPKiKP")
        if(phase_name.lower() == "tts+"
           or phase_name.lower() == "ttbasic"
           or phase_name.lower() == "ttall"):
            names.append("sS")
            names.append("sSdiff")
            names.append("sSKS")
            names.append("sSKIKS")
            names.append("ScS")
            names.append("pS")
            names.append("pSdiff")
            names.append("pSKS")
            names.append("pSKIKS")
        if(phase_name.lower() == "ttbasic"
           or phase_name.lower() == "ttall"):
            names.append("ScP")
            names.append("SKP")
            names.append("SKIKP")
            names.append("PKKP")
            names.append("PKIKKIKP")
            names.append("SKKP")
            names.append("SKIKKIKP")
            names.append("PP")
            names.append("PKPPKP")
            names.append("PKIKPPKIKP")
        if phase_name.lower() == "ttall":
            names.append("SKiKP")
            names.append("PP")
            names.append("ScS")
            names.append("PcS")
            names.append("PKS")
            names.append("PKIKS")
            names.append("PKKS")
            names.append("PKIKKIKS")
            names.append("SKKS")
            names.append("SKIKKIKS")
            names.append("SKSSKS")
            names.append("SKIKSSKIKS")
            names.append("SS")
            names.append("SP")
            names.append("PS")
    else:
        names.append(phase_name)
    return names
