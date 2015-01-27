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
    lphase = phase_name.lower()
    names = []
    if lphase in ("ttp", "tts", "ttbasic", "tts+", "ttp+", "ttall"):
        if lphase in ("ttp", "ttp+", "ttbasic", "ttall"):
            names.extend(["p", "P", "Pn", "Pdiff", "PKP", "PKiKP", "PKIKP"])

        if lphase in ("tts", "tts+", "ttbasic", "ttall"):
            names.extend(["s", "S", "Sn", "Sdiff", "SKS", "SKIKS"])

        if lphase in ("ttp+", "ttbasic", "ttall"):
            names.extend(["PcP", "pP", "pPdiff", "pPKP", "pPKIKP", "pPKiKP",
                          "sP", "sPdiff", "sPKP", "sPKIKP", "sPKiKP"])

        if lphase in ("tts+", "ttbasic", "ttall"):
            names.extend(["sS", "sSdiff", "sSKS", "sSKIKS", "ScS", "pS",
                          "pSdiff", "pSKS", "pSKIKS"])

        if lphase in ("ttbasic", "ttall"):
            names.extend(["ScP", "SKP", "SKIKP", "PKKP", "PKIKKIKP", "SKKP",
                          "SKIKKIKP", "PP", "PKPPKP", "PKIKPPKIKP"])

        if lphase == "ttall":
            names.extend(["SKiKP", "PP", "ScS", "PcS", "PKS", "PKIKS", "PKKS",
                          "PKIKKIKS", "SKKS", "SKIKKIKS", "SKSSKS",
                          "SKIKSSKIKS", "SS", "SP", "PS"])

    else:
        names.append(phase_name)

    return names
