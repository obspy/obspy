#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The tf_misfit test suite.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import os
import unittest

import numpy as np
from scipy.signal import hilbert

from obspy.signal.tf_misfit import (eg, em, feg, fem, fpg, fpm, pg, pm, teg,
                                    tem, tfeg, tfem, tfpg, tfpm, tpg, tpm)


class TfTestCase(unittest.TestCase):
    """
    Test cases for array_analysis functions.
    """
    def setUp(self):
        # path to test files
        self.path = os.path.join(os.path.dirname(__file__), 'data')
        tmax = 3.
        npts = 60
        dt = tmax / (npts - 1)

        fmin = 1.
        fmax = 3.
        nf = 3

        # Constants for S1
        A1 = 4.
        t1 = .1
        f1 = 2.
        phi1 = 0.

        # Constants for S1t and S1a
        ps = 0.1
        A1a = A1 * 1.1

        t = np.linspace(0., tmax, npts)
        f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

        def H(t):
            return (np.sign(t) + 1) / 2

        def S1(t):
            return A1 * (t - t1) * np.exp(-2 * (t - t1)) * \
                np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * H(t - t1)

        # generate analytical signal (hilbert transform) and add phase shift
        s1h = hilbert(S1(t))
        s1p = np.real(
            np.abs(s1h) * np.exp(np.angle(s1h) * 1j + ps * np.pi * 1j))

        # signal with amplitude error
        def S1a(t):
            return A1a * (t - t1) * np.exp(-2 * (t - t1)) * \
                np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * H(t - t1)

        self.S1 = S1
        self.s1p = s1p
        self.S1a = S1a
        self.t = t
        self.f = f
        self.dt = dt

        self.fmin = fmin
        self.fmax = fmax
        self.nf = nf
        self.npts = npts
        self.w0 = 6

    def test_phase_misfit(self):
        """
        Tests all tf misfits with a signal that has phase misfit
        """
        S1 = self.S1
        s1p = self.s1p
        t = self.t
        dt = self.dt

        fmin = self.fmin
        fmax = self.fmax
        nf = self.nf

        TFEM_11p_ref = np.loadtxt(self.path + os.sep + 'TFEM_11p.dat')
        TFPM_11p_ref = np.loadtxt(self.path + os.sep + 'TFPM_11p.dat')
        TEM_11p_ref = np.loadtxt(self.path + os.sep + 'TEM_11p.dat')
        FEM_11p_ref = np.loadtxt(self.path + os.sep + 'FEM_11p.dat')
        FPM_11p_ref = np.loadtxt(self.path + os.sep + 'FPM_11p.dat')
        TPM_11p_ref = np.loadtxt(self.path + os.sep + 'TPM_11p.dat')
        EM_11p_ref = np.loadtxt(self.path + os.sep + 'EM_11p.dat')
        PM_11p_ref = np.loadtxt(self.path + os.sep + 'PM_11p.dat')

        TFEM_11p = tfem(s1p, S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TFPM_11p = tfpm(s1p, S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TEM_11p = tem(s1p, S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        FEM_11p = fem(s1p, S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        FPM_11p = fpm(s1p, S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TPM_11p = tpm(s1p, S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        EM_11p = em(s1p, S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        PM_11p = pm(s1p, S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)

        tol = 1e-5
        atol_min = 1e-15

        self.assertTrue(np.allclose(TFEM_11p, TFEM_11p_ref, rtol=tol,
                        atol=np.abs(TFEM_11p_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(TFPM_11p, TFPM_11p_ref, rtol=tol,
                        atol=np.abs(TFPM_11p_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(TEM_11p, TEM_11p_ref, rtol=tol,
                        atol=np.abs(TEM_11p_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(FEM_11p, FEM_11p_ref, rtol=tol,
                        atol=np.abs(FEM_11p_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(FPM_11p, FPM_11p_ref, rtol=tol,
                        atol=np.abs(FPM_11p_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(TPM_11p, TPM_11p_ref, rtol=tol,
                        atol=np.abs(TPM_11p_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(EM_11p, EM_11p_ref, rtol=tol,
                        atol=np.abs(EM_11p_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(PM_11p, PM_11p_ref, rtol=tol,
                        atol=np.abs(PM_11p_ref).max() * tol + atol_min))

        # keeping the save commands in case the files need to be updated
        # np.savetxt(self.path + os.sep + 'TFEM_11p.dat', TFEM_11p,
        #            fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'TFPM_11p.dat', TFPM_11p,
        #            fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'TEM_11p.dat', TEM_11p, fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'FEM_11p.dat', FEM_11p, fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'FPM_11p.dat', FPM_11p, fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'TPM_11p.dat', TPM_11p, fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'EM_11p.dat', (EM_11p,), fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'PM_11p.dat', (PM_11p,), fmt='%1.5e')

    def test_envelope_misfit(self):
        """
        Tests all tf misfits with a signal that has envelope misfit
        """
        S1 = self.S1
        S1a = self.S1a
        t = self.t
        dt = self.dt

        fmin = self.fmin
        fmax = self.fmax
        nf = self.nf

        TFEM_11a_ref = np.loadtxt(self.path + os.sep + 'TFEM_11a.dat')
        TFPM_11a_ref = np.loadtxt(self.path + os.sep + 'TFPM_11a.dat')
        TEM_11a_ref = np.loadtxt(self.path + os.sep + 'TEM_11a.dat')
        FEM_11a_ref = np.loadtxt(self.path + os.sep + 'FEM_11a.dat')
        FPM_11a_ref = np.loadtxt(self.path + os.sep + 'FPM_11a.dat')
        TPM_11a_ref = np.loadtxt(self.path + os.sep + 'TPM_11a.dat')
        EM_11a_ref = np.loadtxt(self.path + os.sep + 'EM_11a.dat')
        PM_11a_ref = np.loadtxt(self.path + os.sep + 'PM_11a.dat')

        TFEM_11a = tfem(S1a(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TFPM_11a = tfpm(S1a(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TEM_11a = tem(S1a(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TPM_11a = tpm(S1a(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        FEM_11a = fem(S1a(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        FPM_11a = fpm(S1a(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        EM_11a = em(S1a(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        PM_11a = pm(S1a(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)

        tol = 1e-5
        atol_min = 1e-15

        self.assertTrue(np.allclose(TFEM_11a, TFEM_11a_ref, rtol=tol,
                        atol=np.abs(TFEM_11a_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(TFPM_11a, TFPM_11a_ref, rtol=tol,
                        atol=np.abs(TFPM_11a_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(TEM_11a, TEM_11a_ref, rtol=tol,
                        atol=np.abs(TEM_11a_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(FEM_11a, FEM_11a_ref, rtol=tol,
                        atol=np.abs(FEM_11a_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(FPM_11a, FPM_11a_ref, rtol=tol,
                        atol=np.abs(FPM_11a_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(TPM_11a, TPM_11a_ref, rtol=tol,
                        atol=np.abs(TPM_11a_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(EM_11a, EM_11a_ref, rtol=tol,
                        atol=np.abs(EM_11a_ref).max() * tol + atol_min))
        self.assertTrue(np.allclose(PM_11a, PM_11a_ref, rtol=tol,
                        atol=np.abs(PM_11a_ref).max() * tol + atol_min))

        # keeping the save commands in case the files need to be updated
        # np.savetxt(self.path + os.sep + 'TFEM_11a.dat', TFEM_11a,
        #            fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'TFPM_11a.dat', TFPM_11a,
        #            fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'TEM_11a.dat', TEM_11a, fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'FEM_11a.dat', FEM_11a, fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'FPM_11a.dat', FPM_11a, fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'TPM_11a.dat', TPM_11a, fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'EM_11a.dat', (EM_11a,), fmt='%1.5e')
        # np.savetxt(self.path + os.sep + 'PM_11a.dat', (PM_11a,), fmt='%1.5e')

    def test_envelope_gof(self):
        """
        Tests all tf gofs
        """
        S1 = self.S1
        t = self.t
        dt = self.dt

        fmin = self.fmin
        fmax = self.fmax
        nf = self.nf
        npts = self.npts

        tol = 1e-5

        TFEG = tfeg(S1(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TFPG = tfpg(S1(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TEG = teg(S1(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TPG = tpg(S1(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        FEG = feg(S1(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        FPG = fpg(S1(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        EG = eg(S1(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        PG = pg(S1(t), S1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)

        self.assertTrue(np.allclose(TFEG, np.ones((nf, npts)) * 10., rtol=tol))
        self.assertTrue(np.allclose(TFPG, np.ones((nf, npts)) * 10., rtol=tol))
        self.assertTrue(np.allclose(TEG, np.ones(npts) * 10., rtol=tol))
        self.assertTrue(np.allclose(TPG, np.ones(npts) * 10., rtol=tol))
        self.assertTrue(np.allclose(FEG, np.ones(nf) * 10., rtol=tol))
        self.assertTrue(np.allclose(FPG, np.ones(nf) * 10., rtol=tol))
        self.assertTrue(np.allclose(EG, 10., rtol=tol))
        self.assertTrue(np.allclose(PG, 10., rtol=tol))


def suite():
    return unittest.makeSuite(TfTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
