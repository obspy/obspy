#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The tf_misfit test suite.
"""

import numpy as np
from scipy.signal import hilbert
from tf_misfit import *
import unittest

class TfTestCase(unittest.TestCase):
    """
    Test cases for array_analysis functions.
    """
    def setUp(self):
        tmax = 3.
        dt = 0.1
        npts = int(tmax / dt + 1)

        fmin = 1.
        fmax = 3.
        nf = 4.

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

        H = lambda t: (np.sign(t) + 1)/ 2 

        S1 = lambda t: A1 * (t - t1) * np.exp(-2*(t - t1)) * np.cos(2. * np.pi * f1 * \
                       (t - t1) + phi1 * np.pi) * H(t - t1)

        # generate analytical signal (hilbert transform) and add phase shift
        s1h = hilbert(S1(t))
        s1p = np.real(np.abs(s1h) * np.exp(np.angle(s1h) * 1j + ps * np.pi * 1j))

        # signal with amplitude error
        S1a = lambda t: A1a * (t - t1) * np.exp(-2*(t - t1)) * np.cos(2. * np.pi * f1 * \
                        (t - t1) + phi1 * np.pi) * H(t - t1)

        self.S1 = S1
        self.s1p = s1p
        self.S1a = S1a
        self.t = t
        self.f = f
        self.dt = dt

        self.fmin = fmin
        self.fmax = fmax
        self.nf = nf
        self.w0 = 6


    def tearDown(self):
        pass

    def test_phase_misfit(self):
        # tests all tf misfits with a signal that has phase misfit
        S1 = self.S1
        s1p = self.s1p
        t = self.t
        dt = self.dt
        f = self.f
        
        fmin = self.fmin 
        fmax = self.fmax 
        nf = self.nf
        w0 = self.w0

        TFEM_11p_ref = np.loadtxt('test_tf_misfit_data/TFEM_11p.dat')
        TFPM_11p_ref = np.loadtxt('test_tf_misfit_data/TFPM_11p.dat')
        TEM_11p_ref = np.loadtxt('test_tf_misfit_data/TEM_11p.dat')
        FEM_11p_ref = np.loadtxt('test_tf_misfit_data/FEM_11p.dat')
        FPM_11p_ref = np.loadtxt('test_tf_misfit_data/FPM_11p.dat')
        TPM_11p_ref = np.loadtxt('test_tf_misfit_data/TPM_11p.dat')
        EM_11p_ref = np.loadtxt('test_tf_misfit_data/EM_11p.dat')
        PM_11p_ref = np.loadtxt('test_tf_misfit_data/PM_11p.dat')

        TFEM_11p = tfem(S1(t), s1p, dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TFPM_11p = tfpm(S1(t), s1p, dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TEM_11p = tem(S1(t), s1p, dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        FEM_11p = fem(S1(t), s1p, dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        FPM_11p = fpm(S1(t), s1p, dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TPM_11p = tpm(S1(t), s1p, dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        EM_11p = em(S1(t), s1p, dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        PM_11p = pm(S1(t), s1p, dt=dt, fmin=fmin, fmax=fmax, nf=nf)

        np.testing.assert_array_almost_equal(TFEM_11p, TFEM_11p_ref, decimal=5)
        np.testing.assert_array_almost_equal(TFPM_11p, TFPM_11p_ref, decimal=5)
        np.testing.assert_array_almost_equal(TEM_11p, TEM_11p_ref, decimal=5)
        np.testing.assert_array_almost_equal(FEM_11p, FEM_11p_ref, decimal=5)
        np.testing.assert_array_almost_equal(FPM_11p, FPM_11p_ref, decimal=5)
        np.testing.assert_array_almost_equal(TPM_11p, TPM_11p_ref, decimal=5)
        np.testing.assert_array_almost_equal(EM_11p, EM_11p_ref, decimal=5)
        np.testing.assert_array_almost_equal(PM_11p, PM_11p_ref, decimal=5)

        # keeping the savecommands in case the files need to be updated
        #np.savetxt('test_tf_misfit_data/TFEM_11p.dat', TFEM_11p, fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/TFPM_11p.dat', TFPM_11p, fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/TEM_11p.dat', TEM_11p, fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/FEM_11p.dat', FEM_11p, fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/FPM_11p.dat', FPM_11p, fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/TPM_11p.dat', TPM_11p, fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/EM_11p.dat', (EM_11p,), fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/PM_11p.dat', (PM_11p,), fmt='%1.5e')

    def test_envelope_misfit(self):
        # tests all tf misfits with a signal that has envelope misfit
        S1 = self.S1
        S1a = self.S1a
        t = self.t
        dt = self.dt
        f = self.f
        
        fmin = self.fmin 
        fmax = self.fmax 
        nf = self.nf
        w0 = self.w0

        TFEM_11a_ref = np.loadtxt('test_tf_misfit_data/TFEM_11a.dat')
        TFPM_11a_ref = np.loadtxt('test_tf_misfit_data/TFPM_11a.dat')
        TEM_11a_ref = np.loadtxt('test_tf_misfit_data/TEM_11a.dat')
        FEM_11a_ref = np.loadtxt('test_tf_misfit_data/FEM_11a.dat')
        FPM_11a_ref = np.loadtxt('test_tf_misfit_data/FPM_11a.dat')
        TPM_11a_ref = np.loadtxt('test_tf_misfit_data/TPM_11a.dat')
        EM_11a_ref = np.loadtxt('test_tf_misfit_data/EM_11a.dat')
        PM_11a_ref = np.loadtxt('test_tf_misfit_data/PM_11a.dat')

        TFEM_11a = tfem(S1(t), S1a(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TFPM_11a = tfpm(S1(t), S1a(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TEM_11a = tem(S1(t), S1a(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        TPM_11a = tpm(S1(t), S1a(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        FEM_11a = fem(S1(t), S1a(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        FPM_11a = fpm(S1(t), S1a(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        EM_11a = em(S1(t), S1a(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        PM_11a = pm(S1(t), S1a(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)

        np.testing.assert_array_almost_equal(TFEM_11a, TFEM_11a_ref, decimal=5)
        np.testing.assert_array_almost_equal(TFPM_11a, TFPM_11a_ref, decimal=5)
        np.testing.assert_array_almost_equal(TEM_11a, TEM_11a_ref, decimal=5)
        np.testing.assert_array_almost_equal(FEM_11a, FEM_11a_ref, decimal=5)
        np.testing.assert_array_almost_equal(FPM_11a, FPM_11a_ref, decimal=5)
        np.testing.assert_array_almost_equal(TPM_11a, TPM_11a_ref, decimal=5)
        np.testing.assert_array_almost_equal(EM_11a, EM_11a_ref, decimal=5)
        np.testing.assert_array_almost_equal(PM_11a, PM_11a_ref, decimal=5)

        # keeping the savecommands in case the files need to be updated
        #np.savetxt('test_tf_misfit_data/TFEM_11a.dat', TFEM_11a, fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/TFPM_11a.dat', TFPM_11a, fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/TEM_11a.dat', TEM_11a, fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/FEM_11a.dat', FEM_11a, fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/FPM_11a.dat', FPM_11a, fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/TPM_11a.dat', TPM_11a, fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/EM_11a.dat', (EM_11a,), fmt='%1.5e')
        #np.savetxt('test_tf_misfit_data/PM_11a.dat', (PM_11a,), fmt='%1.5e')

def suite():
    return unittest.makeSuite(TfTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

