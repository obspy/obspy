#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The tf_misfit test suite.
"""
import os

import numpy as np
import pytest
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from obspy.signal.tf_misfit import (eg, em, feg, fem, fpg, fpm, pg, pm, teg,
                                    tem, tfeg, tfem, tfpg, tfpm, tpg, tpm)
from obspy.signal.tf_misfit import plot_tfr, plot_tf_misfits, plot_tf_gofs


class TestTf:
    """
    Test cases for tf functions.
    """
    # path to test files
    path = os.path.join(os.path.dirname(__file__), 'data')

    @pytest.fixture(scope='class')
    def state(self):
        """Return a dict with state needed to run tests."""
        tmax = 3.
        npts = 60
        dt = tmax / (npts - 1)

        fmin = 1.
        fmax = 3.
        nf = 3

        # Constants for s1
        a1 = 4.
        t1 = .1
        f1 = 2.
        phi1 = 0.

        # Constants for s1t and s1a
        ps = 0.1
        a1a = a1 * 1.1

        t = np.linspace(0., tmax, npts)
        f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

        def h(t):
            return (np.sign(t) + 1) / 2

        def s1(t):
            return a1 * (t - t1) * np.exp(-2 * (t - t1)) * \
                np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * h(t - t1)

        # generate analytical signal (hilbert transform) and add phase shift
        s1h = hilbert(s1(t))
        s1p = np.real(
            np.abs(s1h) * np.exp(np.angle(s1h) * 1j + ps * np.pi * 1j))

        # signal with amplitude error
        def s1a(t):
            return a1a * (t - t1) * np.exp(-2 * (t - t1)) * \
                np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * h(t - t1)

        out = dict(
            s1=s1,
            s1p=s1p,
            s1a=s1a,
            t=t,
            f=f,
            dt=dt,
            fmin=fmin,
            fmax=fmax,
            nf=nf,
            npts=npts,
            w0=6
        )
        return out

    def test_phase_misfit(self, state):
        """
        Tests all tf misfits with a signal that has phase misfit
        """
        s1 = state['s1']
        s1p = state['s1p']
        t = state['t']
        dt = state['dt']

        fmin = state['fmin']
        fmax = state['fmax']
        nf = state['nf']

        tfem_11p_ref = np.loadtxt(self.path + os.sep + 'TFEM_11p.dat')
        tfpm_11p_ref = np.loadtxt(self.path + os.sep + 'TFPM_11p.dat')
        tem_11p_ref = np.loadtxt(self.path + os.sep + 'TEM_11p.dat')
        fem_11p_ref = np.loadtxt(self.path + os.sep + 'FEM_11p.dat')
        fpm_11p_ref = np.loadtxt(self.path + os.sep + 'FPM_11p.dat')
        tpm_11p_ref = np.loadtxt(self.path + os.sep + 'TPM_11p.dat')
        em_11p_ref = np.loadtxt(self.path + os.sep + 'EM_11p.dat')
        pm_11p_ref = np.loadtxt(self.path + os.sep + 'PM_11p.dat')

        tfem_11p = tfem(s1p, s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        tfpm_11p = tfpm(s1p, s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        tem_11p = tem(s1p, s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        fem_11p = fem(s1p, s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        fpm_11p = fpm(s1p, s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        tpm_11p = tpm(s1p, s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        em_11p = em(s1p, s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        pm_11p = pm(s1p, s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)

        tol = 1e-5
        atol_min = 1e-15

        assert np.allclose(tfem_11p, tfem_11p_ref, rtol=tol,
               atol=np.abs(tfem_11p_ref).max() * tol + atol_min)
        assert np.allclose(tfpm_11p, tfpm_11p_ref, rtol=tol,
               atol=np.abs(tfpm_11p_ref).max() * tol + atol_min)
        assert np.allclose(tem_11p, tem_11p_ref, rtol=tol,
               atol=np.abs(tem_11p_ref).max() * tol + atol_min)
        assert np.allclose(fem_11p, fem_11p_ref, rtol=tol,
               atol=np.abs(fem_11p_ref).max() * tol + atol_min)
        assert np.allclose(fpm_11p, fpm_11p_ref, rtol=tol,
               atol=np.abs(fpm_11p_ref).max() * tol + atol_min)
        assert np.allclose(tpm_11p, tpm_11p_ref, rtol=tol,
               atol=np.abs(tpm_11p_ref).max() * tol + atol_min)
        assert np.allclose(em_11p, em_11p_ref, rtol=tol,
               atol=np.abs(em_11p_ref).max() * tol + atol_min)
        assert np.allclose(pm_11p, pm_11p_ref, rtol=tol,
               atol=np.abs(pm_11p_ref).max() * tol + atol_min)

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

    def test_envelope_misfit(self, state):
        """
        Tests all tf misfits with a signal that has envelope misfit
        """
        s1 = state['s1']
        s1a = state['s1a']
        t = state['t']
        dt = state['dt']

        fmin = state['fmin']
        fmax = state['fmax']
        nf = state['nf']

        tfem_11a_ref = np.loadtxt(self.path + os.sep + 'TFEM_11a.dat')
        tfpm_11a_ref = np.loadtxt(self.path + os.sep + 'TFPM_11a.dat')
        tem_11a_ref = np.loadtxt(self.path + os.sep + 'TEM_11a.dat')
        fem_11a_ref = np.loadtxt(self.path + os.sep + 'FEM_11a.dat')
        fpm_11a_ref = np.loadtxt(self.path + os.sep + 'FPM_11a.dat')
        tpm_11a_ref = np.loadtxt(self.path + os.sep + 'TPM_11a.dat')
        em_11a_ref = np.loadtxt(self.path + os.sep + 'EM_11a.dat')
        pm_11a_ref = np.loadtxt(self.path + os.sep + 'PM_11a.dat')

        tfem_11a = tfem(s1a(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        tfpm_11a = tfpm(s1a(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        tem_11a = tem(s1a(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        tpm_11a = tpm(s1a(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        fem_11a = fem(s1a(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        fpm_11a = fpm(s1a(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        em_11a = em(s1a(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        pm_11a = pm(s1a(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)

        tol = 1e-5
        atol_min = 1e-15

        assert np.allclose(tfem_11a, tfem_11a_ref, rtol=tol,
               atol=np.abs(tfem_11a_ref).max() * tol + atol_min)
        assert np.allclose(tfpm_11a, tfpm_11a_ref, rtol=tol,
               atol=np.abs(tfpm_11a_ref).max() * tol + atol_min)
        assert np.allclose(tem_11a, tem_11a_ref, rtol=tol,
               atol=np.abs(tem_11a_ref).max() * tol + atol_min)
        assert np.allclose(fem_11a, fem_11a_ref, rtol=tol,
               atol=np.abs(fem_11a_ref).max() * tol + atol_min)
        assert np.allclose(fpm_11a, fpm_11a_ref, rtol=tol,
               atol=np.abs(fpm_11a_ref).max() * tol + atol_min)
        assert np.allclose(tpm_11a, tpm_11a_ref, rtol=tol,
               atol=np.abs(tpm_11a_ref).max() * tol + atol_min)
        assert np.allclose(em_11a, em_11a_ref, rtol=tol,
               atol=np.abs(em_11a_ref).max() * tol + atol_min)
        assert np.allclose(pm_11a, pm_11a_ref, rtol=tol,
               atol=np.abs(pm_11a_ref).max() * tol + atol_min)

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

    def test_envelope_gof(self, state):
        """
        Tests all tf gofs
        """
        s1 = state['s1']
        t = state['t']
        dt = state['dt']

        fmin = state['fmin']
        fmax = state['fmax']
        nf = state['nf']
        npts = state['npts']

        tol = 1e-5

        _tfeg = tfeg(s1(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        _tfpg = tfpg(s1(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        _teg = teg(s1(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        _tpg = tpg(s1(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        _feg = feg(s1(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        _fpg = fpg(s1(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        _eg = eg(s1(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)
        _pg = pg(s1(t), s1(t), dt=dt, fmin=fmin, fmax=fmax, nf=nf)

        assert np.allclose(_tfeg, np.ones((nf, npts)) * 10., rtol=tol)
        assert np.allclose(_tfpg, np.ones((nf, npts)) * 10., rtol=tol)
        assert np.allclose(_teg, np.ones(npts) * 10., rtol=tol)
        assert np.allclose(_tpg, np.ones(npts) * 10., rtol=tol)
        assert np.allclose(_feg, np.ones(nf) * 10., rtol=tol)
        assert np.allclose(_fpg, np.ones(nf) * 10., rtol=tol)
        assert np.allclose(_eg, 10., rtol=tol)
        assert np.allclose(_pg, 10., rtol=tol)


class TestTfPlot:
    """
    Test cases for tf plot functions.
    """
    path = os.path.join(os.path.dirname(__file__), 'images')

    @pytest.fixture(scope='class')
    def state(self):
        """return state for testing."""
        from obspy.core.util import AttribDict
        out = AttribDict()
        # path to test files
        # self.path =
        # general constants
        tmax = 6.
        out.dt = 0.01
        npts = int(tmax / out.dt + 1)
        t = np.linspace(0., tmax, npts)
        out.fmin = .5
        out.fmax = 10
        # constants for the signal
        a1 = 4.
        t1 = 2.
        f1 = 2.
        phi1 = 0.
        # amplitude and phase error
        phase_shift = 0.1
        amp_fac = 1.2

        # generate the signal
        h1 = (np.sign(t - t1) + 1) / 2
        out.st1 = a1 * (t - t1) * np.exp(-2 * (t - t1))
        out.st1 *= np.cos(2. * np.pi * f1 * (t - t1) + phi1 * np.pi) * h1

        # reference signal
        out.st2 = out.st1.copy()

        # generate analytical signal (hilbert transform) and add phase shift
        out.st1p = hilbert(out.st1)
        out.st1p = np.real(
            np.abs(out.st1p) *
            np.exp((np.angle(out.st1p) + phase_shift * np.pi) * 1j))

        # signal with amplitude error
        out.st1p /= amp_fac
        return out

    def test_plot_tfr(self, state, ignore_numpy_errors, image_path):
        n = 295
        t, dt = np.linspace(0., 20 * np.pi, n, retstep=True)
        sig = np.sin(t)
        plot_tfr(sig, dt=dt, show=False)
        plt.savefig(image_path)

    def test_plot_tf_misfits(self, image_path, ignore_numpy_errors, state):
        plot_tf_misfits(state.st1p, state.st2, dt=state.dt,
                        fmin=state.fmin, fmax=state.fmax, show=False)
        plt.savefig(image_path)

    def test_plot_tf_gofs(self, state, ignore_numpy_errors, image_path):
        plot_tf_gofs(state.st1p, state.st2, dt=state.dt, fmin=state.fmin,
                     fmax=state.fmax, show=False)
        plt.savefig(image_path)
