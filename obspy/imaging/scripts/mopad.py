#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: mopad.py
#  Purpose: Moment tensor Plotting and Decomposition tool
#   Author: Lars Krieger, Sebastian Heimann
#    Email: lars.krieger@zmaw.de, sebastian.heimann@zmaw.de
#
# Copyright (C) 2010 Lars Krieger, Sebastian Heimann
# --------------------------------------------------------------------
"""
USAGE: obspy-mopad [plot,decompose,gmt,convert] SOURCE_MECHANISM [OPTIONS]

::

    #######################################################################
    #########################   MoPaD  ####################################
    ######### Moment tensor Plotting and Decomposition tool ###############
    #######################################################################

    Multi method tool for:

    - Plotting and saving of focal sphere diagrams ('Beachballs').

    - Decomposition and Conversion of seismic moment tensors.

    - Generating coordinates, describing a focal sphere diagram, to be
      piped into GMT's psxy (Useful where psmeca or pscoupe fail.)

    For more help, please run ``python mopad.py --help``.


    #######################################################################

    Version  0.7

    #######################################################################

    Copyright (C) 2010
    Lars Krieger & Sebastian Heimann

    Contact
    lars.krieger@zmaw.de  &  sebastian.heimann@zmaw.de

    #######################################################################

    License:

    GNU Lesser General Public License, Version 3

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301, USA.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io
import math
import numpy as np
import os
import os.path
import sys
import warnings
from obspy import __version__


MOPAD_VERSION = 0.7


# constants:
dynecm = 1e-7
pi = np.pi

epsilon = 1e-13

rad2deg = 180. / pi


class MTError(Exception):
    pass


class MomentTensor:
    """
    """
    def __init__(self, M=None, system='NED', debug=0):
        """
        Creates a moment tensor object on the basis of a provided mechanism M.

        If M is a non symmetric 3x3-matrix, the upper right triangle
        of the matrix is taken as reference. M is symmetrisised
        w.r.t. these entries. If M is provided as a 3-,4-,6-,7-tuple
        or array, it is converted into a matrix internally according
        to standard conventions (Aki & Richards).

        'system' may be chosen as 'NED','USE','NWU', or 'XYZ'.

        'debug' enables output on the shell at the intermediate steps.
        """

        source_mechanism = M
        self._original_M = M[:]

        # basis system:
        self._input_basis = 'NED'
        self._list_of_possible_input_bases = ['NED', 'USE', 'XYZ', 'NWU']
        self._list_of_possible_output_bases = ['NED', 'USE', 'XYZ', 'NWU']

        self._input_basis = system.upper()

        # bring M to symmetric matrix form
        self._M = self._setup_M(source_mechanism)

        # transform M into NED system for internal calculations
        self._rotate_2_NED()

        #
        # set attributes list
        #

        # decomposition:
        self._decomposition_key = 20

        # eigenvector / principal-axes system:
        self._eigenvalues = None
        self._eigenvectors = None
        self._null_axis = None
        self._t_axis = None
        self._p_axis = None
        self._rotation_matrix = None

        # optional - maybe set afterwards by external application - for later
        # plotting:
        self._best_faultplane = None
        self._auxiliary_plane = None

        #
        # RUN:
        #

        # carry out the MT decomposition - results are in basis NED
        self._decompose_M()

        # set the appropriate principal axis system:
        self._M_to_principal_axis_system()

    def _setup_M(self, mech):
        """
        Brings the provided mechanism into symmetric 3x3 matrix form.

        The source mechanism may be provided in different forms:

        * as 3x3 matrix - symmetry is checked - one basis system has to be
           chosen, or NED as default is taken
        * as 3-element tuple or array - interpreted as strike, dip, slip-rake
           angles in degree
        * as 4-element tuple or array - interpreted as strike, dip, slip-rake
          angles in degree + seismic scalar moment in Nm
        * as 6-element tuple or array - interpreted as the 6 independent
          entries of the moment tensor
        * as 7-element tuple or array - interpreted as the 6 independent
          entries of the moment tensor + seismic scalar moment in Nm
        * as 9-element tuple or array - interpreted as the 9 entries of the
          moment tensor - checked for symmetry
        * as a nesting of one of the upper types (e.g. a list of n-tuples);
          first element of outer nesting is taken
        """
        # set source mechanism to matrix form

        if mech is None:
            print('\n ERROR !! - Please provide a mechanism !!\n')
            raise MTError(' !! ')

        # if some stupid nesting occurs
        if len(mech) == 1:
            mech = mech[0]

        # all 9 elements are given
        if np.prod(np.shape(mech)) == 9:
            if np.shape(mech)[0] == 3:
                # assure symmetry:
                mech[1, 0] = mech[0, 1]
                mech[2, 0] = mech[0, 2]
                mech[2, 1] = mech[1, 2]
                new_M = mech
            else:
                new_M = np.array(mech).reshape(3, 3).copy()
                new_M[1, 0] = new_M[0, 1]
                new_M[2, 0] = new_M[0, 2]
                new_M[2, 1] = new_M[1, 2]

        # mechanism given as 6- or 7-tuple, list or array
        if len(mech) == 6 or len(mech) == 7:
            M = mech
            new_M = np.matrix([M[0], M[3], M[4],
                              M[3], M[1], M[5],
                              M[4], M[5], M[2]]).reshape(3, 3)

            if len(mech) == 7:
                new_M *= M[6]

        # if given as strike, dip, rake, conventions from Jost & Herrmann hold
        # resulting matrix is in NED-basis:
        if len(mech) == 3 or len(mech) == 4:
            try:
                [float(val) for val in mech]
            except:
                msg = "angles must be given as floats, separated by commas"
                sys.exit('\n  ERROR -  %s\n  ' % msg)

            strike = mech[0]
            if not 0 <= strike <= 360:
                msg = "strike angle must be between 0° and 360°"
                sys.exit('\n  ERROR -  %s\n  ' % msg)
            dip = mech[1]
            if not -90 <= dip <= 90:
                msg = "dip angle must be between -90° and 90°"
                sys.exit('\n  ERROR -  %s\n  ' % msg)
            rake = mech[2]
            if not -180 <= rake <= 180:
                msg = "slip-rake angle must be between -180° and 180°"
                sys.exit('\n  ERROR -  %s\n  ' % msg)

            moms = strikediprake_2_moments(strike, dip, rake)

            new_M = np.matrix([moms[0], moms[3], moms[4],
                              moms[3], moms[1], moms[5],
                              moms[4], moms[5], moms[2]]).reshape(3, 3)

            if len(mech) == 4:
                new_M *= mech[3]

            # to assure right basis system - others are meaningless, provided
            # these angles
            self._input_basis = 'NED'

        return np.asmatrix(new_M)

    def _rotate_2_NED(self):
        """
        Rotates the mechanism to the basis NED.

        All internal calculations are carried out within the NED space.
        """
        if self._input_basis not in self._list_of_possible_input_bases:
            print('provided input basis not implemented - please specify one',
                  end=' ')
            print('of the following bases:',
                  self._list_of_possible_input_bases)
            raise MTError(' !! ')

        NED_2_NED = np.asmatrix(np.diag([1, 1, 1]))

        rotmat_USE_2_NED = NED_2_NED.copy()
        rotmat_USE_2_NED[:] = 0
        rotmat_USE_2_NED[0, 1] = -1
        rotmat_USE_2_NED[1, 2] = 1
        rotmat_USE_2_NED[2, 0] = -1

        rotmat_XYZ_2_NED = NED_2_NED.copy()
        rotmat_XYZ_2_NED[:] = 0
        rotmat_XYZ_2_NED[0, 1] = 1
        rotmat_XYZ_2_NED[1, 0] = 1
        rotmat_XYZ_2_NED[2, 2] = -1

        rotmat_NWU_2_NED = NED_2_NED.copy()
        rotmat_NWU_2_NED[1, 1] = -1
        rotmat_NWU_2_NED[2, 2] = -1

        if self._input_basis == 'NED':
            pass
        elif self._input_basis == 'USE':
            self._M = np.dot(rotmat_USE_2_NED,
                             np.dot(self._M, rotmat_USE_2_NED.T))
        elif self._input_basis == 'XYZ':
            self._M = np.dot(rotmat_XYZ_2_NED,
                             np.dot(self._M, rotmat_XYZ_2_NED.T))
        elif self._input_basis == 'NWU':
            self._M = np.dot(rotmat_NWU_2_NED,
                             np.dot(self._M, rotmat_NWU_2_NED.T))

    def _decompose_M(self):
        """
        Running the decomposition of the moment tensor object.

        the standard decompositions M = Isotropic + DC + (CLVD or 2nd DC) are
        supported (C.f. Jost & Herrmann, Aki & Richards)
        """
        if self._decomposition_key == 20:
            self._standard_decomposition()
        elif self._decomposition_key == 21:
            self._decomposition_w_2DC()
        elif self._decomposition_key == 31:
            self._decomposition_w_3DC()
        else:
            raise MTError(' only standard decompositions supported ')

    def _standard_decomposition(self):
        """
        Decomposition according Aki & Richards and Jost & Herrmann into

        isotropic + deviatoric
        = isotropic + DC + CLVD

        parts of the input moment tensor.

        results are given as attributes, callable via the get_* function:

        DC, CLVD, DC_percentage, seismic_moment, moment_magnitude
        """
        M = self._M

        # isotropic part
        M_iso = np.diag(np.array([1. / 3 * np.trace(M),
                                 1. / 3 * np.trace(M),
                                 1. / 3 * np.trace(M)]))
        M0_iso = abs(1. / 3 * np.trace(M))

        # deviatoric part
        M_devi = M - M_iso

        self._isotropic = M_iso
        self._deviatoric = M_devi

        # eigenvalues and -vectors
        eigenwtot, eigenvtot = np.linalg.eig(M)

        # eigenvalues and -vectors of the deviatoric part
        eigenw1, eigenv1 = np.linalg.eig(M_devi)

        # eigenvalues in ascending order:
        eigenw = np.real(np.take(eigenw1, np.argsort(abs(eigenwtot))))
        eigenv = np.real(np.take(eigenv1, np.argsort(abs(eigenwtot)), 1))

        # eigenvalues in ascending order in absolute value!!:
        eigenw_devi = np.real(np.take(eigenw1, np.argsort(abs(eigenw1))))

        M0_devi = max(abs(eigenw_devi))

        # named according to Jost & Herrmann:
        a1 = eigenv[:, 0]
        a2 = eigenv[:, 1]
        a3 = eigenv[:, 2]

        # eigen values can be zero in some cases. this is handled in the
        # following try/except.
        with warnings.catch_warnings(record=True):
            np_err = np.seterr(all="warn")
            F = -eigenw_devi[0] / eigenw_devi[2]

            M_DC = \
                eigenw[2] * (1 - 2 * F) * (np.outer(a3, a3) - np.outer(a2, a2))
            M_CLVD = eigenw[2] * F * (2 * np.outer(a3, a3) - np.outer(a2, a2) -
                                      np.outer(a1, a1))
            np.seterr(**np_err)

        try:
            M_DC_percentage = int(round((1 - 2 * abs(F)) * 100, 6))
        except ValueError:
            # this should only occur in the pure isotropic case
            M_DC_percentage = 0.

        # according to Bowers & Hudson:
        M0 = M0_iso + M0_devi

        M_iso_percentage = int(round(M0_iso / M0 * 100, 6))
        self._iso_percentage = M_iso_percentage

        self._DC = M_DC
        self._CLVD = M_CLVD
        self._DC_percentage = int(round((100 - M_iso_percentage) *
                                        M_DC_percentage / 100.))

        self._seismic_moment = M0
        self._moment_magnitude = \
            np.log10(self._seismic_moment * 1.0e7) / 1.5 - 10.7

    def _decomposition_w_2DC(self):
        """
        Decomposition according Aki & Richards and Jost & Herrmann into

        isotropic + deviatoric
        = isotropic + DC + DC2

        parts of the input moment tensor.

        results are given as attributes, callable via the get_* function:

        DC1, DC2, DC_percentage, seismic_moment, moment_magnitude
        """
        M = self._M

        # isotropic part
        M_iso = np.diag(np.array([1. / 3 * np.trace(M),
                                 1. / 3 * np.trace(M),
                                 1. / 3 * np.trace(M)]))
        M0_iso = abs(1. / 3 * np.trace(M))

        # deviatoric part
        M_devi = M - M_iso

        self._isotropic = M_iso
        self._deviatoric = M_devi

        # eigenvalues and -vectors of the deviatoric part
        eigenw1, eigenv1 = np.linalg.eig(M_devi)

        # eigenvalues in ascending order of their absolute values:
        eigenw = np.real(np.take(eigenw1, np.argsort(abs(eigenw1))))
        eigenv = np.real(np.take(eigenv1, np.argsort(abs(eigenw1)), 1))

        M0_devi = max(abs(eigenw))

        # named according to Jost & Herrmann:
        a1 = eigenv[:, 0]
        a2 = eigenv[:, 1]
        a3 = eigenv[:, 2]

        M_DC = eigenw[2] * (np.outer(a3, a3) - np.outer(a2, a2))
        M_DC2 = eigenw[0] * (np.outer(a1, a1) - np.outer(a2, a2))

        M_DC_percentage = int(round(abs(eigenw[2] / (abs(eigenw[2]) +
                                                     abs(eigenw[0]))) * 100.))

        # according to Bowers & Hudson:
        M0 = M0_iso + M0_devi

        M_iso_percentage = int(round(M0_iso / M0 * 100))
        self._iso_percentage = M_iso_percentage

        self._DC = M_DC
        self._DC2 = M_DC2
        # self._DC_percentage =  M_DC_percentage
        self._DC_percentage = int(round((100 - M_iso_percentage) *
                                        M_DC_percentage / 100.))
        # and M_DC2_percentage?

        self._seismic_moment = M0
        self._moment_magnitude = \
            np.log10(self._seismic_moment * 1.0e7) / 1.5 - 10.7

    def _decomposition_w_3DC(self):
        """
        Decomposition according Aki & Richards and Jost & Herrmann into

        - isotropic
        - deviatoric
        - 3 DC

        parts of the input moment tensor.

        results are given as attributes, callable via the get_* function:

        DC1, DC2, DC3, DC_percentage, seismic_moment, moment_magnitude
        """
        M = self._M

        # isotropic part
        M_iso = np.diag(np.array([1. / 3 * np.trace(M),
                                 1. / 3 * np.trace(M),
                                 1. / 3 * np.trace(M)]))
        M0_iso = abs(1. / 3 * np.trace(M))

        # deviatoric part
        M_devi = M - M_iso

        self._isotropic = M_iso
        self._deviatoric = M_devi

        # eigenvalues and -vectors of the deviatoric part
        eigenw1, eigenv1 = np.linalg.eig(M_devi)
        M0_devi = max(abs(eigenw1))

        # eigenvalues and -vectors of the full M !!!!!!!!
        eigenw1, eigenv1 = np.linalg.eig(M)

        # eigenvalues in ascending order of their absolute values:
        eigenw = np.real(np.take(eigenw1, np.argsort(abs(eigenw1))))
        eigenv = np.real(np.take(eigenv1, np.argsort(abs(eigenw1)), 1))

        # named according to Jost & Herrmann:
        a1 = eigenv[:, 0]
        a2 = eigenv[:, 1]
        a3 = eigenv[:, 2]

        M_DC1 = 1. / 3. * (eigenw[0] - eigenw[1]) * (np.outer(a1, a1) -
                                                     np.outer(a2, a2))
        M_DC2 = 1. / 3. * (eigenw[1] - eigenw[2]) * (np.outer(a2, a2) -
                                                     np.outer(a3, a3))
        M_DC3 = 1. / 3. * (eigenw[2] - eigenw[0]) * (np.outer(a3, a3) -
                                                     np.outer(a1, a1))

        M_DC1_perc = int(100 * abs((eigenw[0] - eigenw[1])) /
                         (abs((eigenw[1] - eigenw[2])) +
                          abs((eigenw[1] - eigenw[2])) +
                          abs((eigenw[2] - eigenw[0]))))
        M_DC2_perc = int(100 * abs((eigenw[1] - eigenw[2])) /
                         (abs((eigenw[1] - eigenw[2])) +
                          abs((eigenw[1] - eigenw[2])) +
                          abs((eigenw[2] - eigenw[0]))))

        self._DC = M_DC1
        self._DC2 = M_DC2
        self._DC3 = M_DC3

        self._DC_percentage = M_DC1_perc
        self._DC2_percentage = M_DC2_perc

        # according to Bowers & Hudson:
        M0 = M0_iso + M0_devi

        M_iso_percentage = int(M0_iso / M0 * 100)
        self._iso_percentage = M_iso_percentage

        # self._seismic_moment   = np.sqrt(1./2*nnp.sum(eigenw**2) )
        self._seismic_moment = M0
        self._moment_magnitude = \
            np.log10(self._seismic_moment * 1.0e7) / 1.5 - 10.7

    def _M_to_principal_axis_system(self):
        """
        Read in Matrix M and set up eigenvalues (EW) and eigenvectors
        (EV) for setting up the principal axis system.

        The internal convention is the 'HNS'-system: H is the
        eigenvector for the smallest absolute eigenvalue, S is the
        eigenvector for the largest absolute eigenvalue, N is the null
        axis.

        Naming due to the geometry: a CLVD is
        Symmetric to the S-axis,
        Null-axis is common sense, and the third (auxiliary) axis
        Helps to construct the R³.

        Additionally builds matrix for basis transformation back to NED system.

        The eigensystem setup defines the colouring order for a later
        plotting in the BeachBall class. This order is set by the
        '_plot_clr_order' attribute.
        """
        M = self._M
        M_devi = self._deviatoric

        # working in framework of 3 principal axes:
        # eigenvalues (EW) are in order from high to low
        # - neutral axis N, belongs to middle EW
        # - symmetry axis S ('sigma') belongs to EW with largest absolute value
        #   (P- or T-axis)
        # - auxiliary axis H ('help') belongs to remaining EW (T- or P-axis)
        # EW sorting from lowest to highest value
        EW_devi, EV_devi = np.linalg.eigh(M_devi)
        EW_order = np.argsort(EW_devi)

        # print('order', EW_order)

        if 1:  # self._plot_isotropic_part:
            trace_M = np.trace(M)
            if abs(trace_M) < epsilon:
                trace_M = 0
            EW, EV = np.linalg.eigh(M)
            for i, ew in enumerate(EW):
                if abs(EW[i]) < epsilon:
                    EW[i] = 0
        else:
            trace_M = np.trace(M_devi)
            if abs(trace_M) < epsilon:
                trace_M = 0

            EW, EV = np.linalg.eigh(M_devi)
            for i, ew in enumerate(EW):
                if abs(EW[i]) < epsilon:
                    EW[i] = 0

        EW1_devi = EW_devi[EW_order[0]]
        EW2_devi = EW_devi[EW_order[1]]
        EW3_devi = EW_devi[EW_order[2]]
        EV1_devi = EV_devi[:, EW_order[0]]
        EV2_devi = EV_devi[:, EW_order[1]]
        EV3_devi = EV_devi[:, EW_order[2]]

        EW1 = EW[EW_order[0]]
        EW2 = EW[EW_order[1]]
        EW3 = EW[EW_order[2]]
        EV1 = EV[:, EW_order[0]]
        EV2 = EV[:, EW_order[1]]
        EV3 = EV[:, EW_order[2]]

        chng_basis_tmp = np.asmatrix(np.zeros((3, 3)))
        chng_basis_tmp[:, 0] = EV1_devi
        chng_basis_tmp[:, 1] = EV2_devi
        chng_basis_tmp[:, 2] = EV3_devi

        symmetry_around_tension = 1
        clr = 1

        if abs(EW2_devi) < epsilon:
            EW2_devi = 0

        # implosion
        if EW1 < 0 and EW2 < 0 and EW3 < 0:
            symmetry_around_tension = 0
            # logger.debug( 'IMPLOSION - symmetry around pressure axis \n\n')
            clr = 1
        # explosion
        elif EW1 > 0 and EW2 > 0 and EW3 > 0:
            symmetry_around_tension = 1
            if abs(EW1_devi) > abs(EW3_devi):
                symmetry_around_tension = 0
            # logger.debug( 'EXPLOSION - symmetry around tension axis \n\n')
            clr = -1
        # net-implosion
        elif EW2 < 0 and sum([EW1, EW2, EW3]) < 0:
            if abs(EW1_devi) < abs(EW3_devi):
                symmetry_around_tension = 1
                clr = 1
            else:
                symmetry_around_tension = 1
                clr = 1
        # net-implosion
        elif EW2_devi >= 0 and sum([EW1, EW2, EW3]) < 0:
            symmetry_around_tension = 0
            clr = -1
            if abs(EW1_devi) < abs(EW3_devi):
                symmetry_around_tension = 1
                clr = 1
        # net-explosion
        elif EW2_devi < 0 and sum([EW1, EW2, EW3]) > 0:
            symmetry_around_tension = 1
            clr = 1
            if abs(EW1_devi) > abs(EW3_devi):
                symmetry_around_tension = 0
                clr = -1
        # net-explosion
        elif EW2_devi >= 0 and sum([EW1, EW2, EW3]) > 0:
            symmetry_around_tension = 0
            clr = -1
        else:
            pass
        if abs(EW1_devi) < abs(EW3_devi):
            symmetry_around_tension = 1
            clr = 1
            if 0:  # EW2 > 0 :#or (EW2 > 0 and EW2_devi > 0) :
                symmetry_around_tension = 0
                clr = -1

        if abs(EW1_devi) >= abs(EW3_devi):
            symmetry_around_tension = 0
            clr = -1
            if 0:  # EW2 < 0 :
                symmetry_around_tension = 1
                clr = 1
        if (EW3 < 0 and np.trace(self._M) >= 0):
            print('Houston, we have had a problem  - check M !!!!!!')
            raise MTError(' !! ')

        if trace_M == 0:
            if EW2 == 0:
                symmetry_around_tension = 1
                clr = 1
            elif 2 * abs(EW2) == abs(EW1) or 2 * abs(EW2) == abs(EW3):
                if abs(EW1) < EW3:
                    symmetry_around_tension = 1
                    clr = 1
                else:
                    symmetry_around_tension = 0
                    clr = -1
            else:
                if abs(EW1) < EW3:
                    symmetry_around_tension = 1
                    clr = 1
                else:
                    symmetry_around_tension = 0
                    clr = -1

        if symmetry_around_tension == 1:
            EWs = EW3.copy()
            EVs = EV3.copy()
            EWh = EW1.copy()
            EVh = EV1.copy()

        else:
            EWs = EW1.copy()
            EVs = EV1.copy()
            EWh = EW3.copy()
            EVh = EV3.copy()

        EWn = EW2
        EVn = EV2

        # build the basis system change matrix:
        chng_basis = np.asmatrix(np.zeros((3, 3)))

        # order of eigenvector's basis: (H,N,S)
        chng_basis[:, 0] = EVh
        chng_basis[:, 1] = EVn
        chng_basis[:, 2] = EVs

        # matrix for basis transformation
        self._rotation_matrix = chng_basis

        # collections of eigenvectors and eigenvalues
        self._eigenvectors = [EVh, EVn, EVs]
        self._eigenvalues = [EWh, EWn, EWs]

        # principal axes
        self._null_axis = EVn
        self._t_axis = EV1
        self._p_axis = EV3

        # plotting order flag - important for plot in BeachBall class
        self._plot_clr_order = clr

        # collection of the faultplanes, given in strike, dip, slip-rake
        self._faultplanes = self._find_faultplanes()

    def _find_faultplanes(self):
        """
        Sets the two angle-triples, describing the faultplanes of the
        Double Couple, defined by the eigenvectors P and T of the
        moment tensor object.

        Defining a reference Double Couple with strike = dip =
        slip-rake = 0, the moment tensor object's DC is transformed
        (rotated) w.r.t. this orientation. The respective rotation
        matrix yields the first fault plane angles as the Euler
        angles. After flipping the first reference plane by
        multiplying the appropriate flip-matrix, one gets the second fault
        plane's geometry.

        All output angles are in degree

        (
        to check:
        mit Sebastians Konventionen:

        rotationsmatrix1 = EV Matrix von M, allerdings in der Reihenfolge TNP
            (nicht, wie hier PNT!!!)

        referenz-DC mit strike, dip, rake = 0,0,0  in NED - Darstellung:
            M = 0,0,0,0,-1,0

        davon die EV ebenfalls in eine Matrix:

        trafo-matrix2 = EV Matrix von Referenz-DC in der REihenfolge TNP

        effektive Rotationsmatrix = (rotationsmatrix1  * trafo-matrix2.T).T

        durch check, ob det <0, schauen, ob die Matrix mit -1 multipliziert
            werden muss

        flip_matrix = 0,0,-1,0,-1,0,-1,0,0

        andere DC Orientierung wird durch flip * effektive Rotationsmatrix
            erhalten

        beide Rotataionmatrizen in matrix_2_euler
        )
        """
        # reference Double Couple (in NED basis)
        # it has strike, dip, slip-rake = 0,0,0
        refDC = np.matrix([[0., 0., -1.], [0., 0., 0.], [-1., 0., 0.]],
                          dtype=np.float)
        refDC_evals, refDC_evecs = np.linalg.eigh(refDC)

        # matrix which is turning from one fault plane to the other
        flip_dc = np.matrix([[0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]],
                            dtype=np.float)

        # euler-tools need matrices of EV sorted in PNT:
        pnt_sorted_EV_matrix = self._rotation_matrix.copy()

        # resort only necessary, if abs(p) <= abs(t)
        # print(self._plot_clr_order)
        if self._plot_clr_order < 0:
            pnt_sorted_EV_matrix[:, 0] = self._rotation_matrix[:, 2]
            pnt_sorted_EV_matrix[:, 2] = self._rotation_matrix[:, 0]

        # rotation matrix, describing the rotation of the eigenvector
        # system of the input moment tensor into the eigenvector
        # system of the reference Double Couple
        rot_matrix_fp1 = (np.dot(pnt_sorted_EV_matrix, refDC_evecs.T)).T

        # check, if rotation has right orientation
        if np.linalg.det(rot_matrix_fp1) < 0.:
            rot_matrix_fp1 *= -1.

        # adding a rotation into the ambiguous system of the second fault plane
        rot_matrix_fp2 = np.dot(flip_dc, rot_matrix_fp1)

        fp1 = self._find_strike_dip_rake(rot_matrix_fp1)
        fp2 = self._find_strike_dip_rake(rot_matrix_fp2)

        return [fp1, fp2]

    def _find_strike_dip_rake(self, rotation_matrix):
        """
        Returns angles strike, dip, slip-rake in degrees, describing the fault
        plane.
        """
        (alpha, beta, gamma) = self._matrix_to_euler(rotation_matrix)
        return (beta * rad2deg, alpha * rad2deg, -gamma * rad2deg)

    def _cvec(self, x, y, z):
        """
        Builds a column vector (matrix type) from a 3 tuple.
        """
        return np.matrix([[x, y, z]], dtype=np.float).T

    def _matrix_to_euler(self, rotmat):
        """
        Returns three Euler angles alpha, beta, gamma (in radians) from a
        rotation matrix.
        """
        ex = self._cvec(1., 0., 0.)
        ez = self._cvec(0., 0., 1.)
        exs = rotmat.T * ex
        ezs = rotmat.T * ez
        enodes = np.cross(ez.T, ezs.T).T
        if np.linalg.norm(enodes) < 1e-10:
            enodes = exs
        enodess = rotmat * enodes
        cos_alpha = float((ez.T * ezs))
        if cos_alpha > 1.:
            cos_alpha = 1.
        if cos_alpha < -1.:
            cos_alpha = -1.
        alpha = np.arccos(cos_alpha)
        beta = np.mod(np.arctan2(enodes[1, 0], enodes[0, 0]), np.pi * 2.)
        gamma = np.mod(-np.arctan2(enodess[1, 0], enodess[0, 0]), np.pi * 2.)
        return self._unique_euler(alpha, beta, gamma)

    def _unique_euler(self, alpha, beta, gamma):
        """
        Uniquify euler angle triplet.

        Puts euler angles into ranges compatible with (dip,strike,-rake) in
        seismology:

            alpha (dip)   : [0, pi/2]
            beta (strike) : [0, 2*pi)
            gamma (-rake) : [-pi, pi)

        If alpha is near to zero, beta is replaced by beta+gamma and gamma is
        set to zero, to prevent that additional ambiguity.

        If alpha is near to pi/2, beta is put into the range [0,pi).
        """
        alpha = np.mod(alpha, 2.0 * pi)

        if 0.5 * pi < alpha and alpha <= pi:
            alpha = pi - alpha
            beta = beta + pi
            gamma = 2.0 * pi - gamma
        elif pi < alpha and alpha <= 1.5 * pi:
            alpha = alpha - pi
            gamma = pi - gamma
        elif 1.5 * pi < alpha and alpha <= 2.0 * pi:
            alpha = 2.0 * pi - alpha
            beta = beta + pi
            gamma = pi + gamma

        alpha = np.mod(alpha, 2.0 * pi)
        beta = np.mod(beta, 2.0 * pi)
        gamma = np.mod(gamma + pi, 2.0 * pi) - pi

        # If dip is exactly 90 degrees, one is still
        # free to choose between looking at the plane from either side.
        # Choose to look at such that beta is in the range [0,180)

        # This should prevent some problems, when dip is close to 90 degrees:
        if abs(alpha - 0.5 * pi) < 1e-10:
            alpha = 0.5 * pi
        if abs(beta - pi) < 1e-10:
            beta = pi
        if abs(beta - 2. * pi) < 1e-10:
            beta = 0.
        if abs(beta) < 1e-10:
            beta = 0.

        if alpha == 0.5 * pi and beta >= pi:
            gamma = -gamma
            beta = np.mod(beta - pi, 2.0 * pi)
            gamma = np.mod(gamma + pi, 2.0 * pi) - pi
            assert 0. <= beta < pi
            assert -pi <= gamma < pi

        if alpha < 1e-7:
            beta = np.mod(beta + gamma, 2.0 * pi)
            gamma = 0.

        return (alpha, beta, gamma)

    def _matrix_w_style_and_system(self, M2return, system, style):
        """
        Gives the provided matrix in the desired basis system.

        If the argument 'style' is set to 'fancy', a 'print' of the return
        value yields a nice shell output of the matrix for better
        visual control.
        """
        if not system.upper() in self._list_of_possible_output_bases:
            print('\nprovided output basis not supported - please specify',
                  end=' ')
            print('one of the following bases: (default=NED)\n', end=' ')
            print(self._list_of_possible_input_bases, '\n')
            raise MTError(' !! ')

        fancy = 0
        if style.lower() in ['f', 'fan', 'fancy']:
            fancy = 1

        if system.upper() == 'NED':
            if fancy:
                return fancy_matrix(M2return)
            else:
                return M2return

        elif system.upper() == 'USE':
            if fancy:
                return fancy_matrix(NED2USE(M2return))
            else:
                return NED2USE(M2return)

        elif system.upper() == 'XYZ':
            if fancy:
                return fancy_matrix(NED2XYZ(M2return))
            else:
                return NED2XYZ(M2return)

        elif system.upper() == 'NWU':
            if fancy:
                return fancy_matrix(NED2NWU(M2return))
            else:
                return NED2NWU(M2return)

    def _vector_w_style_and_system(self, vectors, system, style):
        """
        Gives the provided vector(s) in the desired basis system.

        If the argument 'style' is set to 'fancy', a 'print' of the return
        value yields a nice shell output of the vector(s) for better
        visual control.

        'vectors' can be either a single array, tuple, matrix or a collection
        in form of a list, array or matrix.
        If it's a list, each entry will be checked, if it's 3D - if not, an
        exception is raised.
        If it's a matrix or array with column-length 3, the columns are
        interpreted as vectors, otherwise, its transposed is used.
        """
        if not system.upper() in self._list_of_possible_output_bases:
            print('\n provided output basis not supported - please specify',
                  end=' ')
            print('one of the following bases: (default=NED)\n', end=' ')
            print(self._list_of_possible_input_bases, '\n')
            raise MTError(' !! ')

        fancy = 0
        if style.lower() in ['f', 'fan', 'fancy']:
            fancy = 1

        lo_vectors = []

        # if list of vectors
        if isinstance(vectors, list):
            for vec in vectors:
                if np.prod(np.shape(vec)) != 3:
                    print('\n please provide vector(s) from R³ \n ')
                    raise MTError(' !! ')
            lo_vectors = vectors
        else:
            if np.prod(np.shape(vectors)) % 3 != 0:
                print('\n please provide vector(s) from R³ \n ')
                raise MTError(' !! ')

            if np.shape(vectors)[0] == 3:
                for ii in range(np.shape(vectors)[1]):
                    lo_vectors.append(vectors[:, ii])
            else:
                for ii in range(np.shape(vectors)[0]):
                    lo_vectors.append(vectors[:, ii].transpose())

        lo_vecs_to_show = []

        for vec in lo_vectors:
            if system.upper() == 'NED':
                if fancy:
                    lo_vecs_to_show.append(fancy_vector(vec))
                else:
                    lo_vecs_to_show.append(vec)
            elif system.upper() == 'USE':
                if fancy:
                    lo_vecs_to_show.append(fancy_vector(NED2USE(vec)))
                else:
                    lo_vecs_to_show.append(NED2USE(vec))
            elif system.upper() == 'XYZ':
                if fancy:
                    lo_vecs_to_show.append(fancy_vector(NED2XYZ(vec)))
                else:
                    lo_vecs_to_show.append(NED2XYZ(vec))
            elif system.upper() == 'NWU':
                if fancy:
                    lo_vecs_to_show.append(fancy_vector(NED2NWU(vec)))
                else:
                    lo_vecs_to_show.append(NED2NWU(vec))

        if len(lo_vecs_to_show) == 1:
            return lo_vecs_to_show[0]
        else:
            if fancy:
                return ''.join(lo_vecs_to_show)
            else:
                return lo_vecs_to_show

    def get_M(self, system='NED', style='n'):
        """
        Returns the moment tensor in matrix representation.

        Call with arguments to set ouput in other basis system or in fancy
        style (to be viewed with 'print')
        """
        if style == 'f':
            print('\n   Full moment tensor in %s-coordinates:' % (system))
            return self._matrix_w_style_and_system(self._M, system, style)
        else:
            return self._matrix_w_style_and_system(self._M, system, style)

    def get_decomposition(self, in_system='NED', out_system='NED', style='n'):
        """
        Returns a tuple of the decomposition results.

        Order:
        - 1 - basis of the provided input     (string)
        - 2 - basis of  the representation    (string)
        - 3 - chosen decomposition type      (integer)

        - 4 - full moment tensor              (matrix)

        - 5 - isotropic part                  (matrix)
        - 6 - isotropic percentage             (float)
        - 7 - deviatoric part                 (matrix)
        - 8 - deviatoric percentage            (float)

        - 9 - DC part                         (matrix)
        -10 - DC percentage                    (float)
        -11 - DC2 part                        (matrix)
        -12 - DC2 percentage                   (float)
        -13 - DC3 part                        (matrix)
        -14 - DC3 percentage                   (float)

        -15 - CLVD part                       (matrix)
        -16 - CLVD percentage                 (matrix)

        -17 - seismic moment                   (float)
        -18 - moment magnitude                 (float)

        -19 - eigenvectors                   (3-array)
        -20 - eigenvalues                       (list)
        -21 - p-axis                         (3-array)
        -22 - neutral axis                   (3-array)
        -23 - t-axis                         (3-array)
        -24 - faultplanes       (list of two 3-arrays)
        """
        return [in_system, out_system, self.get_decomp_type(),
                self.get_M(system=out_system),
                self.get_iso(system=out_system), self.get_iso_percentage(),
                self.get_devi(system=out_system), self.get_devi_percentage(),
                self.get_DC(system=out_system), self.get_DC_percentage(),
                self.get_DC2(system=out_system), self.get_DC2_percentage(),
                self.get_DC3(system=out_system), self.get_DC3_percentage(),
                self.get_CLVD(system=out_system), self.get_CLVD_percentage(),
                self.get_moment(), self.get_mag(),
                self.get_eigvecs(system=out_system),
                self.get_eigvals(system=out_system),
                self.get_p_axis(system=out_system),
                self.get_null_axis(system=out_system),
                self.get_t_axis(system=out_system),
                self.get_fps()]

    def get_full_decomposition(self):
        """
        Nice compilation of decomposition result to be viewed in the shell
        (call with 'print').
        """
        mexp = pow(10, np.ceil(np.log10(np.max(np.abs(self._M)))))
        m = self._M / mexp
        s = '\nScalar Moment: M0 = %g Nm (Mw = %3.1f)\n'
        s += 'Moment Tensor: Mnn = %6.3f,  Mee = %6.3f, Mdd = %6.3f,\n'
        s += '               Mne = %6.3f,  Mnd = %6.3f, Med = %6.3f    '
        s += '[ x %g ]\n\n'
        s = s % (self._seismic_moment, self._moment_magnitude, m[0, 0],
                 m[1, 1], m[2, 2], m[0, 1], m[0, 2], m[1, 2], mexp)
        s += self._fault_planes_as_str()
        return s

    def _fault_planes_as_str(self):
        """
        Internal setup of a nice string, containing information about the fault
        planes.
        """
        s = '\n'
        for i, sdr in enumerate(self.get_fps()):
            s += 'Fault plane %i: ' % (i + 1)
            s += 'strike = %3.0f°, dip = %3.0f°, slip-rake = %4.0f°\n' % \
                 (sdr[0], sdr[1], sdr[2])
        return s

    def get_input_system(self, style='n', **kwargs):
        """
        Returns the basis system of the input.
        """
        if style == 'f':
            print('\n Basis system of the input:\n   ')
        return self._input_basis

    def get_output_system(self, style='n', **kwargs):
        """
        Returns the basis system of the input.
        """
        if style == 'f':
            print('\n Basis system of the output: \n  ')
        return self._output_basis

    def get_decomp_type(self, style='n', **kwargs):
        """
        Returns the decomposition type.
        """
        decomp_dict = dict(zip(('20', '21', '31'),
                               ('ISO + DC + CLVD',
                                'ISO + major DC + minor DC',
                                'ISO + DC1 + DC2 + DC3')))
        if style == 'f':
            print('\n Decomposition type: \n  ')
            return decomp_dict[str(self._decomposition_key)]

        return self._decomposition_key

    def get_iso(self, system='NED', style='n'):
        """
        Returns the isotropic part of the moment tensor in matrix
        representation.

        Call with arguments to set ouput in other basis system or in fancy
        style (to be viewed with 'print')
        """
        if style == 'f':
            print('\n Isotropic part in %s-coordinates: ' % (system))
        return self._matrix_w_style_and_system(self._isotropic, system, style)

    def get_devi(self, system='NED', style='n'):
        """
        Returns the deviatoric part of the moment tensor in matrix
        representation.

        Call with arguments to set ouput in other basis system or in fancy
        style (to be viewed with 'print')
        """
        if style == 'f':
            print('\n Deviatoric part in %s-coordinates: ' % (system))
        return self._matrix_w_style_and_system(self._deviatoric, system, style)

    def get_DC(self, system='NED', style='n'):
        """
        Returns the Double Couple part of the moment tensor in matrix
        representation.

        Call with arguments to set ouput in other basis system or in fancy
        style (to be viewed with 'print')
        """
        if style == 'f':
            print('\n Double Couple part in %s-coordinates:' % (system))
        return self._matrix_w_style_and_system(self._DC, system, style)

    def get_DC2(self, system='NED', style='n'):
        """
        Returns the second Double Couple part of the moment tensor in matrix
        representation.

        Call with arguments to set ouput in other basis system or in fancy
        style (to be viewed with 'print')
        """
        if style == 'f':
            print('\n second Double Couple part in %s-coordinates:' % (system))
        if self._DC2 is None:
            if style == 'f':
                print(' not available in this decomposition type ')
            return ''

        return self._matrix_w_style_and_system(self._DC2, system, style)

    def get_DC3(self, system='NED', style='n'):
        """
        Returns the third Double Couple part of the moment tensor in matrix
        representation.

        Call with arguments to set ouput in other basis system or in fancy
        style (to be viewed with 'print')
        """
        if style == 'f':
            print('\n third Double Couple part in %s-coordinates:' % (system))

        if self._DC3 is None:
            if style == 'f':
                print(' not available in this decomposition type ')
            return ''
        return self._matrix_w_style_and_system(self._DC3, system, style)

    def get_CLVD(self, system='NED', style='n'):
        """
        Returns the CLVD part of the moment tensor in matrix representation.

        Call with arguments to set ouput in other basis system or in fancy
        style (to be viewed with 'print')
        """
        if style == 'f':
            print('\n CLVD part in %s-coordinates: \n' % (system))
        if self._CLVD is None:
            if style == 'f':
                print(' not available in this decomposition type ')
            return ''

        return self._matrix_w_style_and_system(self._CLVD, system, style)

    def get_DC_percentage(self, system='NED', style='n'):
        """
        Returns the percentage of the DC part of the moment tensor in matrix
        representation.
        """
        if style == 'f':
            print('\n Double Couple percentage: \n')
        return self._DC_percentage

    def get_CLVD_percentage(self, system='NED', style='n'):
        """
        Returns the percentage of the DC part of the moment tensor in matrix
        representation.
        """
        if style == 'f':
            print('\n CLVD percentage: \n')
        if self._CLVD is None:
            if style == 'f':
                print(' not available in this decomposition type ')
            return ''
        return int(100 - self._DC_percentage - self._iso_percentage)

    def get_DC2_percentage(self, system='NED', style='n'):
        """
        Returns the percentage of the second DC part of the moment tensor in
        matrix representation.
        """
        if style == 'f':
            print("\n second Double Couple's percentage: \n")
        if self._DC2 is None:
            if style == 'f':
                print(' not available in this decomposition type ')
            return ''
        return self._DC2_percentage

    def get_DC3_percentage(self, system='NED', style='n'):
        """
        Returns the percentage of the third DC part of the moment tensor in
        matrix representation.
        """
        if style == 'f':
            print("\n third Double Couple percentage: \n")
        if self._DC3 is None:
            if style == 'f':
                print(' not available in this decomposition type ')
            return ''
        return int(100 - self._DC2_percentage - self._DC_percentage)

    def get_iso_percentage(self, system='NED', style='n'):
        """
        Returns the percentage of the isotropic part of the moment tensor in
        matrix representation.
        """
        if style == 'f':
            print('\n Isotropic percentage: \n')
        return self._iso_percentage

    def get_devi_percentage(self, system='NED', style='n'):
        """
        Returns the percentage of the deviatoric part of the moment tensor in
        matrix representation.
        """
        if style == 'f':
            print('\n Deviatoric percentage: \n')
        return int(100 - self._iso_percentage)

    def get_moment(self, system='NED', style='n'):
        """
        Returns the seismic moment (in Nm) of the moment tensor.
        """
        if style == 'f':
            print('\n Seismic moment (in Nm) : \n ')
        return self._seismic_moment

    def get_mag(self, system='NED', style='n'):
        """
        Returns the  moment magnitude M_w of the moment tensor.
        """
        if style == 'f':
            print('\n Moment magnitude Mw: \n ')
        return self._moment_magnitude

    def get_decomposition_key(self, system='NED', style='n'):
        """
        10 = standard decomposition (Jost & Herrmann)
        """
        if style == 'f':
            print('\n Decomposition key (standard = 10): \n ')
        return self._decomposition_key

    def get_eigvals(self, system='NED', style='n', **kwargs):
        """
        Returns a list of the eigenvalues of the moment tensor.
        """
        if style == 'f':
            if self._plot_clr_order < 0:
                print('\n    Eigenvalues T N P :\n')
            else:
                print('\n    Eigenvalues P N T :\n')
        # in the order HNS:
        return self._eigenvalues

    def get_eigvecs(self, system='NED', style='n'):
        """
        Returns the eigenvectors  of the moment tensor.

        Call with arguments to set ouput in other basis system or in fancy
        style (to be viewed with 'print')
        """
        if style == 'f':

            if self._plot_clr_order < 0:
                print('\n    Eigenvectors T N P (in basis system %s): ' %
                      (system))
            else:
                print('\n    Eigenvectors P N T (in basis system %s): ' %
                      (system))

        return self._vector_w_style_and_system(self._eigenvectors, system,
                                               style)

    def get_null_axis(self, system='NED', style='n'):
        """
        Returns the neutral axis of the moment tensor.

        Call with arguments to set ouput in other basis system or in fancy
        style (to be viewed with 'print')
        """
        if style == 'f':
            print('\n Null-axis in %s -coordinates: ' % (system))
        return self._vector_w_style_and_system(self._null_axis, system, style)

    def get_t_axis(self, system='NED', style='n'):
        """
        Returns the tension axis of the moment tensor.

        Call with arguments to set ouput in other basis system or in fancy
        style (to be viewed with 'print')
        """
        if style == 'f':
            print('\n Tension-axis in %s -coordinates: ' % (system))
        return self._vector_w_style_and_system(self._t_axis, system, style)

    def get_p_axis(self, system='NED', style='n'):
        """
        Returns the pressure axis of the moment tensor.

        Call with arguments to set ouput in other basis system or in fancy
        style (to be viewed with 'print')
        """
        if style == 'f':
            print('\n Pressure-axis in %s -coordinates: ' % (system))
        return self._vector_w_style_and_system(self._p_axis, system, style)

    def get_transform_matrix(self, system='NED', style='n'):
        """
        Returns the transformation matrix (input system to principal axis
        system.

        Call with arguments to set ouput in other basis system or in fancy
        style (to be viewed with 'print')
        """
        if style == 'f':
            print('\n rotation matrix in %s -coordinates: ' % (system))
        return self._matrix_w_style_and_system(self._rotation_matrix, system,
                                               style)

    def get_fps(self, **kwargs):
        """
        Returns a list of the two faultplane 3-tuples, each showing strike,
        dip, slip-rake.
        """
        fancy_key = kwargs.get('style', '0')
        if fancy_key[0].lower() == 'f':
            return self._fault_planes_as_str()
        else:
            return self._faultplanes

    def get_colour_order(self, **kwargs):
        """
        Returns the value of the plotting order (only important in BeachBall
        instances).
        """
        style = kwargs.get('style', '0')[0].lower()
        if style == 'f':
            print('\n Colour order key: ')
        return self._plot_clr_order


# ---------------------------------------------------------------
#
#   external functions:
#
# ---------------------------------------------------------------

def _puzzle_basis_transformation(mat_tup_arr_vec, in_basis, out_basis):
    lo_bases = ['NED', 'USE', 'XYZ', 'NWU']
    if (in_basis not in lo_bases) and (out_basis in lo_bases):
        sys.exit('wrong basis chosen')

    if in_basis == out_basis:
        transformed_in = mat_tup_arr_vec
    elif in_basis == 'NED':
        if out_basis == 'USE':
            transformed_in = NED2USE(mat_tup_arr_vec)
        if out_basis == 'XYZ':
            transformed_in = NED2XYZ(mat_tup_arr_vec)
        if out_basis == 'NWU':
            transformed_in = NED2NWU(mat_tup_arr_vec)
    elif in_basis == 'USE':
        if out_basis == 'NED':
            transformed_in = USE2NED(mat_tup_arr_vec)
        if out_basis == 'XYZ':
            transformed_in = USE2XYZ(mat_tup_arr_vec)
        if out_basis == 'NWU':
            transformed_in = USE2NWU(mat_tup_arr_vec)
    elif in_basis == 'XYZ':
        if out_basis == 'NED':
            transformed_in = XYZ2NED(mat_tup_arr_vec)
        if out_basis == 'USE':
            transformed_in = XYZ2USE(mat_tup_arr_vec)
        if out_basis == 'NWU':
            transformed_in = XYZ2NWU(mat_tup_arr_vec)
    elif in_basis == 'NWU':
        if out_basis == 'NED':
            transformed_in = NWU2NED(mat_tup_arr_vec)
        if out_basis == 'USE':
            transformed_in = NWU2USE(mat_tup_arr_vec)
        if out_basis == 'XYZ':
            transformed_in = NWU2XYZ(mat_tup_arr_vec)

    if len(mat_tup_arr_vec) == 3 and np.prod(np.shape(mat_tup_arr_vec)) != 9:
        tmp_array = np.array([0, 0, 0])
        tmp_array[:] = transformed_in
        return tmp_array
    else:
        return transformed_in


def _return_matrix_vector_array(ma_ve_ar, basis_change_matrix):
    """
    Generates the output for the functions, yielding matrices, vectors, and
    arrays in new basis systems.

    Allowed input are 3x3 matrices, 3-vectors, 3-vector collections,
    3-arrays, and 6-tuples.  Matrices are transformed directly,
    3-vectors the same.

    6-arrays are interpreted as 6 independent components of a moment
    tensor, so they are brought into symmetric 3x3 matrix form. This
    is transformed, and the 6 standard components 11,22,33,12,13,23
    are returned.
    """
    if (not np.prod(np.shape(ma_ve_ar)) in [3, 6, 9]) or \
       (not len(np.shape(ma_ve_ar)) in [1, 2]):
        print('\n wrong input - ', end=' ')
        print('provide either 3x3 matrix or 3-element vector \n')
        raise MTError(' !! ')

    if np.prod(np.shape(ma_ve_ar)) == 9:
        return np.dot(basis_change_matrix,
                      np.dot(ma_ve_ar, basis_change_matrix.T))
    elif np.prod(np.shape(ma_ve_ar)) == 6:
        m_in = ma_ve_ar
        orig_matrix = np.matrix([[m_in[0], m_in[3], m_in[4]],
                                [m_in[3], m_in[1], m_in[5]],
                                [m_in[4], m_in[5], m_in[2]]], dtype=np.float)
        m_out_mat = np.dot(basis_change_matrix,
                           np.dot(orig_matrix, basis_change_matrix.T))

        return m_out_mat[0, 0], m_out_mat[1, 1], m_out_mat[2, 2], \
            m_out_mat[0, 1], m_out_mat[0, 2], m_out_mat[1, 2]
    else:
        if np.shape(ma_ve_ar)[0] == 1:
            return np.dot(basis_change_matrix, ma_ve_ar.transpose())
        else:
            return np.dot(basis_change_matrix, ma_ve_ar)


def USE2NED(some_matrix_or_vector):
    """
    Function for basis transform from basis USE to NED.

    Input:
    3x3 matrix or 3-element vector or 6-element array in USE basis
    representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in NED basis
    representation
    """
    basis_change_matrix = np.matrix([[0., -1., 0.],
                                    [0., 0., 1.],
                                    [-1., 0., 0.]], dtype=np.float)
    return _return_matrix_vector_array(some_matrix_or_vector,
                                       basis_change_matrix)


def XYZ2NED(some_matrix_or_vector):
    """
    Function for basis transform from basis XYZ to NED.

    Input:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis
    representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in NED basis
    representation
    """
    basis_change_matrix = np.matrix([[0., 1., 0.],
                                    [1., 0., 0.],
                                    [0., 0., -1.]], dtype=np.float)
    return _return_matrix_vector_array(some_matrix_or_vector,
                                       basis_change_matrix)


def NWU2NED(some_matrix_or_vector):
    """
    Function for basis transform from basis NWU to NED.

    Input:
    3x3 matrix or 3-element vector or 6-element array in NWU basis
    representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in NED basis
    representation
    """
    basis_change_matrix = np.matrix([[1., 0., 0.],
                                    [0., -1., 0.],
                                    [0., 0., -1.]], dtype=np.float)
    return _return_matrix_vector_array(some_matrix_or_vector,
                                       basis_change_matrix)


def NED2USE(some_matrix_or_vector):
    """
    Function for basis transform from basis  NED to USE.

    Input:
    3x3 matrix or 3-element vector or 6-element array in NED basis
    representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in USE basis
    representation
    """
    basis_change_matrix = np.matrix([[0., -1., 0.],
                                    [0., 0., 1.],
                                    [-1., 0., 0.]], dtype=np.float).I
    return _return_matrix_vector_array(some_matrix_or_vector,
                                       basis_change_matrix)


def XYZ2USE(some_matrix_or_vector):
    """
    Function for basis transform from basis XYZ to USE.

    Input:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis
    representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in USE basis
    representation
    """
    basis_change_matrix = np.matrix([[0., 0., 1.],
                                    [0., -1., 0.],
                                    [1., 0., 0.]], dtype=np.float)
    return _return_matrix_vector_array(some_matrix_or_vector,
                                       basis_change_matrix)


def NED2XYZ(some_matrix_or_vector):
    """
    Function for basis transform from basis NED to XYZ.

    Input:
    3x3 matrix or 3-element vector or 6-element array in NED basis
    representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis
    representation
    """
    basis_change_matrix = np.matrix([[0., 1., 0.],
                                    [1., 0., 0.],
                                    [0., 0., -1.]], dtype=np.float).I
    return _return_matrix_vector_array(some_matrix_or_vector,
                                       basis_change_matrix)


def NED2NWU(some_matrix_or_vector):
    """
    Function for basis transform from basis NED to NWU.

    Input:
    3x3 matrix or 3-element vector or 6-element array in NED basis
    representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in NWU basis
    representation
    """
    basis_change_matrix = np.matrix([[1., 0., 0.],
                                    [0., -1., 0.],
                                    [0., 0., -1.]], dtype=np.float).I
    return _return_matrix_vector_array(some_matrix_or_vector,
                                       basis_change_matrix)


def USE2XYZ(some_matrix_or_vector):

    """
    Function for basis transform from basis USE to XYZ.

    Input:
    3x3 matrix or 3-element vector or 6-element array in USE basis
    representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis
    representation
    """
    basis_change_matrix = np.matrix([[0., 0., 1.],
                                    [0., -1., 0.],
                                    [1., 0., 0.]], dtype=np.float).I
    return _return_matrix_vector_array(some_matrix_or_vector,
                                       basis_change_matrix)


def NWU2XYZ(some_matrix_or_vector):

    """
    Function for basis transform from basis USE to XYZ.

    Input:
    3x3 matrix or 3-element vector or 6-element array in USE basis
    representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis
    representation
    """
    basis_change_matrix = np.matrix([[0., -1., 0.],
                                    [1., 0., 0.],
                                    [0., 0., 1.]], dtype=np.float)
    return _return_matrix_vector_array(some_matrix_or_vector,
                                       basis_change_matrix)


def NWU2USE(some_matrix_or_vector):

    """
    Function for basis transform from basis USE to XYZ.

    Input:
    3x3 matrix or 3-element vector or 6-element array in USE basis
    representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis
    representation
    """
    basis_change_matrix = np.matrix([[0., 0., 1.],
                                    [-1., 0., 0.],
                                    [0., -1., 0.]], dtype=np.float)
    return _return_matrix_vector_array(some_matrix_or_vector,
                                       basis_change_matrix)


def XYZ2NWU(some_matrix_or_vector):
    """
    Function for basis transform from basis USE to XYZ.

    Input:
    3x3 matrix or 3-element vector or 6-element array in USE basis
    representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis
    representation
    """
    basis_change_matrix = np.matrix([[0., -1., 0.],
                                    [1., 0., 0.],
                                    [0., 0., 1.]], dtype=np.float).I
    return _return_matrix_vector_array(some_matrix_or_vector,
                                       basis_change_matrix)


def USE2NWU(some_matrix_or_vector):
    """
    Function for basis transform from basis USE to XYZ.

    Input:
    3x3 matrix or 3-element vector or 6-element array in USE basis
    representation

    Output:
    3x3 matrix or 3-element vector or 6-element array in XYZ basis
    representation
    """
    basis_change_matrix = np.matrix([[0., 0., 1.],
                                    [-1., 0., 0.],
                                    [0., -1., 0.]], dtype=np.float).I
    return _return_matrix_vector_array(some_matrix_or_vector,
                                       basis_change_matrix)


def strikediprake_2_moments(strike, dip, rake):
    """
    angles are defined as in Jost&Herman (given in degrees)

    strike
        angle clockwise between north and plane ( in [0,360] )
    dip
        angle between surface and dipping plane ( in [0,90] ), 0 = horizontal,
        90 = vertical
    rake
        angle on the rupture plane between strike vector and actual movement
        (defined mathematically positive: ccw rotation is positive)

    basis for output is NED (= X,Y,Z)

    output:
    M = M_nn, M_ee, M_dd, M_ne, M_nd, M_ed
    """
    S_rad = strike / rad2deg
    D_rad = dip / rad2deg
    R_rad = rake / rad2deg

    for ang in S_rad, D_rad, R_rad:
        if abs(ang) < epsilon:
            ang = 0.

    M1 = -(np.sin(D_rad) * np.cos(R_rad) * np.sin(2 * S_rad) +
           np.sin(2 * D_rad) * np.sin(R_rad) * np.sin(S_rad) ** 2)
    M2 = (np.sin(D_rad) * np.cos(R_rad) * np.sin(2 * S_rad) -
          np.sin(2 * D_rad) * np.sin(R_rad) * np.cos(S_rad) ** 2)
    M3 = (np.sin(2 * D_rad) * np.sin(R_rad))
    M4 = (np.sin(D_rad) * np.cos(R_rad) * np.cos(2 * S_rad) +
          0.5 * np.sin(2 * D_rad) * np.sin(R_rad) * np.sin(2 * S_rad))
    M5 = -(np.cos(D_rad) * np.cos(R_rad) * np.cos(S_rad) +
           np.cos(2 * D_rad) * np.sin(R_rad) * np.sin(S_rad))
    M6 = -(np.cos(D_rad) * np.cos(R_rad) * np.sin(S_rad) -
           np.cos(2 * D_rad) * np.sin(R_rad) * np.cos(S_rad))

    Moments = [M1, M2, M3, M4, M5, M6]

    return tuple(Moments)


def fancy_matrix(m_in):
    """
    Returns a given 3x3 matrix or array in a cute way on the shell, if you
    use 'print' on the return value.
    """
    m = m_in.copy()

    norm_factor = round(np.max(np.abs(m.flatten())), 5)

    try:
        if (norm_factor < 0.1) or (norm_factor >= 10):
            if not abs(norm_factor) == 0:
                m = m / norm_factor
                out = "\n  / %5.2F %5.2F %5.2F \\\n" % \
                    (m[0, 0], m[0, 1], m[0, 2])
                out += "  | %5.2F %5.2F %5.2F  |   x  %F\n" % \
                    (m[1, 0], m[1, 1], m[1, 2], norm_factor)
                out += "  \\ %5.2F %5.2F %5.2F /\n" % \
                    (m[2, 0], m[2, 1], m[2, 2])
                return out
    except:
        pass

    return "\n  / %5.2F %5.2F %5.2F \\\n" % (m[0, 0], m[0, 1], m[0, 2]) + \
           "  | %5.2F %5.2F %5.2F  |\n" % (m[1, 0], m[1, 1], m[1, 2]) + \
           "  \\ %5.2F %5.2F %5.2F /\n" % (m[2, 0], m[2, 1], m[2, 2])


def fancy_vector(v):
    """
    Returns a given 3-vector or array in a cute way on the shell, if you
    use 'print' on the return value.
    """
    return "\n  / %5.2F \\\n" % (v[0]) + \
        "  | %5.2F  |\n" % (v[1]) + \
        "  \\ %5.2F /\n" % (v[2])


# ---------------------------------------------------------------
#
#   Class for plotting:
#
# ---------------------------------------------------------------

class BeachBall:
    """
    Class for generating a beachball projection for a provided moment tensor
    object.

    Input: a MomentTensor object

    Output can be plots of
    - the eigensystem
    - the complete sphere
    - the projection to a unit sphere
      .. either lower (standard) or upper half

    Beside the plots, the unit sphere projection may be saved in a given file.

    Alternatively, only the file can be provided without showing anything
    directly.
    """
    def __init__(self, MT=MomentTensor, kwargs_dict={}, npoints=360):
        self.MT = MT
        self._M = MT._M
        self._set_standard_attributes()
        self._update_attributes(kwargs_dict)
        self._plot_n_points = npoints
        self._nodallines_in_NED_system()
        self.arange_1 = np.arange(3 * npoints) - 1
        # self._identify_faultplanes()

    def ploBB(self, kwargs, ax=None):
        """
        Plots the projection of the beachball onto a unit sphere.
        """
        self._update_attributes(kwargs)
        self._setup_BB()
        self._plot_US(ax=ax)

    def save_BB(self, kwargs):
        """
        Saves the 2D projection of the beachball without plotting.

        :param outfile: name of outfile, addressing w.r.t. current directory
        :param format: if no implicit valid format is provided within the
            filename, add file format
        """
        self._update_attributes(kwargs)
        self._setup_BB()
        self._just_save_bb()

    def _just_save_bb(self):
        """
        Saves the beachball unit sphere plot into a given  file.
        """
        import matplotlib

        if self._plot_outfile_format == 'svg':
            try:
                matplotlib.use('SVG')
            except:
                matplotlib.use('Agg')
        elif self._plot_outfile_format == 'pdf':
            try:
                matplotlib.use('PDF')
            except:
                matplotlib.use('Agg')
                pass
        elif self._plot_outfile_format == 'ps':
            try:
                matplotlib.use('PS')
            except:
                matplotlib.use('Agg')
                pass
        elif self._plot_outfile_format == 'eps':
            try:
                matplotlib.use('Agg')
            except:
                matplotlib.use('PS')
                pass
        elif self._plot_outfile_format == 'png':
            try:
                matplotlib.use('AGG')
            except:
                mp_out = matplotlib.use('GTKCairo')
                if mp_out:
                    mp_out2 = matplotlib.use('Cairo')
                    if mp_out2:
                        matplotlib.use('GDK')

        import pylab as P

        plotfig = self._setup_plot_US(P)

        outfile_format = self._plot_outfile_format
        outfile_name = self._plot_outfile

        outfile_abs_name = os.path.realpath(
            os.path.abspath(os.path.join(os.curdir, outfile_name)))

        try:
            plotfig.savefig(outfile_abs_name, dpi=self._plot_dpi,
                            transparent=True, facecolor='k',
                            format=outfile_format)
        except:
            print('ERROR!! -- Saving of plot not possible')
            return
        P.close(667)
        del P
        del matplotlib

    def get_psxy(self, kwargs):
        """
        Returns one string, to be piped into psxy of GMT.

        :param GMT_type: fill/lines/EVs (select type of string),
            default is 'fill'
        :param GMT_scaling: scale the beachball - default radius is 1.0
        :param GMT_tension_colour: tension area of BB - colour flag for -Z in
            psxy, default is 1
        :param GMT_pressure_colour: pressure area of BB - colour flag for -Z in
            psxy, default is 0
        :param GMT_show_2FPs: flag, if both faultplanes are to be shown,
            default is 0
        :param GMT_show_1FP: flag, if one faultplane is to be shown, default
            is 1
        :param GMT_FP_index: 1 or 2, default is 2
        """
        self._GMT_type = 'fill'
        self._GMT_2fps = False
        self._GMT_1fp = 0

        self._GMT_psxy_fill = None
        self._GMT_psxy_nodals = None
        self._GMT_psxy_EVs = None
        self._GMT_scaling = 1.

        self._GMT_tension_colour = 1
        self._GMT_pressure_colour = 0

        self._update_attributes(kwargs)

        self._setup_BB()

        self._set_GMT_attributes()

        if self._GMT_type == 'fill':
            self._GMT_psxy_fill.seek(0)
            GMT_string = self._GMT_psxy_fill.getvalue()
        elif self._GMT_type == 'lines':
            self._GMT_psxy_nodals.seek(0)
            GMT_string = self._GMT_psxy_nodals.getvalue()
        else:
            GMT_string = self._GMT_psxy_EVs.getvalue()

        return GMT_string

    def _add_2_GMT_string(self, FH_string, curve, colour):
        """
        Writes coordinate pair list of given curve  as string into temporal
        file handler.
        """
        colour_Z = colour
        wstring = bytes('> -Z%i\n' % (colour_Z), encoding='utf-8')
        FH_string.write(wstring)
        np.savetxt(FH_string, self._GMT_scaling * curve.transpose())

    def _set_GMT_attributes(self):
        """
        Set the beachball lines and nodals as strings into a file handler.
        """
        neg_nodalline = self._nodalline_negative_final_US
        pos_nodalline = self._nodalline_positive_final_US
        FP1_2_plot = self._FP1_final_US
        FP2_2_plot = self._FP2_final_US
        EV_2_plot = self._all_EV_2D_US[:, :2].transpose()
        US = self._unit_sphere

        tension_colour = self._GMT_tension_colour
        pressure_colour = self._GMT_pressure_colour

        # build strings for possible GMT-output, used by 'psxy'
        GMT_string_FH = io.BytesIO()
        GMT_linestring_FH = io.BytesIO()
        GMT_EVs_FH = io.BytesIO()

        self._add_2_GMT_string(GMT_EVs_FH, EV_2_plot, tension_colour)
        GMT_EVs_FH.flush()

        if self._plot_clr_order > 0:
            self._add_2_GMT_string(GMT_string_FH, US, pressure_colour)
            self._add_2_GMT_string(GMT_string_FH, neg_nodalline,
                                   tension_colour)
            self._add_2_GMT_string(GMT_string_FH, pos_nodalline,
                                   tension_colour)
            GMT_string_FH.flush()

            if self._plot_curve_in_curve != 0:
                self._add_2_GMT_string(GMT_string_FH, US, tension_colour)

                if self._plot_curve_in_curve < 1:
                    self._add_2_GMT_string(GMT_string_FH, neg_nodalline,
                                           pressure_colour)
                    self._add_2_GMT_string(GMT_string_FH, pos_nodalline,
                                           tension_colour)
                    GMT_string_FH.flush()
                else:
                    self._add_2_GMT_string(GMT_string_FH, pos_nodalline,
                                           pressure_colour)
                    self._add_2_GMT_string(GMT_string_FH, neg_nodalline,
                                           tension_colour)
                    GMT_string_FH.flush()
        else:
            self._add_2_GMT_string(GMT_string_FH, US, tension_colour)
            self._add_2_GMT_string(GMT_string_FH, neg_nodalline,
                                   pressure_colour)
            self._add_2_GMT_string(GMT_string_FH, pos_nodalline,
                                   pressure_colour)
            GMT_string_FH.flush()

            if self._plot_curve_in_curve != 0:
                self._add_2_GMT_string(GMT_string_FH, US, pressure_colour)
                if self._plot_curve_in_curve < 1:
                    self._add_2_GMT_string(GMT_string_FH, neg_nodalline,
                                           tension_colour)
                    self._add_2_GMT_string(GMT_string_FH, pos_nodalline,
                                           pressure_colour)
                    GMT_string_FH.flush()
                else:
                    self._add_2_GMT_string(GMT_string_FH, pos_nodalline,
                                           tension_colour)
                    self._add_2_GMT_string(GMT_string_FH, neg_nodalline,
                                           pressure_colour)

                    GMT_string_FH.flush()

        # set all nodallines and faultplanes for plotting:
        self._add_2_GMT_string(GMT_linestring_FH, neg_nodalline,
                               tension_colour)
        self._add_2_GMT_string(GMT_linestring_FH, pos_nodalline,
                               tension_colour)

        if self._GMT_2fps:
            self._add_2_GMT_string(GMT_linestring_FH, FP1_2_plot,
                                   tension_colour)
            self._add_2_GMT_string(GMT_linestring_FH, FP2_2_plot,
                                   tension_colour)

        elif self._GMT_1fp:
            if not int(self._GMT_1fp) in [1, 2]:
                print('no fault plane specified for being plotted...continue',
                      end=' ')
                print('without fault plane(s)')
                pass
            else:
                if int(self._GMT_1fp) == 1:
                    self._add_2_GMT_string(GMT_linestring_FH, FP1_2_plot,
                                           tension_colour)
                else:
                    self._add_2_GMT_string(GMT_linestring_FH, FP2_2_plot,
                                           tension_colour)

        self._add_2_GMT_string(GMT_linestring_FH, US, tension_colour)

        GMT_linestring_FH.flush()

        setattr(self, '_GMT_psxy_nodals', GMT_linestring_FH)
        setattr(self, '_GMT_psxy_fill', GMT_string_FH)
        setattr(self, '_GMT_psxy_EVs', GMT_EVs_FH)

    def get_MT(self):
        """
        Returns the original moment tensor object, handed over to the class at
        generating this instance.
        """
        return self.MT

    def full_sphere_plot(self, kwargs):
        """
        Plot of the full beachball, projected on a circle with a radius 2.
        """
        self._update_attributes(kwargs)
        self._setup_BB()
        self._aux_plot()

    def _aux_plot(self):
        """
        Generates the final plot of the total sphere (according to the chosen
        2D-projection.
        """
        from matplotlib import interactive
        import pylab as P

        P.close('all')
        plotfig = P.figure(665, figsize=(self._plot_aux_plot_size,
                                         self._plot_aux_plot_size))

        plotfig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        ax = plotfig.add_subplot(111, aspect='equal')
        # P.axis([-1.1,1.1,-1.1,1.1],'equal')
        ax.axison = False

        EV_2_plot = getattr(self, '_all_EV' + '_final')
        BV_2_plot = getattr(self, '_all_BV' + '_final').transpose()
        curve_pos_2_plot = getattr(self, '_nodalline_positive' + '_final')
        curve_neg_2_plot = getattr(self, '_nodalline_negative' + '_final')
        FP1_2_plot = getattr(self, '_FP1' + '_final')
        FP2_2_plot = getattr(self, '_FP2' + '_final')

        tension_colour = self._plot_tension_colour
        pressure_colour = self._plot_pressure_colour

        if self._plot_clr_order > 0:
            if self._plot_fill_flag:

                alpha = self._plot_fill_alpha * self._plot_total_alpha
                ax.fill(self._outer_circle[0, :], self._outer_circle[1, :],
                        fc=pressure_colour, alpha=alpha)
                ax.fill(curve_pos_2_plot[0, :], curve_pos_2_plot[1, :],
                        fc=tension_colour, alpha=alpha)
                ax.fill(curve_neg_2_plot[0, :], curve_neg_2_plot[1, :],
                        fc=tension_colour, alpha=alpha)

                if self._plot_curve_in_curve != 0:
                    ax.fill(self._outer_circle[0, :], self._outer_circle[1, :],
                            fc=tension_colour, alpha=alpha)
                    if self._plot_curve_in_curve < 1:
                        ax.fill(curve_neg_2_plot[0, :], curve_neg_2_plot[1, :],
                                fc=pressure_colour, alpha=alpha)
                        ax.fill(curve_pos_2_plot[0, :], curve_pos_2_plot[1, :],
                                fc=tension_colour, alpha=alpha)
                    else:
                        ax.fill(curve_pos_2_plot[0, :], curve_pos_2_plot[1, :],
                                fc=pressure_colour, alpha=alpha)
                        ax.fill(curve_neg_2_plot[0, :], curve_neg_2_plot[1, :],
                                fc=tension_colour, alpha=alpha)

            if self._plot_show_princ_axes:
                alpha = self._plot_princ_axes_alpha * self._plot_total_alpha
                ax.plot([EV_2_plot[0, 0]], [EV_2_plot[1, 0]], 'm^',
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)
                ax.plot([EV_2_plot[0, 3]], [EV_2_plot[1, 3]], 'mv',
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)
                ax.plot([EV_2_plot[0, 1]], [EV_2_plot[1, 1]], 'b^',
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)
                ax.plot([EV_2_plot[0, 4]], [EV_2_plot[1, 4]], 'bv',
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)
                ax.plot([EV_2_plot[0, 2]], [EV_2_plot[1, 2]], 'g^',
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)
                ax.plot([EV_2_plot[0, 5]], [EV_2_plot[1, 5]], 'gv',
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)
        else:
            if self._plot_fill_flag:
                alpha = self._plot_fill_alpha * self._plot_total_alpha
                ax.fill(self._outer_circle[0, :], self._outer_circle[1, :],
                        fc=tension_colour, alpha=alpha)
                ax.fill(curve_pos_2_plot[0, :], curve_pos_2_plot[1, :],
                        fc=pressure_colour, alpha=alpha)
                ax.fill(curve_neg_2_plot[0, :], curve_neg_2_plot[1, :],
                        fc=pressure_colour, alpha=alpha)

                if self._plot_curve_in_curve != 0:
                    ax.fill(self._outer_circle[0, :], self._outer_circle[1, :],
                            fc=pressure_colour, alpha=alpha)
                    if self._plot_curve_in_curve < 0:
                        ax.fill(curve_neg_2_plot[0, :], curve_neg_2_plot[1, :],
                                fc=tension_colour, alpha=alpha)
                        ax.fill(curve_pos_2_plot[0, :], curve_pos_2_plot[1, :],
                                fc=pressure_colour, alpha=alpha)
                        pass
                    else:
                        ax.fill(curve_pos_2_plot[0, :], curve_pos_2_plot[1, :],
                                fc=tension_colour, alpha=alpha)
                        ax.fill(curve_neg_2_plot[0, :], curve_neg_2_plot[1, :],
                                fc=pressure_colour, alpha=alpha)
                        pass

            if self._plot_show_princ_axes:
                alpha = self._plot_princ_axes_alpha * self._plot_total_alpha
                ax.plot([EV_2_plot[0, 0]], [EV_2_plot[1, 0]], 'g^',
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)
                ax.plot([EV_2_plot[0, 3]], [EV_2_plot[1, 3]], 'gv',
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)
                ax.plot([EV_2_plot[0, 1]], [EV_2_plot[1, 1]], 'b^',
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)
                ax.plot([EV_2_plot[0, 4]], [EV_2_plot[1, 4]], 'bv',
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)
                ax.plot([EV_2_plot[0, 2]], [EV_2_plot[1, 2]], 'm^',
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)
                ax.plot([EV_2_plot[0, 5]], [EV_2_plot[1, 5]], 'mv',
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)

        self._plot_nodalline_colour = 'y'

        ax.plot(curve_neg_2_plot[0, :], curve_neg_2_plot[1, :], 'o',
                c=self._plot_nodalline_colour, lw=self._plot_nodalline_width,
                alpha=self._plot_nodalline_alpha * self._plot_total_alpha,
                ms=3)

        self._plot_nodalline_colour = 'b'

        ax.plot(curve_pos_2_plot[0, :], curve_pos_2_plot[1, :], 'D',
                c=self._plot_nodalline_colour, lw=self._plot_nodalline_width,
                alpha=self._plot_nodalline_alpha * self._plot_total_alpha,
                ms=3)

        if self._plot_show_1faultplane:
            if self._plot_show_FP_index == 1:
                ax.plot(FP1_2_plot[0, :], FP1_2_plot[1, :], '+',
                        c=self._plot_faultplane_colour,
                        lw=self._plot_faultplane_width,
                        alpha=self._plot_faultplane_alpha *
                        self._plot_total_alpha, ms=5)
            elif self._plot_show_FP_index == 2:
                ax.plot(FP2_2_plot[0, :], FP2_2_plot[1, :], '+',
                        c=self._plot_faultplane_colour,
                        lw=self._plot_faultplane_width,
                        alpha=self._plot_faultplane_alpha *
                        self._plot_total_alpha, ms=5)

        elif self._plot_show_faultplanes:
            ax.plot(FP1_2_plot[0, :], FP1_2_plot[1, :], '+',
                    c=self._plot_faultplane_colour,
                    lw=self._plot_faultplane_width,
                    alpha=self._plot_faultplane_alpha * self._plot_total_alpha,
                    ms=4)
            ax.plot(FP2_2_plot[0, :], FP2_2_plot[1, :], '+',
                    c=self._plot_faultplane_colour,
                    lw=self._plot_faultplane_width,
                    alpha=self._plot_faultplane_alpha * self._plot_total_alpha,
                    ms=4)
        else:
            pass

        # if isotropic part shall be displayed, fill the circle completely with
        # the appropriate colour
        if self._pure_isotropic:
            if abs(np.trace(self._M)) > epsilon:
                if self._plot_clr_order < 0:
                    ax.fill(self._outer_circle[0, :], self._outer_circle[1, :],
                            fc=tension_colour, alpha=1, zorder=100)
                else:
                    ax.fill(self._outer_circle[0, :], self._outer_circle[1, :],
                            fc=pressure_colour, alpha=1, zorder=100)

        # plot NED basis vectors
        if self._plot_show_basis_axes:
            plot_size_in_points = self._plot_size * 2.54 * 72
            points_per_unit = plot_size_in_points / 2.

            fontsize = plot_size_in_points / 66.
            symsize = plot_size_in_points / 77.

            direction_letters = 'NSEWDU'
            for idx, val in enumerate(BV_2_plot):
                x_coord = val[0]
                y_coord = val[1]
                np_letter = direction_letters[idx]

                rot_angle = -np.arctan2(y_coord, x_coord) + pi / 2.
                original_rho = np.sqrt(x_coord ** 2 + y_coord ** 2)

                marker_x = (original_rho - (3 * symsize / points_per_unit)) * \
                    np.sin(rot_angle)
                marker_y = (original_rho - (3 * symsize / points_per_unit)) * \
                    np.cos(rot_angle)
                annot_x = (original_rho - (8.5 * fontsize / points_per_unit)) \
                    * np.sin(rot_angle)
                annot_y = (original_rho - (8.5 * fontsize / points_per_unit)) \
                    * np.cos(rot_angle)

                ax.text(annot_x, annot_y, np_letter,
                        horizontalalignment='center', size=fontsize,
                        weight='bold', verticalalignment='center',
                        bbox=dict(edgecolor='white', facecolor='white',
                                  alpha=1))

                if original_rho > epsilon:
                    ax.scatter([marker_x], [marker_y],
                               marker=(3, 0, rot_angle), s=symsize ** 2, c='k',
                               facecolor='k', zorder=300)
                else:
                    ax.scatter([x_coord], [y_coord], marker=(4, 1, rot_angle),
                               s=symsize ** 2, c='k', facecolor='k',
                               zorder=300)

        # plot both circle lines (radius 1 and 2)
        ax.plot(self._unit_sphere[0, :], self._unit_sphere[1, :],
                c=self._plot_outerline_colour, lw=self._plot_outerline_width,
                alpha=self._plot_outerline_alpha * self._plot_total_alpha)
        ax.plot(self._outer_circle[0, :], self._outer_circle[1, :],
                c=self._plot_outerline_colour, lw=self._plot_outerline_width,
                alpha=self._plot_outerline_alpha * self._plot_total_alpha)

        # dummy points for setting plot plot size more accurately
        ax.plot([0, 2.1, 0, -2.1], [2.1, 0, -2.1, 0], ',', alpha=0.)

        ax.autoscale_view(tight=True, scalex=True, scaley=True)
        interactive(True)

        if self._plot_save_plot:
            try:
                plotfig.savefig(self._plot_outfile + '.' +
                                self._plot_outfile_format, dpi=self._plot_dpi,
                                transparent=True,
                                format=self._plot_outfile_format)
            except:
                print('saving of plot not possible')

        P.show()

    def pa_plot(self, kwargs):
        """
        Plot of the solution in the principal axes system.
        """
        import pylab as P

        self._update_attributes(kwargs)

        r_hor = self._r_hor_for_pa_plot
        r_hor_FP = self._r_hor_FP_for_pa_plot

        P.rc('grid', color='#316931', linewidth=0.5, linestyle='-.')
        P.rc('xtick', labelsize=12)
        P.rc('ytick', labelsize=10)

        width, height = P.rcParams['figure.figsize']
        size = min(width, height)

        fig = P.figure(34, figsize=(size, size))
        P.clf()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, axisbg='#d5de9c')

        r_steps = [0.000001]
        for i in (np.arange(4) + 1) * 0.2:
            r_steps.append(i)
        r_labels = ['S']
        for ii in range(len(r_steps)):
            if (ii + 1) % 2 == 0:
                r_labels.append(str(r_steps[ii]))
            else:
                r_labels.append(' ')

        t_angles = np.arange(0., 360., 90)
        t_labels = [' N ', ' H ', ' - N', ' - H']

        P.thetagrids(t_angles, labels=t_labels)

        ax.plot(self._phi_curve, r_hor, color='r', lw=3)
        ax.plot(self._phi_curve, r_hor_FP, color='b', lw=1.5)
        ax.set_rmax(1.0)
        P.grid(True)

        P.rgrids((r_steps), labels=r_labels)

        ax.set_title("beachball in eigenvector system", fontsize=15)

        if self._plot_save_plot:
            try:
                fig.savefig(self._plot_outfile + '.' +
                            self._plot_outfile_format, dpi=self._plot_dpi,
                            transparent=True,
                            format=self._plot_outfile_format)
            except:
                print('saving of plot not possible')
        P.show()

    def _set_standard_attributes(self):
        """
        Sets default values of mandatory arguments.
        """
        # plot basis system and view point:
        self._plot_basis = 'NED'
        self._plot_projection = 'stereo'
        self._plot_viewpoint = [0., 0., 0.]
        self._plot_basis_change = None

        # flag, if upper hemisphere is seen instead
        self._plot_show_upper_hemis = False

        # flag, if isotropic part shall be considered
        self._plot_isotropic_part = False
        self._pure_isotropic = False

        # number of minimum points per line and full circle (number/360 is
        # minimum of points per degree at rounded lines)
        self._plot_n_points = 360

        # nodal line of pressure and tension regimes:
        self._plot_nodalline_width = 2
        self._plot_nodalline_colour = 'k'
        self._plot_nodalline_alpha = 1.

        # outer circle line
        self._plot_outerline_width = 2
        self._plot_outerline_colour = 'k'
        self._plot_outerline_alpha = 1.

        # faultplane(s)
        self._plot_faultplane_width = 4
        self._plot_faultplane_colour = 'b'
        self._plot_faultplane_alpha = 1.

        self._plot_show_faultplanes = False
        self._plot_show_1faultplane = False
        self._plot_show_FP_index = 1

        # principal  axes:
        self._plot_show_princ_axes = False
        self._plot_princ_axes_symsize = 10
        self._plot_princ_axes_lw = 3
        self._plot_princ_axes_alpha = 0.5

        # NED basis:
        self._plot_show_basis_axes = False

        # filling of the area:
        self._plot_clr_order = self.MT.get_colour_order()
        self._plot_curve_in_curve = 0
        self._plot_fill_flag = True
        self._plot_tension_colour = 'r'
        self._plot_pressure_colour = 'w'
        self._plot_fill_alpha = 1.

        # general plot options
        self._plot_size = 5
        self._plot_aux_plot_size = 5
        self._plot_dpi = 200

        self._plot_total_alpha = 1.

        # possibility to add external data (e.g. measured polarizations)
        self._plot_external_data = False
        self._external_data = None

        # if, howto, whereto save the plot
        self._plot_save_plot = False
        self._plot_outfile = './BB_plot_example'
        self._plot_outfile_format = 'svg'

    def _update_attributes(self, kwargs):
        """
        Makes an internal update of the object's attributes with the
        provided list of keyword arguments.

        If the keyword (extended by a leading _ ) is in the dict of
        the object, the value is updated. Otherwise, the keyword is
        ignored.
        """
        for key in kwargs.keys():
            if key[0] == '_':
                kw = key[1:]
            else:
                kw = key
            if '_' + kw in dir(self):
                setattr(self, '_' + kw, kwargs[key])
        if kwargs.get('plot_only_lines', False):
            setattr(self, '_plot_fill_flag', False)

    def _setup_BB(self, unit_circle=True):
        """
        Setup of the beachball, when a plotting method is evoked.

        Contains all the technical stuff for generating the final view of the
        beachball:

        - Finding a rotation matrix, describing the given viewpoint onto the
          beachball projection
        - Rotating all elements (lines, points) w.r.t. the given viewpoint
        - Projecting the 3D sphere into the 2D plane
        - Building circle lines in radius r=1 and r=2
        - Correct the order of line points, yielding a consecutive set of
          points for drawing lines
        - Smoothing of all curves, avoiding nasty sectioning connection lines
        - Checking, if the two nodalline curves are laying completely within
          each other ( cahnges plotting order of overlay plot construction)
        - Projection of final smooth solution onto the standard unit sphere
        """
        self._find_basis_change_2_new_viewpoint()
        self._rotate_all_objects_2_new_view()
        self._vertical_2D_projection()

        if unit_circle:
            self._build_circles()

        if not self.MT._iso_percentage == 100:
            self._correct_curves()
            self._smooth_curves()
            self._check_curve_in_curve()

        self._projection_2_unit_sphere()

        if self.MT._iso_percentage == 100:
            if np.trace(self.MT.get_M()) < 0:
                self._plot_clr_order = 1
            else:
                self._plot_clr_order = -1

    def _correct_curves(self):
        """
        Correcting potentially wrong curves.

        Checks, if the order of the given coordinates of the lines must be
        re-arranged, allowing for an automatical line plotting.
        """
        list_of_curves_2_correct = ['nodalline_negative', 'nodalline_positive',
                                    'FP1', 'FP2']
        n_curve_points = self._plot_n_points

        for obj in list_of_curves_2_correct:
            obj2cor_name = '_' + obj + '_2D'
            obj2cor = getattr(self, obj2cor_name)

            obj2cor_in_right_order = self._sort_curve_points(obj2cor)

            # logger.debug( 'curve: ', str(obj))
            # check, if curve closed !!!!!!
            start_r = np.sqrt(obj2cor_in_right_order[0, 0] ** 2 +
                              obj2cor_in_right_order[1, 0] ** 2)
            r_last_point = np.sqrt(obj2cor_in_right_order[0, -1] ** 2 +
                                   obj2cor_in_right_order[1, -1] ** 2)
            dist_last_first_point = \
                np.sqrt((obj2cor_in_right_order[0, -1] -
                         obj2cor_in_right_order[0, 0]) ** 2 +
                        (obj2cor_in_right_order[1, -1] -
                         obj2cor_in_right_order[1, 0]) ** 2)

            # check, if distance between last and first point is smaller than
            # the distance between last point and the edge (at radius=2)
            if dist_last_first_point > (2 - r_last_point):
                # add points on edge to polygon, if it is an open curve
                # logger.debug( str(obj)+' not closed - closing over edge... ')
                phi_end = np.arctan2(obj2cor_in_right_order[0, -1],
                                     obj2cor_in_right_order[1, -1]) % (2 * pi)
                R_end = r_last_point
                phi_start = np.arctan2(obj2cor_in_right_order[0, 0],
                                       obj2cor_in_right_order[1, 0]) % (2 * pi)
                R_start = start_r

                # add one point on the edge every fraction of degree given by
                # input parameter, increase the radius linearly
                phi_end_larger = np.sign(phi_end - phi_start)
                angle_smaller_pi = np.sign(pi - np.abs(phi_end - phi_start))

                if phi_end_larger * angle_smaller_pi > 0:
                    go_ccw = True
                    openangle = (phi_end - phi_start) % (2 * pi)
                else:
                    go_ccw = False
                    openangle = (phi_start - phi_end) % (2 * pi)

                radius_interval = R_start - R_end  # closing from end to start

                n_edgepoints = int(openangle * rad2deg *
                                   n_curve_points / 360.) - 1
                # logger.debug( 'open angle %.2f degrees - filling with %i
                # points on the edge\n'%(openangle/pi*180,n_edgepoints))
                if go_ccw:
                    obj2cor_in_right_order = \
                        list(obj2cor_in_right_order.transpose())
                    for kk in range(n_edgepoints + 1):
                        current_phi = phi_end - kk * openangle / \
                            (n_edgepoints + 1)
                        current_radius = R_end + kk * radius_interval / \
                            (n_edgepoints + 1)
                        temp = [current_radius * math.sin(current_phi),
                                current_radius * np.cos(current_phi)]
                        obj2cor_in_right_order.append(temp)
                    obj2cor_in_right_order = \
                        np.array(obj2cor_in_right_order).transpose()
                else:
                    obj2cor_in_right_order = \
                        list(obj2cor_in_right_order.transpose())
                    for kk in range(n_edgepoints + 1):
                        current_phi = phi_end + kk * openangle / \
                            (n_edgepoints + 1)
                        current_radius = R_end + kk * radius_interval / \
                            (n_edgepoints + 1)
                        temp = [current_radius * math.sin(current_phi),
                                current_radius * np.cos(current_phi)]
                        obj2cor_in_right_order.append(temp)
                    obj2cor_in_right_order = \
                        np.array(obj2cor_in_right_order).transpose()
            setattr(self, '_' + obj + '_in_order', obj2cor_in_right_order)
        return 1

    def _nodallines_in_NED_system(self):
        """
        The two nodal lines between the areas on a beachball are given by the
        points, where tan²(alpha) = (-EWs/(EWN*cos(phi)**2 + EWh*sin(phi)**2))
        is fulfilled.

        This solution is gained in the principal axes system and then expressed
        in terms of the NED basis system

        output:
        - set of points, building the first nodal line,  coordinates in the
          input basis system (standard NED)
        - set of points, building the second nodal line,  coordinates in the
          input basis system (standard NED)
        - array with 6 points, describing positive and negative part of 3
          principal axes
        - array with partition of full circle (angle values in degrees)
          fraction is given by parameter n_curve_points
        """
        # build the nodallines of positive/negative areas in the principal axes
        # system
        n_curve_points = self._plot_n_points

        # phi is the angle between neutral axis and horizontal projection
        # of the curve point to the surface, spanned by H- and
        # N-axis. Running mathematically negative (clockwise) around the
        # SIGMA-axis. Stepsize is given by the parameter for number of
        # curve points
        phi = (np.arange(n_curve_points) / float(n_curve_points) +
               1. / n_curve_points) * 2 * pi
        self._phi_curve = phi

        # analytical/geometrical solution for separatrix curve - alpha is
        # opening angle between principal axis SIGMA and point of curve. (alpha
        # is 0, if curve lies directly on the SIGMA axis)

        # CASE: including isotropic part
        # sigma axis flips, if EWn flips sign

        EWh_devi = self.MT.get_eigvals()[0] - 1. / 3 * np.trace(self._M)
        EWn_devi = self.MT.get_eigvals()[1] - 1. / 3 * np.trace(self._M)
        EWs_devi = self.MT.get_eigvals()[2] - 1. / 3 * np.trace(self._M)

        if not self._plot_isotropic_part:
            EWh = EWh_devi
            EWn = EWn_devi
            EWs = EWs_devi
        else:
            EWh_tmp = self.MT.get_eigvals()[0]
            EWn_tmp = self.MT.get_eigvals()[1]
            EWs_tmp = self.MT.get_eigvals()[2]

            trace_m = np.sum(self.MT.get_eigvals())
            EWh = EWh_tmp.copy()
            EWs = EWs_tmp.copy()

            if trace_m != 0:
                if (self._plot_clr_order > 0 and EWn_tmp >= 0 and
                        abs(EWs_tmp) > abs(EWh_tmp)) or \
                        (self._plot_clr_order < 0 and
                         EWn_tmp <= 0 and abs(EWs_tmp) > abs(EWh_tmp)):

                    EWs = EWh_tmp.copy()
                    EWh = EWs_tmp.copy()
                    print('changed order!!\n')
                    EVs_tmp = self.MT._rotation_matrix[:, 2].copy()
                    EVh_tmp = self.MT._rotation_matrix[:, 0].copy()

                    self.MT._rotation_matrix[:, 0] = EVs_tmp
                    self.MT._rotation_matrix[:, 2] = EVh_tmp
                    self._plot_clr_order *= -1

            EWn = EWn_tmp.copy()

        if abs(EWn) < epsilon:
            EWn = 0
        norm_factor = max(np.abs([EWh, EWn, EWs]))

        # norm_factor is be zero in some cases
        with warnings.catch_warnings(record=True):
            np_err = np.seterr(all="warn")
            [EWh, EWn, EWs] = [xx / norm_factor for xx in [EWh, EWn, EWs]]
            np.seterr(**np_err)

        RHS = -EWs / (EWn * np.cos(phi) ** 2 + EWh * np.sin(phi) ** 2)

        if np.all([np.sign(xx) >= 0 for xx in RHS]):
            alpha = np.arctan(np.sqrt(RHS)) * rad2deg
        else:
            alpha = phi.copy()
            alpha[:] = 90
            self._pure_isotropic = 1

        # fault planes:
        RHS_FP = 1. / (np.sin(phi) ** 2)
        alpha_FP = np.arctan(np.sqrt(RHS_FP)) * rad2deg

        # horizontal coordinates of curves
        r_hor = np.sin(alpha / rad2deg)
        r_hor_FP = np.sin(alpha_FP / rad2deg)

        self._r_hor_for_pa_plot = r_hor
        self._r_hor_FP_for_pa_plot = r_hor_FP

        H_values = np.sin(phi) * r_hor
        N_values = np.cos(phi) * r_hor
        H_values_FP = np.sin(phi) * r_hor_FP
        N_values_FP = np.cos(phi) * r_hor_FP

        # set vertical value of curve point coordinates - two symmetric curves
        # exist
        S_values_positive = np.cos(alpha / rad2deg)
        S_values_negative = -np.cos(alpha / rad2deg)
        S_values_positive_FP = np.cos(alpha_FP / rad2deg)
        S_values_negative_FP = -np.cos(alpha_FP / rad2deg)

        #############
        # change basis back to original input reference system
        #########

        chng_basis = self.MT._rotation_matrix

        line_tuple_pos = np.zeros((3, n_curve_points))
        line_tuple_neg = np.zeros((3, n_curve_points))

        for ii in range(n_curve_points):
            pos_vec_in_EV_basis = np.array([H_values[ii], N_values[ii],
                                           S_values_positive[ii]]).transpose()
            neg_vec_in_EV_basis = np.array([H_values[ii], N_values[ii],
                                           S_values_negative[ii]]).transpose()
            line_tuple_pos[:, ii] = np.dot(chng_basis, pos_vec_in_EV_basis)
            line_tuple_neg[:, ii] = np.dot(chng_basis, neg_vec_in_EV_basis)

        EVh = self.MT.get_eigvecs()[0]
        EVn = self.MT.get_eigvecs()[1]
        EVs = self.MT.get_eigvecs()[2]

        all_EV = np.zeros((3, 6))

        all_EV[:, 0] = EVh.transpose()
        all_EV[:, 1] = EVn.transpose()
        all_EV[:, 2] = EVs.transpose()
        all_EV[:, 3] = -EVh.transpose()
        all_EV[:, 4] = -EVn.transpose()
        all_EV[:, 5] = -EVs.transpose()

        # basis vectors:
        all_BV = np.zeros((3, 6))
        all_BV[:, 0] = np.array((1, 0, 0))
        all_BV[:, 1] = np.array((-1, 0, 0))
        all_BV[:, 2] = np.array((0, 1, 0))
        all_BV[:, 3] = np.array((0, -1, 0))
        all_BV[:, 4] = np.array((0, 0, 1))
        all_BV[:, 5] = np.array((0, 0, -1))

        # re-sort the two 90 degree nodal lines to 2 fault planes - cut each at
        # halves and merge pairs
        # additionally change basis system to NED reference system

        midpoint_idx = int(n_curve_points / 2.)

        FP1 = np.zeros((3, n_curve_points))
        FP2 = np.zeros((3, n_curve_points))

        for ii in range(midpoint_idx):
            FP1_vec = np.array([H_values_FP[ii], N_values_FP[ii],
                               S_values_positive_FP[ii]]).transpose()
            FP2_vec = np.array([H_values_FP[ii], N_values_FP[ii],
                               S_values_negative_FP[ii]]).transpose()
            FP1[:, ii] = np.dot(chng_basis, FP1_vec)
            FP2[:, ii] = np.dot(chng_basis, FP2_vec)

        for jj in range(midpoint_idx):
            ii = n_curve_points - jj - 1

            FP1_vec = np.array([H_values_FP[ii], N_values_FP[ii],
                               S_values_negative_FP[ii]]).transpose()
            FP2_vec = np.array([H_values_FP[ii], N_values_FP[ii],
                               S_values_positive_FP[ii]]).transpose()
            FP1[:, ii] = np.dot(chng_basis, FP1_vec)
            FP2[:, ii] = np.dot(chng_basis, FP2_vec)

        # identify with faultplane index, gotten from 'get_fps':
        self._FP1 = FP1
        self._FP2 = FP2

        self._all_EV = all_EV
        self._all_BV = all_BV
        self._nodalline_negative = line_tuple_neg
        self._nodalline_positive = line_tuple_pos

    def _identify_faultplanes(self):
        """
        See, if the 2 faultplanes, given as attribute of the moment
        tensor object, handed to this instance, are consistent with
        the faultplane lines, obtained from the basis solution. If
        not, interchange the indices of the newly found ones.
        """
        # TODO !!!!!!
        pass

    def _find_basis_change_2_new_viewpoint(self):
        """
        Finding the Eulerian angles, if you want to rotate an object.

        Your original view point is the position (0,0,0). Input are the
        coordinates of the new point of view, equivalent to geographical
        coordinates.

        Example:

        Original view onto the Earth is from right above lat=0, lon=0 with
        north=upper edge, south=lower edge. Now you want to see the Earth
        from a position somewhere near Baku. So lat=45,
        lon=45, azimuth=0.

        The Earth must be rotated around some axis, not to be determined.
        The rotation matrixx is the matrix for the change of basis to the
        new local orthonormal system.

        input:
        - latitude in degrees from -90 (south) to 90 (north)
        - longitude in degrees from -180 (west) to 180 (east)
        - azimuth in degrees from 0 (heading north) to 360 (north again)
        """
        new_latitude = self._plot_viewpoint[0]
        new_longitude = self._plot_viewpoint[1]
        new_azimuth = self._plot_viewpoint[2]

        s_lat = np.sin(new_latitude / rad2deg)
        if abs(s_lat) < epsilon:
            s_lat = 0
        c_lat = np.cos(new_latitude / rad2deg)
        if abs(c_lat) < epsilon:
            c_lat = 0
        s_lon = np.sin(new_longitude / rad2deg)
        if abs(s_lon) < epsilon:
            s_lon = 0
        c_lon = np.cos(new_longitude / rad2deg)
        if abs(c_lon) < epsilon:
            c_lon = 0
        # assume input basis as NED!!!

        # original point of view therein is (0,0,-1)
        # new point at lat=latitude, lon=longitude, az=0, given in old
        # NED-coordinates:
        # (cos(latitude), sin(latitude)*sin(longitude),
        # sin(latitude)*cos(longitude) )
        #
        # new " down' " is given by the negative position vector, so pointing
        # inwards to the centre point
        # down_prime = - ( np.array( ( s_lat, c_lat*c_lon, -c_lat*s_lon ) ) )
        down_prime = -(np.array((s_lat, c_lat * s_lon, -c_lat * c_lon)))

        # normalise:
        down_prime /= np.sqrt(np.dot(down_prime, down_prime))

        # get second local basis vector " north' " by orthogonalising
        # (Gram-Schmidt method) the original north w.r.t. the new " down' "
        north_prime_not_normalised = np.array((1., 0., 0.)) - \
            (np.dot(down_prime, np.array((1., 0., 0.))) /
             (np.dot(down_prime, down_prime)) * down_prime)

        len_north_prime_not_normalised = \
            np.sqrt(np.dot(north_prime_not_normalised,
                           north_prime_not_normalised))
        # check for poles:
        if np.abs(len_north_prime_not_normalised) < epsilon:
            # case: north pole
            if s_lat > 0:
                north_prime = np.array((0., 0., 1.))
            # case: south pole
            else:
                north_prime = np.array((0., 0., -1.))
        else:
            north_prime = \
                north_prime_not_normalised / len_north_prime_not_normalised

        # third basis vector is obtained by a cross product of the first two
        east_prime = np.cross(down_prime, north_prime)

        # normalise:
        east_prime /= np.sqrt(np.dot(east_prime, east_prime))

        rotmat_pos_raw = np.zeros((3, 3))
        rotmat_pos_raw[:, 0] = north_prime
        rotmat_pos_raw[:, 1] = east_prime
        rotmat_pos_raw[:, 2] = down_prime

        rotmat_pos = np.asmatrix(rotmat_pos_raw).T
        # this matrix gives the coordinates of a given point in the old
        # coordinates w.r.t. the new system

        # up to here, only the position has changed, the angle of view
        # (azimuth) has to be added by an additional rotation around the
        # down'-axis (in the frame of the new coordinates)
        # set up the local rotation around the new down'-axis by the given
        # angle 'azimuth'. Positive values turn view counterclockwise from the
        # new north'
        only_rotation = np.zeros((3, 3))
        s_az = np.sin(new_azimuth / rad2deg)
        if abs(s_az) < epsilon:
            s_az = 0.
        c_az = np.cos(new_azimuth / rad2deg)
        if abs(c_az) < epsilon:
            c_az = 0.

        only_rotation[2, 2] = 1
        only_rotation[0, 0] = c_az
        only_rotation[1, 1] = c_az
        only_rotation[0, 1] = -s_az
        only_rotation[1, 0] = s_az

        local_rotation = np.asmatrix(only_rotation)

        # apply rotation from left!!
        total_rotation_matrix = np.dot(local_rotation, rotmat_pos)

        # yields the complete matrix for representing the old coordinates in
        # the new (rotated) frame:
        self._plot_basis_change = total_rotation_matrix

    def _rotate_all_objects_2_new_view(self):
        """
        Rotate all relevant parts of the solution - namely the
        eigenvector-projections, the 2 nodallines, and the faultplanes
        - so that they are seen from the new viewpoint.
        """
        objects_2_rotate = ['all_EV', 'all_BV', 'nodalline_negative',
                            'nodalline_positive', 'FP1', 'FP2']

        for obj in objects_2_rotate:
            object2rotate = getattr(self, '_' + obj).transpose()

            rotated_thing = object2rotate.copy()
            for i in range(len(object2rotate)):
                rotated_thing[i] = np.dot(self._plot_basis_change,
                                          object2rotate[i])

            rotated_object = rotated_thing.copy()
            setattr(self, '_' + obj + '_rotated', rotated_object.transpose())

    # ---------------------------------------------------------------

    def _vertical_2D_projection(self):
        """
        Start the vertical projection of the 3D beachball onto the 2D plane.

        The projection is chosen according to the attribute '_plot_projection'
        """
        list_of_possible_projections = ['stereo', 'ortho', 'lambert', 'gnom']

        if self._plot_projection not in list_of_possible_projections:
            print('desired projection not possible - choose from:\n ', end=' ')
            print(list_of_possible_projections)
            raise MTError(' !! ')

        if self._plot_projection == 'stereo':
            if not self._stereo_vertical():
                print('ERROR in stereo_vertical')
                raise MTError(' !! ')
        elif self._plot_projection == 'ortho':
            if not self._orthographic_vertical():
                print('ERROR in stereo_vertical')
                raise MTError(' !! ')
        elif self._plot_projection == 'lambert':
            if not self._lambert_vertical():
                print('ERROR in stereo_vertical')
                raise MTError(' !! ')
        elif self._plot_projection == 'gnom':
            if not self._gnomonic_vertical():
                print('ERROR in stereo_vertical')
                raise MTError(' !! ')

    def _stereo_vertical(self):
        """
        Stereographic/azimuthal conformal 2D projection onto a plane, tangent
        to the lowest point (0,0,1).

        Keeps the angles constant!

        The parts in the lower hemisphere are projected to the unit
        sphere, the upper half to an annular region between radii r=1
        and r=2. If the attribute '_show_upper_hemis' is set, the
        projection is reversed.
        """
        objects_2_project = ['all_EV', 'all_BV', 'nodalline_negative',
                             'nodalline_positive', 'FP1', 'FP2']

        available_coord_systems = ['NED']

        if self._plot_basis not in available_coord_systems:
            print('desired plotting projection not possible - choose from :\n',
                  end=' ')
            print(available_coord_systems)
            raise MTError(' !! ')

        plot_upper_hem = self._plot_show_upper_hemis

        for obj in objects_2_project:
            obj_name = '_' + obj + '_rotated'
            o2proj = getattr(self, obj_name)
            coords = o2proj.copy()

            n_points = len(o2proj[0, :])
            stereo_coords = np.zeros((2, n_points))

            for ll in range(n_points):
                # second component is EAST
                co_x = coords[1, ll]
                # first component is NORTH
                co_y = coords[0, ll]
                # z given in DOWN
                co_z = -coords[2, ll]

                rho_hor = np.sqrt(co_x ** 2 + co_y ** 2)

                if rho_hor == 0:
                    new_y = 0
                    new_x = 0
                    if plot_upper_hem:
                        if co_z < 0:
                            new_x = 2
                    else:
                        if co_z > 0:
                            new_x = 2
                else:
                    if co_z < 0:
                        new_rho = rho_hor / (1. - co_z)
                        if plot_upper_hem:
                            new_rho = 2 - (rho_hor / (1. - co_z))

                        new_x = co_x / rho_hor * new_rho
                        new_y = co_y / rho_hor * new_rho
                    else:
                        new_rho = 2 - (rho_hor / (1. + co_z))
                        if plot_upper_hem:
                            new_rho = rho_hor / (1. + co_z)

                        new_x = co_x / rho_hor * new_rho
                        new_y = co_y / rho_hor * new_rho

                stereo_coords[0, ll] = new_x
                stereo_coords[1, ll] = new_y

            setattr(self, '_' + obj + '_2D', stereo_coords)
            setattr(self, '_' + obj + '_final', stereo_coords)

        return 1

    def _orthographic_vertical(self):
        """
        Orthographic 2D projection onto a plane, tangent to the lowest
        point (0,0,1).

        Shows the natural view on a 2D sphere from large distances (assuming
        parallel projection)

        The parts in the lower hemisphere are projected to the unit
        sphere, the upper half to an annular region between radii r=1
        and r=2. If the attribute '_show_upper_hemis' is set, the
        projection is reversed.
        """

        objects_2_project = ['all_EV', 'all_BV', 'nodalline_negative',
                             'nodalline_positive', 'FP1', 'FP2']

        available_coord_systems = ['NED']

        if self._plot_basis not in available_coord_systems:
            print('desired plotting projection not possible - choose from :\n',
                  end=' ')
            print(available_coord_systems)
            raise MTError(' !! ')

        plot_upper_hem = self._plot_show_upper_hemis

        for obj in objects_2_project:
            obj_name = '_' + obj + '_rotated'
            o2proj = getattr(self, obj_name)
            coords = o2proj.copy()

            n_points = len(o2proj[0, :])
            coords2D = np.zeros((2, n_points))

            for ll in range(n_points):
                # second component is EAST
                co_x = coords[1, ll]
                # first component is NORTH
                co_y = coords[0, ll]
                # z given in DOWN
                co_z = -coords[2, ll]

                rho_hor = np.sqrt(co_x ** 2 + co_y ** 2)

                if rho_hor == 0:
                    new_y = 0
                    new_x = 0
                    if plot_upper_hem:
                        if co_z < 0:
                            new_x = 2
                    else:
                        if co_z > 0:
                            new_x = 2
                else:
                    if co_z < 0:
                        new_rho = rho_hor
                        if plot_upper_hem:
                            new_rho = 2 - rho_hor

                        new_x = co_x / rho_hor * new_rho
                        new_y = co_y / rho_hor * new_rho
                    else:
                        new_rho = 2 - rho_hor
                        if plot_upper_hem:
                            new_rho = rho_hor

                        new_x = co_x / rho_hor * new_rho
                        new_y = co_y / rho_hor * new_rho

                coords2D[0, ll] = new_x
                coords2D[1, ll] = new_y

            setattr(self, '_' + obj + '_2D', coords2D)
            setattr(self, '_' + obj + '_final', coords2D)

        return 1

    def _lambert_vertical(self):
        """
        Lambert azimuthal equal-area 2D projection onto a plane, tangent to the
        lowest point (0,0,1).

        Keeps the area constant!

        The parts in the lower hemisphere are projected to the unit
        sphere (only here the area is kept constant), the upper half to an
        annular region between radii r=1 and r=2. If the attribute
        '_show_upper_hemis' is set, the projection is reversed.
        """
        objects_2_project = ['all_EV', 'all_BV', 'nodalline_negative',
                             'nodalline_positive', 'FP1', 'FP2']

        available_coord_systems = ['NED']

        if self._plot_basis not in available_coord_systems:
            print('desired plotting projection not possible - choose from :\n',
                  end=' ')
            print(available_coord_systems)
            raise MTError(' !! ')

        plot_upper_hem = self._plot_show_upper_hemis

        for obj in objects_2_project:
            obj_name = '_' + obj + '_rotated'
            o2proj = getattr(self, obj_name)
            coords = o2proj.copy()

            n_points = len(o2proj[0, :])
            coords2D = np.zeros((2, n_points))

            for ll in range(n_points):
                # second component is EAST
                co_x = coords[1, ll]
                # first component is NORTH
                co_y = coords[0, ll]
                # z given in DOWN
                co_z = -coords[2, ll]

                rho_hor = np.sqrt(co_x ** 2 + co_y ** 2)

                if rho_hor == 0:
                    new_y = 0
                    new_x = 0
                    if plot_upper_hem:
                        if co_z < 0:
                            new_x = 2
                    else:
                        if co_z > 0:
                            new_x = 2
                else:
                    if co_z < 0:
                        new_rho = rho_hor / np.sqrt(1. - co_z)

                        if plot_upper_hem:
                            new_rho = 2 - (rho_hor / np.sqrt(1. - co_z))

                        new_x = co_x / rho_hor * new_rho
                        new_y = co_y / rho_hor * new_rho

                    else:
                        new_rho = 2 - (rho_hor / np.sqrt(1. + co_z))

                        if plot_upper_hem:
                            new_rho = rho_hor / np.sqrt(1. + co_z)

                        new_x = co_x / rho_hor * new_rho
                        new_y = co_y / rho_hor * new_rho

                coords2D[0, ll] = new_x
                coords2D[1, ll] = new_y

            setattr(self, '_' + obj + '_2D', coords2D)
            setattr(self, '_' + obj + '_final', coords2D)

        return 1

    def _gnomonic_vertical(self):
        """
        Gnomonic 2D projection onto a plane, tangent to the lowest
        point (0,0,1).

        Keeps the great circles as straight lines (geodetics constant) !

        The parts in the lower hemisphere are projected to the unit
        sphere, the upper half to an annular region between radii r=1
        and r=2. If the attribute '_show_upper_hemis' is set, the
        projection is reversed.
        """

        objects_2_project = ['all_EV', 'all_BV', 'nodalline_negative',
                             'nodalline_positive', 'FP1', 'FP2']

        available_coord_systems = ['NED']

        if self._plot_basis not in available_coord_systems:
            print('desired plotting projection not possible - choose from :\n',
                  end=' ')
            print(available_coord_systems)
            raise MTError(' !! ')

        plot_upper_hem = self._plot_show_upper_hemis

        for obj in objects_2_project:
            obj_name = '_' + obj + '_rotated'
            o2proj = getattr(self, obj_name)
            coords = o2proj.copy()

            n_points = len(o2proj[0, :])
            coords2D = np.zeros((2, n_points))

            for ll in range(n_points):
                # second component is EAST
                co_x = coords[1, ll]
                # first component is NORTH
                co_y = coords[0, ll]
                # z given in DOWN
                co_z = -coords[2, ll]

                rho_hor = np.sqrt(co_x ** 2 + co_y ** 2)

                if rho_hor == 0:
                    new_y = 0
                    new_x = 0
                    if co_z > 0:
                        new_x = 2
                        if plot_upper_hem:
                            new_x = 0
                else:
                    if co_z < 0:
                        new_rho = np.cos(np.arcsin(rho_hor)) * \
                            np.tan(np.arcsin(rho_hor))

                        if plot_upper_hem:
                            new_rho = 2 - (np.cos(np.arcsin(rho_hor)) *
                                           np.tan(np.arcsin(rho_hor)))

                        new_x = co_x / rho_hor * new_rho
                        new_y = co_y / rho_hor * new_rho

                    else:
                        new_rho = 2 - (np.cos(np.arcsin(rho_hor)) *
                                       np.tan(np.arcsin(rho_hor)))

                        if plot_upper_hem:
                            new_rho = np.cos(np.arcsin(rho_hor)) * \
                                np.tan(np.arcsin(rho_hor))

                        new_x = co_x / rho_hor * new_rho
                        new_y = co_y / rho_hor * new_rho

                coords2D[0, ll] = new_x
                coords2D[1, ll] = new_y

            setattr(self, '_' + obj + '_2D', coords2D)
            setattr(self, '_' + obj + '_final', coords2D)

        return 1

    def _build_circles(self):
        """
        Sets two sets of points, describing the unit sphere and the outer
        circle with r=2.

        Added as attributes '_unit_sphere' and '_outer_circle'.
        """
        phi = self._phi_curve

        UnitSphere = np.zeros((2, len(phi)))
        UnitSphere[0, :] = np.cos(phi)
        UnitSphere[1, :] = np.sin(phi)

        # outer circle ( radius for stereographic projection is set to 2 )
        outer_circle_points = 2 * UnitSphere

        self._unit_sphere = UnitSphere
        self._outer_circle = outer_circle_points

    def _sort_curve_points(self, curve):
        """
        Checks, if curve points are in right order for line plotting.

        If not, a re-arranging is carried out.
        """
        sorted_curve = np.zeros((2, len(curve[0, :])))
        # in polar coordinates
        r_phi_curve = np.zeros((len(curve[0, :]), 2))
        for ii in range(curve.shape[1]):
            r_phi_curve[ii, 0] = \
                math.sqrt(curve[0, ii] ** 2 + curve[1, ii] ** 2)
            r_phi_curve[ii, 1] = \
                math.atan2(curve[0, ii], curve[1, ii]) % (2 * pi)
        # find index with highest r
        largest_r_idx = np.argmax(r_phi_curve[:, 0])

        # check, if perhaps more values with same r - if so, take point with
        # lowest phi
        other_idces = \
            np.where(r_phi_curve[:, 0] == r_phi_curve[largest_r_idx, 0])
        if len(other_idces) > 1:
            best_idx = np.argmin(r_phi_curve[other_idces, 1])
            start_idx_curve = other_idces[best_idx]
        else:
            start_idx_curve = largest_r_idx

        if not start_idx_curve == 0:
            pass

        # check orientation - want to go inwards
        start_r = r_phi_curve[start_idx_curve, 0]
        next_idx = (start_idx_curve + 1) % len(r_phi_curve[:, 0])
        prep_idx = (start_idx_curve - 1) % len(r_phi_curve[:, 0])
        next_r = r_phi_curve[next_idx, 0]

        keep_direction = True
        if next_r <= start_r:
            # check, if next R is on other side of area - look at total
            # distance - if yes, reverse direction
            dist_first_next = \
                (curve[0, next_idx] - curve[0, start_idx_curve]) ** 2 + \
                (curve[1, next_idx] - curve[1, start_idx_curve]) ** 2
            dist_first_other = \
                (curve[0, prep_idx] - curve[0, start_idx_curve]) ** 2 + \
                (curve[1, prep_idx] - curve[1, start_idx_curve]) ** 2

            if dist_first_next > dist_first_other:
                keep_direction = False

        if keep_direction:
            # direction is kept
            for jj in range(curve.shape[1]):
                running_idx = (start_idx_curve + jj) % len(curve[0, :])
                sorted_curve[0, jj] = curve[0, running_idx]
                sorted_curve[1, jj] = curve[1, running_idx]
        else:
            # direction  is reversed
            for jj in range(curve.shape[1]):
                running_idx = (start_idx_curve - jj) % len(curve[0, :])
                sorted_curve[0, jj] = curve[0, running_idx]
                sorted_curve[1, jj] = curve[1, running_idx]

        # check if step of first to second point does not have large angle
        # step (problem caused by projection of point (pole) onto whole
        # edge - if this first angle step is larger than the one between
        # points 2 and three, correct position of first point: keep R, but
        # take angle with same difference as point 2 to point 3

        angle_point_1 = (math.atan2(sorted_curve[0, 0],
                                    sorted_curve[1, 0]) % (2 * pi))
        angle_point_2 = (math.atan2(sorted_curve[0, 1],
                                    sorted_curve[1, 1]) % (2 * pi))
        angle_point_3 = (math.atan2(sorted_curve[0, 2],
                                    sorted_curve[1, 2]) % (2 * pi))

        angle_diff_23 = (angle_point_3 - angle_point_2)
        if angle_diff_23 > pi:
            angle_diff_23 = (-angle_diff_23) % (2 * pi)

        angle_diff_12 = (angle_point_2 - angle_point_1)
        if angle_diff_12 > pi:
            angle_diff_12 = (-angle_diff_12) % (2 * pi)

        if abs(angle_diff_12) > abs(angle_diff_23):
            r_old = \
                math.sqrt(sorted_curve[0, 0] ** 2 + sorted_curve[1, 0] ** 2)
            new_angle = (angle_point_2 - angle_diff_23) % (2 * pi)
            sorted_curve[0, 0] = r_old * math.sin(new_angle)
            sorted_curve[1, 0] = r_old * math.cos(new_angle)

        return sorted_curve

    def _smooth_curves(self):
        """
        Corrects curves for potential large gaps, resulting in strange
        intersection lines on nodals of round and irreagularly shaped
        areas.

        At least one coordinte point on each degree on the circle is assured.
        """
        list_of_curves_2_smooth = ['nodalline_negative', 'nodalline_positive',
                                   'FP1', 'FP2']

        points_per_degree = self._plot_n_points / 360.

        for curve2smooth in list_of_curves_2_smooth:
            obj_name = curve2smooth + '_in_order'
            obj = getattr(self, '_' + obj_name).transpose()

            smoothed_array = np.zeros((1, 2))
            smoothed_array[0, :] = obj[0]
            smoothed_list = [smoothed_array]

            # now in shape (n_points,2)
            for idx, val in enumerate(obj[:-1]):
                r1 = math.sqrt(val[0] ** 2 + val[1] ** 2)
                r2 = math.sqrt(obj[idx + 1][0] ** 2 + obj[idx + 1][1] ** 2)
                phi1 = math.atan2(val[0], val[1])
                phi2 = math.atan2(obj[idx + 1][0], obj[idx + 1][1])

                phi2_larger = np.sign(phi2 - phi1)
                angle_smaller_pi = np.sign(pi - abs(phi2 - phi1))

                if phi2_larger * angle_smaller_pi > 0:
                    go_cw = True
                    openangle = (phi2 - phi1) % (2 * pi)
                else:
                    go_cw = False
                    openangle = (phi1 - phi2) % (2 * pi)

                openangle_deg = openangle * rad2deg
                radius_diff = r2 - r1

                if openangle_deg > 1. / points_per_degree:

                    n_fillpoints = int(openangle_deg * points_per_degree)
                    fill_array = np.zeros((n_fillpoints, 2))
                    if go_cw:
                        angles = ((np.arange(n_fillpoints) + 1) * openangle /
                                  (n_fillpoints + 1) + phi1) % (2 * pi)
                    else:
                        angles = (phi1 - (np.arange(n_fillpoints) + 1) *
                                  openangle / (n_fillpoints + 1)) % (2 * pi)

                    radii = (np.arange(n_fillpoints) + 1) * \
                        radius_diff / (n_fillpoints + 1) + r1

                    fill_array[:, 0] = radii * np.sin(angles)
                    fill_array[:, 1] = radii * np.cos(angles)

                    smoothed_list.append(fill_array)

                smoothed_list.append([obj[idx + 1]])

            smoothed_array = np.vstack(smoothed_list)
            setattr(self, '_' + curve2smooth + '_final',
                    smoothed_array.transpose())

    def _check_curve_in_curve(self):
        """
        Checks, if one of the two nodallines contains the other one
        completely. If so, the order of colours is re-adapted,
        assuring the correct order when doing the overlay plotting.
        """
        lo_points_in_pos_curve = \
            list(self._nodalline_positive_final.transpose())
        lo_points_in_pos_curve_array = \
            self._nodalline_positive_final.transpose()
        lo_points_in_neg_curve = \
            list(self._nodalline_negative_final.transpose())
        lo_points_in_neg_curve_array = \
            self._nodalline_negative_final.transpose()

        # check, if negative curve completely within positive curve
        mask_neg_in_pos = 0
        for neg_point in lo_points_in_neg_curve:
            mask_neg_in_pos += self._pnpoly(lo_points_in_pos_curve_array,
                                            neg_point[:2])
        if mask_neg_in_pos > len(lo_points_in_neg_curve) - 3:
            self._plot_curve_in_curve = 1

        # check, if positive curve completely within negative curve
        mask_pos_in_neg = 0
        for pos_point in lo_points_in_pos_curve:
            mask_pos_in_neg += self._pnpoly(lo_points_in_neg_curve_array,
                                            pos_point[:2])
        if mask_pos_in_neg > len(lo_points_in_pos_curve) - 3:
            self._plot_curve_in_curve = -1

        # correct for ONE special case: double couple with its
        # eigensystem = NED basis system:
        testarray = [1., 0, 0, 0, 1, 0, 0, 0, 1]
        if np.prod(self.MT._rotation_matrix.A1 == testarray) and \
           (self.MT._eigenvalues[1] == 0):
            self._plot_curve_in_curve = -1
            self._plot_clr_order = 1

    def _point_inside_polygon(self, x, y, poly):
        """
        Determine if a point is inside a given polygon or not.

        Polygon is a list of (x,y) pairs.
        """
        n = len(poly)
        inside = False

        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = \
                                (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _pnpoly(self, verts, point):
        """
        Check whether point is in the polygon defined by verts.

        verts - 2xN array
        point - (2,) array

        See
        http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
        """
        # using take instead of getitem, about ten times faster, see
        # http://wesmckinney.com/blog/?p=215
        verts = np.require(verts, dtype=np.float64)
        x, y = point

        xpi = verts[:, 0]
        ypi = verts[:, 1]
        # shift
        xpj = xpi.take(self.arange_1[:xpi.size])
        ypj = ypi.take(self.arange_1[:ypi.size])

        possible_crossings = \
            ((ypi <= y) & (y < ypj)) | ((ypj <= y) & (y < ypi))

        xpi = xpi[possible_crossings]
        ypi = ypi[possible_crossings]
        xpj = xpj[possible_crossings]
        ypj = ypj[possible_crossings]

        crossings = x < (xpj - xpi) * (y - ypi) / (ypj - ypi) + xpi

        return crossings.sum() % 2

    def _projection_2_unit_sphere(self):
        """
        Brings the complete solution (from stereographic projection)
        onto the unit sphere by just shrinking the maximum radius of
        all points to 1.

        This keeps the area definitions, so the colouring is not affected.
        """
        list_of_objects_2_project = ['nodalline_positive_final',
                                     'nodalline_negative_final']
        lo_fps = ['FP1_final', 'FP2_final']

        for obj2proj in list_of_objects_2_project:
            obj = getattr(self, '_' + obj2proj).transpose().copy()
            for idx, val in enumerate(obj):
                old_radius = np.sqrt(val[0] ** 2 + val[1] ** 2)
                if old_radius > 1:
                    obj[idx, 0] = val[0] / old_radius
                    obj[idx, 1] = val[1] / old_radius

            setattr(self, '_' + obj2proj + '_US', obj.transpose())

        for fp in lo_fps:
            obj = getattr(self, '_' + fp).transpose().copy()

            tmp_obj = []
            for idx, val in enumerate(obj):
                old_radius = np.sqrt(val[0] ** 2 + val[1] ** 2)
                if old_radius <= 1 + epsilon:
                    tmp_obj.append(val)
            tmp_obj2 = np.array(tmp_obj).transpose()
            tmp_obj3 = self._sort_curve_points(tmp_obj2)

            setattr(self, '_' + fp + '_US', tmp_obj3)

        lo_visible_EV = []

        for idx, val in enumerate(self._all_EV_2D.transpose()):
            r_ev = np.sqrt(val[0] ** 2 + val[1] ** 2)
            if r_ev <= 1:
                lo_visible_EV.append([val[0], val[1], idx])
        visible_EVs = np.array(lo_visible_EV)

        self._all_EV_2D_US = visible_EVs

        lo_visible_BV = []
        dummy_list1 = []
        direction_letters = 'NSEWDU'

        for idx, val in enumerate(self._all_BV_2D.transpose()):
            r_bv = math.sqrt(val[0] ** 2 + val[1] ** 2)
            if r_bv <= 1:
                if idx == 1 and 'N' in dummy_list1:
                    continue
                elif idx == 3 and 'E' in dummy_list1:
                    continue
                elif idx == 5 and 'D' in dummy_list1:
                    continue
                else:
                    lo_visible_BV.append([val[0], val[1], idx])
                    dummy_list1.append(direction_letters[idx])

        visible_BVs = np.array(lo_visible_BV)

        self._all_BV_2D_US = visible_BVs

    def _plot_US(self, ax=None):
        """
        Generates the final plot of the beachball projection on the unit
        sphere.

        Additionally, the plot can be saved in a file on the fly.
        """
        import pylab as P

        plotfig = self._setup_plot_US(P, ax=ax)

        if self._plot_save_plot:
            try:
                plotfig.savefig(self._plot_outfile + '.' +
                                self._plot_outfile_format, dpi=self._plot_dpi,
                                transparent=True,
                                format=self._plot_outfile_format)
            except:
                print('saving of plot not possible')
        P.show()
        P.close('all')

    def _setup_plot_US(self, P, ax=None):
        """
        Setting up the figure with the final plot of the unit sphere.

        Either called by _plot_US or by _just_save_bb
        """
        P.close(667)
        if ax is None:
            plotfig = P.figure(667, figsize=(self._plot_size, self._plot_size))
            plotfig.subplots_adjust(left=0, bottom=0, right=1, top=1)
            ax = plotfig.add_subplot(111, aspect='equal')

        ax.axison = False

        neg_nodalline = self._nodalline_negative_final_US
        pos_nodalline = self._nodalline_positive_final_US
        FP1_2_plot = self._FP1_final_US
        FP2_2_plot = self._FP2_final_US

        US = self._unit_sphere

        tension_colour = self._plot_tension_colour
        pressure_colour = self._plot_pressure_colour

        if self._plot_fill_flag:
            if self._plot_clr_order > 0:
                alpha = self._plot_fill_alpha * self._plot_total_alpha

                ax.fill(US[0, :], US[1, :], fc=pressure_colour, alpha=alpha)
                ax.fill(neg_nodalline[0, :], neg_nodalline[1, :],
                        fc=tension_colour, alpha=alpha)
                ax.fill(pos_nodalline[0, :], pos_nodalline[1, :],
                        fc=tension_colour, alpha=alpha)

                if self._plot_curve_in_curve != 0:
                    ax.fill(US[0, :], US[1, :], fc=tension_colour,
                            alpha=alpha)

                    if self._plot_curve_in_curve < 1:
                        ax.fill(neg_nodalline[0, :], neg_nodalline[1, :],
                                fc=pressure_colour, alpha=alpha)
                        ax.fill(pos_nodalline[0, :], pos_nodalline[1, :],
                                fc=tension_colour, alpha=alpha)
                        pass
                    else:
                        ax.fill(pos_nodalline[0, :], pos_nodalline[1, :],
                                fc=pressure_colour, alpha=alpha)
                        ax.fill(neg_nodalline[0, :], neg_nodalline[1, :],
                                fc=tension_colour, alpha=alpha)
                        pass

                EV_sym = ['m^', 'b^', 'g^', 'mv', 'bv', 'gv']

                if self._plot_show_princ_axes:
                    alpha = \
                        self._plot_princ_axes_alpha * self._plot_total_alpha

                    for val in self._all_EV_2D_US:
                        ax.plot([val[0]], [val[1]], EV_sym[int(val[2])],
                                ms=self._plot_princ_axes_symsize,
                                lw=self._plot_princ_axes_lw, alpha=alpha)

            else:
                alpha = self._plot_fill_alpha * self._plot_total_alpha

                ax.fill(US[0, :], US[1, :], fc=tension_colour, alpha=alpha)
                ax.fill(neg_nodalline[0, :], neg_nodalline[1, :],
                        fc=pressure_colour, alpha=alpha)
                ax.fill(pos_nodalline[0, :], pos_nodalline[1, :],
                        fc=pressure_colour, alpha=alpha)

                if self._plot_curve_in_curve != 0:
                    ax.fill(US[0, :], US[1, :], fc=pressure_colour,
                            alpha=alpha)

                    if self._plot_curve_in_curve < 1:
                        ax.fill(neg_nodalline[0, :], neg_nodalline[1, :],
                                fc=tension_colour, alpha=alpha)
                        ax.fill(pos_nodalline[0, :], pos_nodalline[1, :],
                                fc=pressure_colour, alpha=alpha)
                        pass
                    else:
                        ax.fill(pos_nodalline[0, :], pos_nodalline[1, :],
                                fc=tension_colour, alpha=alpha)
                        ax.fill(neg_nodalline[0, :], neg_nodalline[1, :],
                                fc=pressure_colour, alpha=alpha)
                        pass

        EV_sym = ['g^', 'b^', 'm^', 'gv', 'bv', 'mv']
        if self._plot_show_princ_axes:
            alpha = self._plot_princ_axes_alpha * self._plot_total_alpha

            for val in self._all_EV_2D_US:
                ax.plot([val[0]], [val[1]], EV_sym[int(val[2])],
                        ms=self._plot_princ_axes_symsize,
                        lw=self._plot_princ_axes_lw, alpha=alpha)

        #
        # set all nodallines and faultplanes for plotting:
        #

        ax.plot(neg_nodalline[0, :], neg_nodalline[1, :],
                c=self._plot_nodalline_colour, ls='-',
                lw=self._plot_nodalline_width,
                alpha=self._plot_nodalline_alpha * self._plot_total_alpha)
        # ax.plot( neg_nodalline[0,:] ,neg_nodalline[1,:],'go')

        ax.plot(pos_nodalline[0, :], pos_nodalline[1, :],
                c=self._plot_nodalline_colour, ls='-',
                lw=self._plot_nodalline_width,
                alpha=self._plot_nodalline_alpha * self._plot_total_alpha)

        if self._plot_show_faultplanes:
            ax.plot(FP1_2_plot[0, :], FP1_2_plot[1, :],
                    c=self._plot_faultplane_colour, ls='-',
                    lw=self._plot_faultplane_width,
                    alpha=self._plot_faultplane_alpha * self._plot_total_alpha)
            ax.plot(FP2_2_plot[0, :], FP2_2_plot[1, :],
                    c=self._plot_faultplane_colour, ls='-',
                    lw=self._plot_faultplane_width,
                    alpha=self._plot_faultplane_alpha * self._plot_total_alpha)

        elif self._plot_show_1faultplane:
            if self._plot_show_FP_index not in [1, 2]:
                print('no fault plane specified for being plotted... ',
                      end=' ')
                print('continue without faultplane')
                pass
            else:
                alpha = self._plot_faultplane_alpha * self._plot_total_alpha
                if self._plot_show_FP_index == 1:
                    ax.plot(FP1_2_plot[0, :], FP1_2_plot[1, :],
                            c=self._plot_faultplane_colour, ls='-',
                            lw=self._plot_faultplane_width, alpha=alpha)
                else:
                    ax.plot(FP2_2_plot[0, :], FP2_2_plot[1, :],
                            c=self._plot_faultplane_colour, ls='-',
                            lw=self._plot_faultplane_width, alpha=alpha)

        # if isotropic part shall be displayed, fill the circle completely with
        # the appropriate colour

        if self._pure_isotropic:
            # f abs( np.trace( self._M )) > epsilon:
            if self._plot_clr_order < 0:
                ax.fill(US[0, :], US[1, :], fc=tension_colour, alpha=1,
                        zorder=100)
            else:
                ax.fill(US[0, :], US[1, :], fc=pressure_colour, alpha=1,
                        zorder=100)

        # plot outer circle line of US
        ax.plot(US[0, :], US[1, :], c=self._plot_outerline_colour, ls='-',
                lw=self._plot_outerline_width,
                alpha=self._plot_outerline_alpha * self._plot_total_alpha)

        # plot NED basis vectors
        if self._plot_show_basis_axes:
            plot_size_in_points = self._plot_size * 2.54 * 72
            points_per_unit = plot_size_in_points / 2.

            fontsize = plot_size_in_points / 40.
            symsize = plot_size_in_points / 61.

            direction_letters = 'NSEWDU'

            for val in self._all_BV_2D_US:
                x_coord = val[0]
                y_coord = val[1]
                np_letter = direction_letters[int(val[2])]

                rot_angle = -np.arctan2(y_coord, x_coord) + pi / 2.
                original_rho = np.sqrt(x_coord ** 2 + y_coord ** 2)

                marker_x = (original_rho - (1.5 * symsize / points_per_unit)) \
                    * np.sin(rot_angle)
                marker_y = (original_rho - (1.5 * symsize / points_per_unit)) \
                    * np.cos(rot_angle)
                annot_x = (original_rho - (4.5 * fontsize / points_per_unit)) \
                    * np.sin(rot_angle)
                annot_y = (original_rho - (4.5 * fontsize / points_per_unit)) \
                    * np.cos(rot_angle)

                ax.text(annot_x, annot_y, np_letter,
                        horizontalalignment='center', size=fontsize,
                        weight='bold', verticalalignment='center',
                        bbox=dict(edgecolor='white', facecolor='white',
                                  alpha=1))

                if original_rho > epsilon:
                    ax.scatter([marker_x], [marker_y],
                               marker=(3, 0, rot_angle), s=symsize ** 2,
                               c='k', facecolor='k', zorder=300)
                else:
                    ax.scatter([x_coord], [y_coord], marker=(4, 1, rot_angle),
                               s=symsize ** 2, c='k', facecolor='k',
                               zorder=300)

        # plot 4 fake points, guaranteeing full visibility of the sphere
        ax.plot([0, 1.05, 0, -1.05], [1.05, 0, -1.05, 0], ',', alpha=0.)
        # scaling behavior
        ax.autoscale_view(tight=True, scalex=True, scaley=True)

        return plotfig


# -------------------------------------------------------------------
#
#  input and call management
#
# -------------------------------------------------------------------

def main(argv=None):
    """
    Usage

    obspy-mopad plot M
    obspy-mopad save M
    obspy-mopad gmt  M
    obspy-mopad convert M
    obspy-mopad decompose M
    """
    def _handle_input(M_in, args):
        """
        take the original method and its arguments, the source mechanism,
        and the dictionary with proper parsers for each call,
        """
        # construct a dict with consistent keyword args suited for the current
        # call
        kwargs = args.build(args)
        # set the fitting input basis system
        in_system = kwargs.get('in_system', 'NED')
        # build the moment tensor object
        mt = MomentTensor(M=M_in, system=in_system)
        # call the main routine to handle the moment tensor
        return args.call(mt, kwargs)

    def _call_plot(MT, kwargs_dict):
        """
        """
        bb2plot = BeachBall(MT, kwargs_dict)

        if kwargs_dict['plot_save_plot']:
            bb2plot.save_BB(kwargs_dict)
            return

        if kwargs_dict['plot_pa_plot']:
            bb2plot.pa_plot(kwargs_dict)
            return

        if kwargs_dict['plot_full_sphere']:
            bb2plot.full_sphere_plot(kwargs_dict)
            return

        bb2plot.ploBB(kwargs_dict)

        return

    def _call_convert(MT, kwargs_dict):
        """
        """
        if kwargs_dict['type_conversion']:

            if kwargs_dict['type_conversion'] == 'SDR':
                if kwargs_dict['fancy_conversion']:
                    return MT.get_fps(style='f')
                else:
                    return MT.get_fps(style='n')
            elif kwargs_dict['type_conversion'] == 'T':
                if kwargs_dict['basis_conversion']:
                    out_system = kwargs_dict['out_system']
                    if kwargs_dict['fancy_conversion']:
                        return MT.get_M(system=out_system, style='f')
                    else:
                        return MT.get_M(system=out_system, style='n')
                else:
                    if kwargs_dict['fancy_conversion']:
                        return MT.get_M(style='f')
                    else:
                        return MT.get_M(style='n')

        if kwargs_dict['basis_conversion']:
            if len(MT._original_M) in [6, 7]:
                if len(MT._original_M) == 6:
                    M_converted = _puzzle_basis_transformation(
                        MT.get_M(),
                        'NED', kwargs_dict['out_system'])
                    if kwargs_dict['fancy_conversion']:
                        print('\n  Moment tensor in basis  %s:\n ' %
                              (kwargs_dict['in_system']))
                        print(fancy_matrix(MT.get_M(
                              system=kwargs_dict['in_system'])))
                        print()
                        print('\n Moment tensor in basis  %s:\n ' %
                              (kwargs_dict['out_system']))
                        return fancy_matrix(M_converted)
                    else:
                        return M_converted[0, 0], M_converted[1, 1], \
                            M_converted[2, 2], M_converted[0, 1], \
                            M_converted[0, 2], M_converted[1, 2]
                else:
                    M_converted = _puzzle_basis_transformation(
                        MT.get_M(), 'NED', kwargs_dict['out_system'])
                    if kwargs_dict['fancy_conversion']:
                        print('\n  Moment tensor in basis  %s:\n ' %
                              (kwargs_dict['in_system']))
                        print(fancy_matrix(MT.get_M(
                              system=kwargs_dict['in_system'])))
                        print()
                        print('\n Moment tensor in basis  %s:\n ' %
                              (kwargs_dict['out_system']))
                        return fancy_matrix(M_converted)
                    else:
                        return M_converted[0, 0], M_converted[1, 1], \
                            M_converted[2, 2], M_converted[0, 1], \
                            M_converted[0, 2], M_converted[1, 2], \
                            MT._original_M[6]
            elif len(MT._original_M) == 9:
                new_M = np.asarray(MT._original_M).reshape(3, 3).copy()
                if kwargs_dict['fancy_conversion']:
                    return fancy_matrix(_puzzle_basis_transformation(
                        new_M, kwargs_dict['in_system'],
                        kwargs_dict['out_system']))
                else:
                    return _puzzle_basis_transformation(
                        new_M, kwargs_dict['in_system'],
                        kwargs_dict['out_system'])
            elif len(MT._original_M) == 3:
                M_converted = _puzzle_basis_transformation(
                    MT.get_M(), 'NED', kwargs_dict['out_system'])
                if kwargs_dict['fancy_conversion']:
                    print('\n  Moment tensor in basis  %s: ' %
                          (kwargs_dict['out_system']))
                    return fancy_matrix(M_converted)
                else:
                    return M_converted[0, 0], M_converted[1, 1], \
                        M_converted[2, 2], M_converted[0, 1], \
                        M_converted[0, 2], M_converted[1, 2]
            elif len(MT._original_M) == 4:
                M_converted = MT._original_M[3] * \
                    _puzzle_basis_transformation(MT.get_M(), 'NED',
                                                 kwargs_dict['out_system'])
                if kwargs_dict['fancy_conversion']:
                    print('\n  Momemnt tensor in basis  %s: ' %
                          (kwargs_dict['out_system']))
                    return fancy_matrix(M_converted)
                else:
                    return M_converted[0, 0], M_converted[1, 1], \
                        M_converted[2, 2], M_converted[0, 1], \
                        M_converted[0, 2], M_converted[1, 2]
            else:
                print('this try is meaningless - read the possible', end=' ')
                print('choices!\n(perhaps you want option "-v"(convert a',
                      end=' ')
                print('vector) or "-t"(convert strike, dip, rake to a matrix',
                      end=' ')
                print('and show THAT matrix in another basis system)', end=' ')
                print('instead!?!?)\n')
                sys.exit(-1)

        if kwargs_dict['vector_conversion']:
            if kwargs_dict['fancy_conversion']:
                print('\n  Vector in basis  %s:\n ' %
                      (kwargs_dict['vector_in_system']))
                print(fancy_vector(MT._original_M))
                print()
                print('\n  Vector in basis  %s:\n ' %
                      (kwargs_dict['vector_out_system']))
                return fancy_vector(_puzzle_basis_transformation(
                    MT._original_M, kwargs_dict['vector_in_system'],
                    kwargs_dict['vector_out_system']))
            else:
                return _puzzle_basis_transformation(
                    MT._original_M,
                    kwargs_dict['vector_in_system'],
                    kwargs_dict['vector_out_system'])
        else:
            msg = 'provide either option -t with one argument, or -b with '
            msg += 'two arguments, or -v with 2 arguments'
            sys.exit(msg)

    def _call_gmt(MT, kwargs_dict):
        """
        """
        bb = BeachBall(MT, kwargs_dict)
        return bb.get_psxy(kwargs_dict).decode('utf-8')

    def _call_decompose(MT, kwargs_dict):
        """
        """
        MT._isotropic = None
        MT._deviatoric = None
        MT._DC = None
        MT._iso_percentage = None
        MT._DC_percentage = None
        MT._DC2 = None
        MT._DC3 = None
        MT._DC2_percentage = None
        MT._CLVD = None
        MT._seismic_moment = None
        MT._moment_magnitude = None

        out_system = kwargs_dict['decomp_out_system']
        MT._output_basis = out_system
        MT._decomposition_key = kwargs_dict['decomposition_key']

        MT._decompose_M()

        # if total decomposition:
        if kwargs_dict['decomp_out_complete']:
            if kwargs_dict['decomp_out_fancy']:
                try:
                    print(MT.get_full_decomposition())
                except:
                    print(MT.get_full_decomposition().encode("utf-8"))
                return
            else:
                return MT.get_decomposition(in_system=kwargs_dict['in_system'],
                                            out_system=out_system)
        # otherwise:
        else:
            # argument dictionary - setting the appropriate calls
            do_calls = dict(zip(('in', 'out',
                                 'type',
                                 'full', 'm',
                                 'iso', 'iso_perc',
                                 'dev', 'devi', 'devi_perc',
                                 'dc', 'dc_perc',
                                 'dc2', 'dc2_perc',
                                 'dc3', 'dc3_perc',
                                 'clvd', 'clvd_perc',
                                 'mom', 'mag',
                                 'eigvals', 'eigvecs',
                                 't', 'n', 'p',
                                 'fps', 'faultplanes', 'fp',
                                 'decomp_key'),
                                ('input_system', 'output_system',
                                 'decomp_type',
                                 'M', 'M',
                                 'iso', 'iso_percentage',
                                 'devi', 'devi', 'devi_percentage',
                                 'DC', 'DC_percentage',
                                 'DC2', 'DC2_percentage',
                                 'DC3', 'DC3_percentage',
                                 'CLVD', 'CLVD_percentage',
                                 'moment', 'mag',
                                 'eigvals', 'eigvecs',
                                 't_axis', 'null_axis', 'p_axis',
                                 'fps', 'fps', 'fps',
                                 'decomp_type')
                                ))

            # build argument for local call within MT object:
            lo_args = kwargs_dict['decomp_out_part']

            # for single element output:
            if len(lo_args) == 1:
                if kwargs_dict['decomp_out_fancy']:
                    print(getattr(MT, 'get_' + do_calls[lo_args[0]])(
                        style='f', system=out_system))
                    return
                else:
                    return getattr(MT, 'get_' + do_calls[lo_args[0]])(
                        style='n', system=out_system)
            # for list of elements:
            else:
                outlist = []
                for arg in lo_args:
                    if kwargs_dict['decomp_out_fancy']:
                        print(getattr(MT, 'get_' + do_calls[arg])(
                            style='f', system=out_system))
                    else:
                        outlist.append(getattr(MT, 'get_' + do_calls[arg])(
                            style='n', system=out_system))
                if kwargs_dict['decomp_out_fancy']:
                    return
                else:
                    return outlist

    def _build_gmt_dict(args):
        """
        """
        consistent_kwargs_dict = {}

        if args.GMT_show_1FP:
            consistent_kwargs_dict['_GMT_1fp'] = args.GMT_show_1FP

        if args.GMT_show_2FP2:
            args.GMT_show_1FP = 0

            consistent_kwargs_dict['_GMT_2fps'] = True
            consistent_kwargs_dict['_GMT_1fp'] = 0

        if args.GMT_string_type[0] not in ['F', 'L', 'E']:
            print('type of desired string not known - taking "fill" instead')
            consistent_kwargs_dict['_GMT_type'] = 'fill'

        else:
            if args.GMT_string_type[0] == 'F':
                consistent_kwargs_dict['_GMT_type'] = 'fill'
            elif args.GMT_string_type[0] == 'L':
                consistent_kwargs_dict['_GMT_type'] = 'lines'
            else:
                consistent_kwargs_dict['_GMT_type'] = 'EVs'

        if args.GMT_scaling < epsilon:
            print('GMT scaling factor must be a factor larger than')
            print('%f - set to 1, due to obviously stupid input value' %
                  (epsilon))
            args.GMT_scaling = 1.0

        if args.plot_viewpoint:
            try:
                vp = args.plot_viewpoint.split(',')
                if not len(vp) == 3:
                    raise
                if not -90 <= float(vp[0]) <= 90:
                    raise
                if not -180 <= float(vp[1]) <= 180:
                    raise
                if not 0 <= float(vp[2]) % 360 <= 360:
                    raise
                consistent_kwargs_dict['plot_viewpoint'] = \
                    [float(vp[0]), float(vp[1]), float(vp[2])]
            except:
                pass

        if args.GMT_projection:
            lo_allowed_projections = ['STEREO', 'ORTHO', 'LAMBERT']  # ,'GNOM']
            do_allowed_projections = dict((x[0], x.lower()) for x in
                                          lo_allowed_projections)
            try:
                gmtp = args.GMT_projection
                if gmtp in lo_allowed_projections:
                    consistent_kwargs_dict['plot_projection'] = gmtp.lower()
                elif gmtp in do_allowed_projections.keys():
                    consistent_kwargs_dict['plot_projection'] = \
                        do_allowed_projections[gmtp]
                else:
                    consistent_kwargs_dict['plot_projection'] = 'stereo'
            except:
                pass

        consistent_kwargs_dict['_GMT_scaling'] = args.GMT_scaling
        consistent_kwargs_dict['_GMT_tension_colour'] = args.GMT_tension_colour
        consistent_kwargs_dict['_GMT_pressure_colour'] = \
            args.GMT_pressure_colour
        consistent_kwargs_dict['_plot_isotropic_part'] = \
            args.GMT_plot_isotropic_part

        return consistent_kwargs_dict

    def _build_decompose_dict(args):
        """
        """
        consistent_kwargs_dict = {}

        consistent_kwargs_dict['in_system'] = args.decomp_in_system

        consistent_kwargs_dict['decomp_out_system'] = args.decomp_out_system

        consistent_kwargs_dict['decomposition_key'] = args.decomp_key

        # if option 'partial' is not chosen, take complete decomposition:
        if not args.decomp_out_part:
            consistent_kwargs_dict['decomp_out_complete'] = 1
            consistent_kwargs_dict['decomp_out_part'] = 0

            if args.decomp_out_fancy:
                consistent_kwargs_dict['decomp_out_fancy'] = 1
            else:
                consistent_kwargs_dict['decomp_out_fancy'] = 0
        # otherwise take only partial  decomposition:
        else:
            lo_allowed_attribs = ['in', 'out', 'type',
                                  'full', 'm',
                                  'iso', 'iso_perc',
                                  'dev', 'devi', 'devi_perc',
                                  'dc', 'dc_perc',
                                  'dc2', 'dc2_perc',
                                  'dc3', 'dc3_perc',
                                  'clvd', 'clvd_perc',
                                  'mom', 'mag',
                                  'eigvals', 'eigvecs',
                                  't', 'n', 'p',
                                  'fps', 'faultplanes', 'fp',
                                  'decomp_key']
            lo_desired_attribs = args.decomp_out_part.split(',')
            lo_correct_attribs = []

            # check for allowed parts of decomposition:
            for da in lo_desired_attribs:
                if da.lower() in lo_allowed_attribs:
                    lo_correct_attribs.append(da.lower())

            # if only wrong or no arguments are handed over, change to complete
            # decomposition:
            if len(lo_correct_attribs) == 0:
                print(' no correct attributes for partial decomposition - '
                      'returning complete decomposition')
                consistent_kwargs_dict['decomp_out_complete'] = 1
                consistent_kwargs_dict['decomp_out_part'] = 0
                if args.decomp_out_fancy:
                    consistent_kwargs_dict['decomp_out_fancy'] = 1
                else:
                    consistent_kwargs_dict['decomp_out_fancy'] = 0

            # if only one part is desired to be shown, fancy style is possible
            elif len(lo_correct_attribs) == 1:
                consistent_kwargs_dict['decomp_out_complete'] = 0
                consistent_kwargs_dict['decomp_out_part'] = lo_correct_attribs
                if args.decomp_out_fancy:
                    consistent_kwargs_dict['decomp_out_fancy'] = 1
                else:
                    consistent_kwargs_dict['decomp_out_fancy'] = 0

            # if several parts are desired to be shown, fancy style is
            # NOT possible:
            else:
                consistent_kwargs_dict['decomp_out_complete'] = 0
                consistent_kwargs_dict['decomp_out_part'] = lo_correct_attribs
                if args.decomp_out_fancy:
                    consistent_kwargs_dict['decomp_out_fancy'] = 1
                else:
                    consistent_kwargs_dict['decomp_out_fancy'] = 0

        consistent_kwargs_dict['style'] = 'n'
        if consistent_kwargs_dict['decomp_out_fancy']:
            consistent_kwargs_dict['style'] = 'f'

        return consistent_kwargs_dict

    def _build_convert_dict(args):
        """
        """
        consistent_kwargs_dict = {}
        lo_allowed_options = ['type_conversion', 'basis_conversion',
                              'vector_conversion', 'fancy_conversion']
        # check for allowed options:
        for ao in lo_allowed_options:
            if hasattr(args, ao):
                consistent_kwargs_dict[ao] = getattr(args, ao)

        consistent_kwargs_dict['in_system'] = 'NED'
        if 'out_system' not in consistent_kwargs_dict:
            consistent_kwargs_dict['out_system'] = 'NED'

        if args.type_conversion and args.vector_conversion:
            print('decide for ONE option of "-t" OR "-v"')
            sys.exit(-1)

        if args.basis_conversion:
            consistent_kwargs_dict['in_system'] = args.basis_conversion[0]
            consistent_kwargs_dict['out_system'] = args.basis_conversion[1]

        if args.type_conversion and args.type_conversion == 'SDR':
            if args.basis_conversion:
                if args.basis_conversion[1] != 'NED':
                    print('output "sdr" from type conversion cannot be '
                          'displayed in another basis system!')
                    consistent_kwargs_dict['out_system'] = 'NED'

        if args.vector_conversion:
            consistent_kwargs_dict['vector_in_system'] = \
                args.vector_conversion[0]
            consistent_kwargs_dict['vector_out_system'] = \
                args.vector_conversion[1]

        return consistent_kwargs_dict

    def _build_plot_dict(args):
        """
        """
        consistent_kwargs_dict = {}

        consistent_kwargs_dict['plot_save_plot'] = False
        if args.plot_outfile:
            consistent_kwargs_dict['plot_save_plot'] = True
            lo_possible_formats = ['svg', 'png', 'eps', 'pdf', 'ps']

            try:
                (filepath, filename) = os.path.split(args.plot_outfile)
                if not filename:
                    filename = 'dummy_filename.svg'
                (shortname, extension) = os.path.splitext(filename)
                if not shortname:
                    shortname = 'dummy_shortname'

                if extension[1:].lower() in lo_possible_formats:
                    consistent_kwargs_dict['plot_outfile_format'] = \
                        extension[1:].lower()

                    if shortname.endswith('.'):
                        consistent_kwargs_dict['plot_outfile'] = \
                            os.path.realpath(os.path.abspath(os.path.join(
                                os.curdir, filepath,
                                shortname + extension[1:].lower())))
                    else:
                        consistent_kwargs_dict['plot_outfile'] = \
                            os.path.realpath(os.path.abspath(os.path.join(
                                os.curdir, filepath, shortname + '.' +
                                extension[1:].lower())))
                else:
                    if filename.endswith('.'):
                        consistent_kwargs_dict['plot_outfile'] = \
                            os.path.realpath(os.path.abspath(os.path.join(
                                os.curdir, filepath,
                                filename + lo_possible_formats[0])))
                    else:
                        consistent_kwargs_dict['plot_outfile'] = \
                            os.path.realpath(os.path.abspath(os.path.join(
                                os.curdir, filepath, filename + '.' +
                                lo_possible_formats[0])))
                    consistent_kwargs_dict['plot_outfile_format'] = \
                        lo_possible_formats[0]

            except:
                msg = 'please provide valid filename: <name>.<format>  !!\n'
                msg += ' <format> must be svg, png, eps, pdf, or ps '
                exit(msg)

        if args.plot_pa_plot:
            consistent_kwargs_dict['plot_pa_plot'] = True
        else:
            consistent_kwargs_dict['plot_pa_plot'] = False

        if args.plot_full_sphere:
            consistent_kwargs_dict['plot_full_sphere'] = True
            consistent_kwargs_dict['plot_pa_plot'] = False
        else:
            consistent_kwargs_dict['plot_full_sphere'] = False

        if args.plot_viewpoint:
            try:
                vp = args.plot_viewpoint.split(',')
                if not len(vp) == 3:
                    raise
                if not -90 <= float(vp[0]) <= 90:
                    raise
                if not -180 <= float(vp[1]) <= 180:
                    raise
                if not 0 <= float(vp[2]) % 360 <= 360:
                    raise
                consistent_kwargs_dict['plot_viewpoint'] = \
                    [float(vp[0]), float(vp[1]), float(vp[2])]
            except:
                pass

        if args.plot_projection:
            lo_allowed_projections = ['STEREO', 'ORTHO', 'LAMBERT']  # ,'GNOM']
            do_allowed_projections = dict((x[0], x.lower()) for x in
                                          lo_allowed_projections)
            try:
                ppl = args.plot_projection
                if ppl in lo_allowed_projections:
                    consistent_kwargs_dict['plot_projection'] = ppl.lower()
                elif ppl in do_allowed_projections.keys():
                    consistent_kwargs_dict['plot_projection'] = \
                        do_allowed_projections[ppl]
                else:
                    consistent_kwargs_dict['plot_projection'] = 'stereo'
            except:
                pass

        if args.plot_show_upper_hemis:
            consistent_kwargs_dict['plot_show_upper_hemis'] = True

        if args.plot_n_points and args.plot_n_points > 360:
            consistent_kwargs_dict['plot_n_points'] = args.plot_n_points

        if args.plot_size:
            try:
                if 0.01 < args.plot_size <= 1:
                    consistent_kwargs_dict['plot_size'] = \
                        args.plot_size * 10 / 2.54
                elif 1 < args.plot_size < 45:
                    consistent_kwargs_dict['plot_size'] = \
                        args.plot_size / 2.54
                else:
                    consistent_kwargs_dict['plot_size'] = 5
                consistent_kwargs_dict['plot_aux_plot_size'] = \
                    consistent_kwargs_dict['plot_size']
            except:
                pass

        if args.plot_pressure_colour:
            try:
                sec_colour_raw = args.plot_pressure_colour.split(',')
                if len(sec_colour_raw) == 1:
                    if sec_colour_raw[0].lower()[0] in 'bgrcmykw':
                        consistent_kwargs_dict['plot_pressure_colour'] = \
                            sec_colour_raw[0].lower()[0]
                    else:
                        raise
                elif len(sec_colour_raw) == 3:
                    for sc in sec_colour_raw:
                        if not 0 <= (int(sc)) <= 255:
                            raise
                    consistent_kwargs_dict['plot_pressure_colour'] = \
                        (float(sec_colour_raw[0]) / 255.,
                         float(sec_colour_raw[1]) / 255.,
                         float(sec_colour_raw[2]) / 255.)
                else:
                    raise
            except:
                pass

        if args.plot_tension_colour:
            try:
                sec_colour_raw = args.plot_tension_colour.split(',')
                if len(sec_colour_raw) == 1:
                    if sec_colour_raw[0].lower()[0] in 'bgrcmykw':
                        consistent_kwargs_dict['plot_tension_colour'] = \
                            sec_colour_raw[0].lower()[0]
                    else:
                        raise
                elif len(sec_colour_raw) == 3:
                    for sc in sec_colour_raw:
                        if not 0 <= (int(float(sc))) <= 255:
                            raise
                    consistent_kwargs_dict['plot_tension_colour'] = \
                        (float(sec_colour_raw[0]) / 255.,
                         float(sec_colour_raw[1]) / 255.,
                         float(sec_colour_raw[2]) / 255.)
                else:
                    raise
            except:
                pass

        if args.plot_total_alpha:
            if not 0 <= args.plot_total_alpha <= 1:
                consistent_kwargs_dict['plot_total_alpha'] = 1
            else:
                consistent_kwargs_dict['plot_total_alpha'] = \
                    args.plot_total_alpha

        if args.plot_show_1faultplane:
            consistent_kwargs_dict['plot_show_1faultplane'] = True
            try:
                fp_args = args.plot_show_1faultplane

                if not int(fp_args[0]) in [1, 2]:
                    consistent_kwargs_dict['plot_show_FP_index'] = 1
                else:
                    consistent_kwargs_dict['plot_show_FP_index'] = \
                        int(fp_args[0])

                if not 0 < float(fp_args[1]) <= 20:
                    consistent_kwargs_dict['plot_faultplane_width'] = 2
                else:
                    consistent_kwargs_dict['plot_faultplane_width'] = \
                        float(fp_args[1])

                try:
                    sec_colour_raw = fp_args[2].split(',')
                    if len(sec_colour_raw) == 1:
                        sc = sec_colour_raw[0].lower()[0]
                        if sc not in 'bgrcmykw':
                            raise
                        consistent_kwargs_dict['plot_faultplane_colour'] = \
                            sec_colour_raw[0].lower()[0]
                    elif len(sec_colour_raw) == 3:
                        for sc in sec_colour_raw:
                            if not 0 <= (int(sc)) <= 255:
                                raise
                        consistent_kwargs_dict['plot_faultplane_colour'] = \
                            (float(sec_colour_raw[0]) / 255.,
                             float(sec_colour_raw[1]) / 255.,
                             float(sec_colour_raw[2]) / 255.)
                    else:
                        raise
                except:
                    consistent_kwargs_dict['plot_faultplane_colour'] = 'k'

                try:
                    if 0 <= float(fp_args[3]) <= 1:
                        consistent_kwargs_dict['plot_faultplane_alpha'] = \
                            float(fp_args[3])
                except:
                    consistent_kwargs_dict['plot_faultplane_alpha'] = 1
            except:
                pass

        if args.plot_show_faultplanes:
            consistent_kwargs_dict['plot_show_faultplanes'] = True
            consistent_kwargs_dict['plot_show_1faultplane'] = False

        if args.plot_dpi:
            if 200 <= args.plot_dpi <= 2000:
                consistent_kwargs_dict['plot_dpi'] = args.plot_dpi

        if args.plot_only_lines:
            consistent_kwargs_dict['plot_fill_flag'] = False

        if args.plot_outerline:
            consistent_kwargs_dict['plot_outerline'] = True
            try:
                fp_args = args.plot_outerline
                if not 0 < float(fp_args[0]) <= 20:
                    consistent_kwargs_dict['plot_outerline_width'] = 2
                else:
                    consistent_kwargs_dict['plot_outerline_width'] = \
                        float(fp_args[0])
                try:
                    sec_colour_raw = fp_args[1].split(',')
                    if len(sec_colour_raw) == 1:
                        if sec_colour_raw[0].lower()[0] in 'bgrcmykw':
                            consistent_kwargs_dict['plot_outerline_colour'] = \
                                sec_colour_raw[0].lower()[0]
                        else:
                            raise
                    elif len(sec_colour_raw) == 3:
                        for sc in sec_colour_raw:
                            if not 0 <= (int(sc)) <= 255:
                                raise
                        consistent_kwargs_dict['plot_outerline_colour'] = \
                            (float(sec_colour_raw[0]) / 255.,
                             float(sec_colour_raw[1]) / 255.,
                             float(sec_colour_raw[2]) / 255.)
                    else:
                        raise
                except:
                    consistent_kwargs_dict['plot_outerline_colour'] = 'k'

                try:
                    if 0 <= float(fp_args[2]) <= 1:
                        consistent_kwargs_dict['plot_outerline_alpha'] = \
                            float(fp_args[2])
                except:
                    consistent_kwargs_dict['plot_outerline_alpha'] = 1
            except:
                pass

        if args.plot_nodalline:
            consistent_kwargs_dict['plot_nodalline'] = True
            try:
                fp_args = args.plot_nodalline

                if not 0 < float(fp_args[0]) <= 20:
                    consistent_kwargs_dict['plot_nodalline_width'] = 2
                else:
                    consistent_kwargs_dict['plot_nodalline_width'] = \
                        float(fp_args[0])
                try:
                    sec_colour_raw = fp_args[1].split(',')
                    if len(sec_colour_raw) == 1:
                        if sec_colour_raw[0].lower()[0] in 'bgrcmykw':
                            consistent_kwargs_dict['plot_nodalline_colour'] = \
                                sec_colour_raw[0].lower()[0]
                        else:
                            raise
                    elif len(sec_colour_raw) == 3:
                        for sc in sec_colour_raw:
                            if not 0 <= (int(sc)) <= 255:
                                raise
                        consistent_kwargs_dict['plot_nodalline_colour'] = \
                            (float(sec_colour_raw[0]) / 255.,
                             float(sec_colour_raw[1]) / 255.,
                             float(sec_colour_raw[2]) / 255.)
                    else:
                        raise
                except:
                    consistent_kwargs_dict['plot_nodalline_colour'] = 'k'
                try:
                    if 0 <= float(fp_args[2]) <= 1:
                        consistent_kwargs_dict['plot_nodalline_alpha'] = \
                            float(fp_args[2])
                except:
                    consistent_kwargs_dict['plot_nodalline_alpha'] = 1
            except:
                pass

        if args.plot_show_princ_axes:
            consistent_kwargs_dict['plot_show_princ_axes'] = True
            try:
                fp_args = args.plot_show_princ_axes

                if not 0 < float(fp_args[0]) <= 40:
                    consistent_kwargs_dict['plot_princ_axes_symsize'] = 10
                else:
                    consistent_kwargs_dict['plot_princ_axes_symsize'] = \
                        float(fp_args[0])

                if not 0 < float(fp_args[1]) <= 20:
                    consistent_kwargs_dict['plot_princ_axes_lw '] = 3
                else:
                    consistent_kwargs_dict['plot_princ_axes_lw '] = \
                        float(fp_args[1])
                try:
                    if 0 <= float(fp_args[2]) <= 1:
                        consistent_kwargs_dict['plot_princ_axes_alpha'] = \
                            float(fp_args[2])
                except:
                    consistent_kwargs_dict['plot_princ_axes_alpha'] = 1
            except:
                pass

        if args.plot_show_basis_axes:
            consistent_kwargs_dict['plot_show_basis_axes'] = True

        consistent_kwargs_dict['in_system'] = args.plot_input_system

        if args.plot_isotropic_part:
            consistent_kwargs_dict['plot_isotropic_part'] = \
                args.plot_isotropic_part

        return consistent_kwargs_dict

    def _build_parsers():
        """
        build dictionary with 4 (5 incl. 'save') sets of options, belonging to
        the 4 (5) possible calls
        """
        from argparse import (ArgumentParser,
                              RawDescriptionHelpFormatter,
                              RawTextHelpFormatter,
                              SUPPRESS)
        from obspy.core.util.base import _DeprecatedArgumentAction

        parser = ArgumentParser(prog='obspy-mopad',
                                formatter_class=RawDescriptionHelpFormatter,
                                description="""
###############################################################################
################################     MoPaD     ################################
################ Moment tensor Plotting and Decomposition tool ################
###############################################################################

Multi method tool for:

- Plotting and saving of focal sphere diagrams ('Beachballs').

- Decomposition and Conversion of seismic moment tensors.

- Generating coordinates, describing a focal sphere diagram, to be
  piped into GMT's psxy (Useful where psmeca or pscoupe fail.)

For more help, please run ``%(prog)s {command} --help''.

-------------------------------------------------------------------------------

Example:

To generate a beachball for a normal faulting mechanism (a snake's eye type):

    %(prog)s plot 0,45,-90   or   %(prog)s plot p 0,1,-1,0,0,0
            """)

        parser.add_argument('-V', '--version', action='version',
                            version='%(prog)s ' + __version__)

        mechanism = ArgumentParser(add_help=False,
                                   formatter_class=RawTextHelpFormatter)
        mechanism.add_argument('mechanism', metavar='source-mechanism',
                               help="""
The 'source mechanism' as a comma-separated list of length:

3:
    strike, dip, rake;
4:
    strike, dip, rake, moment;
6:
    M11, M22, M33, M12, M13, M23;
7:
    M11, M22, M33, M12, M13, M23, moment;
9:
    full moment tensor

(With all angles to be given in degrees)
                            """)

        subparsers = parser.add_subparsers(title='commands')

        # Case-insensitive typing
        class caps(str):
            def __new__(self, content):
                return str.__new__(self, content.upper())

        # Possible basis systems
        ALLOWED_BASES = ['NED', 'USE', 'XYZ', 'NWU']

        # gmt
        help = "return the beachball as a string, to be piped into GMT's psxy"
        desc = """Tool providing strings to be piped into the 'psxy' from GMT.

        Either a string describing the fillable area (to be used with option
        '-L' within psxy) or the nodallines or the coordinates of the principle
        axes are given.
        """

        parser_gmt = subparsers.add_parser('gmt', help=help, description=desc,
                                           parents=[mechanism])

        group_type = parser_gmt.add_argument_group('Output')
        group_show = parser_gmt.add_argument_group('Appearance')
        group_geo = parser_gmt.add_argument_group('Geometry')

        group_type.add_argument(
            '-t', '--type', dest='GMT_string_type', metavar='<type>',
            type=caps, default='FILL',
            help='choice of psxy data: area to fill (fill), nodal lines '
                 '(lines), or eigenvector positions (ev)')
        group_show.add_argument(
            '-s', '--scaling', dest='GMT_scaling', metavar='<scaling factor>',
            type=float, default=1.0,
            help='spatial scaling of the beachball')
        group_show.add_argument(
            '-r', '--color1', '--colour1', dest='GMT_tension_colour',
            metavar='<tension colour>', type=int, default=1,
            help="-Z option's key for the tension colour of the beachball - "
                 'type: integer')
        group_show.add_argument(
            '-w', '--color2', '--colour2', dest='GMT_pressure_colour',
            metavar='<pressure colour>', type=int, default=0,
            help="-Z option's key for the pressure colour of the beachball - "
                 'type: integer')
        group_show.add_argument(
            '-D', '--faultplanes', dest='GMT_show_2FP2', action='store_true',
            help='boolean key, if 2 faultplanes shall be shown')
        group_show.add_argument(
            '-d', '--show-1fp', dest='GMT_show_1FP', metavar='<FP index>',
            type=int, choices=[1, 2], default=False,
            help='integer key (1,2), what faultplane shall be shown '
                 '[%(default)s]')
        group_geo.add_argument(
            '-V', '--viewpoint', dest='plot_viewpoint',
            metavar='<lat,lon,azi>',
            help='coordinates of the viewpoint - 3-tuple of angles in degree')
        group_geo.add_argument(
            '-p', '--projection', dest='GMT_projection',
            metavar='<projection>', type=caps,
            help='projection of the sphere')
        group_show.add_argument(
            '-I', '--show-isotropic-part', dest='GMT_plot_isotropic_part',
            action='store_true',
            help='if isotropic part shall be considered for plotting '
                 '[%(default)s]')

        # Deprecated arguments

        action = _DeprecatedArgumentAction('--show_1fp', '--show-1fp')
        group_show.add_argument(
            '--show_1fp', dest='GMT_show_1FP', action=action, help=SUPPRESS)

        action = _DeprecatedArgumentAction('--show_isotropic_part',
                                           '--show-isotropic-part',
                                           real_action='store_true')
        group_show.add_argument(
            '--show_isotropic_part', dest='GMT_plot_isotropic_part', nargs=0,
            action=action, help=SUPPRESS)

        parser_gmt.set_defaults(call=_call_gmt, build=_build_gmt_dict)

        # convert
        help = 'convert a mechanism to/in (strike,dip,slip-rake) from/to ' \
               'matrix form *or* convert a matrix, vector, tuple into ' \
               'different basis representations'
        desc = """Tool providing converted input.

        Choose between the conversion from/to matrix-moment-tensor
        form (-t), the change of the output basis system for a given
        moment tensor (-b), or the change of basis for a 3D vector
        (-v).
        """
        parser_convert = subparsers.add_parser('convert', help=help,
                                               description=desc,
                                               parents=[mechanism])

        group_type = parser_convert.add_argument_group('Type conversion')
        group_basis = parser_convert.add_argument_group('M conversion')
        group_vector = parser_convert.add_argument_group('Vector conversion')
#        group_show = parser_convert.add_argument_group('Appearance')

        group_type.add_argument(
            '-t', '--type', dest='type_conversion',
            type=caps, choices=['SDR', 'T'],
            help='type conversion - convert to: strike,dip,rake (sdr) or '
                 'Tensor (T)')
        group_basis.add_argument(
            '-b', '--basis', dest='basis_conversion',
            type=caps, choices=ALLOWED_BASES, nargs=2,
            help='basis conversion for M - provide 2 arguments: input and '
                 'output bases')
        group_vector.add_argument(
            '-v', '--vector', dest='vector_conversion',
            type=caps, choices=ALLOWED_BASES, nargs=2,
            help='basis conversion for a vector - provide M as a 3Dvector '
                 'and 2 option-arguments of -v: input and output bases')
        parser_convert.add_argument(
            '-y', '--fancy', dest='fancy_conversion', action='store_true',
            help='output in a stylish way')

        parser_convert.set_defaults(call=_call_convert,
                                    build=_build_convert_dict)

        # plot
        help = 'plot a beachball projection of the provided mechanism'
        desc = """Plots a beachball diagram of the provided mechanism.

        Several styles and configurations are available. Also saving
        on the fly can be enabled.
        """
        parser_plot = subparsers.add_parser('plot', help=help,
                                            description=desc,
                                            parents=[mechanism])

        group_save = parser_plot.add_argument_group('Saving')
        group_type = parser_plot.add_argument_group('Type of plot')
        group_quality = parser_plot.add_argument_group('Quality')
        group_colours = parser_plot.add_argument_group('Colours')
        group_misc = parser_plot.add_argument_group('Miscellaneous')
        group_dc = parser_plot.add_argument_group('Fault planes')
        group_geo = parser_plot.add_argument_group('Geometry')
        group_app = parser_plot.add_argument_group('Appearance')

        group_save.add_argument(
            '-f', '--output-file', dest='plot_outfile', metavar='<filename>',
            help='filename for saving')
        group_type.add_argument(
            '-P', '--pa-system', dest='plot_pa_plot', action='store_true',
            help='if principal axis system shall be plotted instead')
        group_type.add_argument(
            '-O', '--full-sphere', dest='plot_full_sphere',
            action='store_true',
            help='if full sphere shall be plotted instead')
        group_geo.add_argument(
            '-V', '--viewpoint', dest='plot_viewpoint',
            metavar='<lat,lon,azi>',
            help='coordinates of the viewpoint - 3-tuple')
        group_geo.add_argument(
            '-p', '--projection', dest='plot_projection',
            metavar='<projection>', type=caps,
            help='projection of the sphere')
        group_type.add_argument(
            '-U', '--upper', dest='plot_show_upper_hemis', action='store_true',
            help='if upper hemisphere shall be shown ')
        group_quality.add_argument(
            '-N', '--points', dest='plot_n_points', metavar='<no. of points>',
            type=int,
            help='minimum number of points, used for nodallines')
        group_app.add_argument(
            '-s', '--size', dest='plot_size', metavar='<size in cm>',
            type=float,
            help='size of plot in cm')
        group_colours.add_argument(
            '-w', '--pressure-color', '--pressure-colour',
            dest='plot_pressure_colour', metavar='<colour>',
            help='colour of the tension area')
        group_colours.add_argument(
            '-r', '--tension-color', '--tension-colour',
            dest='plot_tension_colour', metavar='<colour>',
            help='colour of the pressure area ')
        group_app.add_argument(
            '-a', '--alpha', dest='plot_total_alpha', metavar='<alpha>',
            type=float,
            help='alpha value for the plot - float from 1=opaque to '
                 '0=transparent')
        group_dc.add_argument(
            '-D', '--dc', dest='plot_show_faultplanes', action='store_true',
            help='if double couple faultplanes shall be plotted ')
        group_dc.add_argument(
            '-d', '--show-1fp', dest='plot_show_1faultplane', nargs=4,
            metavar=('<index>', '<linewidth>', '<colour>', '<alpha>'),
            help='plot 1 faultplane - arguments are: index [1,2] of the '
                 'resp. FP, linewidth (float), line colour (string or '
                 'rgb-tuple), and alpha value (float between 0 and 1)')
        group_misc.add_argument(
            '-e', '--eigenvectors', dest='plot_show_princ_axes',
            metavar=('<size>', '<linewidth>', '<alpha>'), nargs=3,
            help='show eigenvectors - if used, provide 3 arguments: symbol '
                 'size, symbol linewidth, and symbol alpha value')
        group_misc.add_argument(
            '-b', '--basis-vectors', dest='plot_show_basis_axes',
            action='store_true',
            help='show NED basis in plot')
        group_app.add_argument(
            '-l', '--lines', dest='plot_outerline',
            metavar=('<linewidth>', '<colour>', '<alpha>'), nargs=3,
            help='gives the style of the outer line - 3 arguments needed: '
                 'linewidth (float), line colour (string or RGB-tuple), and '
                 'alpha value (float between 0 and 1)')
        group_app.add_argument(
            '-n', '--nodals', dest='plot_nodalline',
            metavar=('<linewidth>', '<colour>', '<alpha>'), nargs=3,
            help='gives the style of the nodal lines - 3 arguments needed: '
                 'linewidth (float), line colour (string or RGB-tuple), and '
                 'alpha value (float between 0 and 1)')
        group_quality.add_argument(
            '-q', '--quality', dest='plot_dpi', metavar='<dpi>', type=int,
            help='changes the quality for the plot in terms of dpi '
                 '(minimum=200)')
        group_type.add_argument(
            '-L', '--lines-only', dest='plot_only_lines', action='store_true',
            help='if only lines are shown (no fill - so overwrites '
                 '"fill"-related options)')
        group_misc.add_argument(
            '-i', '--input-system', dest='plot_input_system',
            type=caps, choices=ALLOWED_BASES, default='NED',
            help='if source mechanism is given as tensor in a system other '
                 'than NED')
        group_type.add_argument(
            '-I', '--show-isotropic-part', dest='plot_isotropic_part',
            action='store_true',
            help='if isotropic part shall be considered for plotting '
                 '[%(default)s]')

        # Deprecated arguments
        action = _DeprecatedArgumentAction('--basis_vectors',
                                           '--basis-vectors',
                                           real_action='store_true')
        group_misc.add_argument(
            '--basis_vectors', dest='plot_show_basis_axes', nargs=0,
            action=action, help=SUPPRESS)

        action = _DeprecatedArgumentAction('--full_sphere', '--full-sphere',
                                           real_action='store_true')
        group_misc.add_argument(
            '--full_sphere', dest='plot_full_sphere', nargs=0,
            action=action, help=SUPPRESS)

        action = _DeprecatedArgumentAction('--input_system', '--input-system')
        group_misc.add_argument(
            '--input_system', dest='plot_input_system',
            type=caps, choices=ALLOWED_BASES, default='NED',
            action=action, help=SUPPRESS)

        action = _DeprecatedArgumentAction('--lines_only', '--lines-only',
                                           real_action='store_true')
        group_misc.add_argument(
            '--lines_only', dest='plot_only_lines', nargs=0,
            action=action, help=SUPPRESS)

        action = _DeprecatedArgumentAction('--output_file', '--output-file')
        group_misc.add_argument(
            '--output_file', dest='plot_outfile', action=action, help=SUPPRESS)

        action = _DeprecatedArgumentAction('--pa_system', '--pa-system',
                                           real_action='store_true')
        group_misc.add_argument(
            '--pa_system', dest='plot_pa_plot', nargs=0,
            action=action, help=SUPPRESS)

        action = _DeprecatedArgumentAction('--pressure_colour',
                                           '--pressure-colour')
        group_misc.add_argument(
            '--pressure_colour', dest='plot_pressure_colour',
            action=action, help=SUPPRESS)

        action = _DeprecatedArgumentAction('--show1fp', '--show-1fp',
                                           real_action='store_true')
        group_misc.add_argument(
            '--show1fp', dest='plot_show_1faultplane', nargs=0,
            action=action, help=SUPPRESS)

        action = _DeprecatedArgumentAction('--show_isotropic_part',
                                           '--show-isotropic-part',
                                           real_action='store_true')
        group_misc.add_argument(
            '--show_isotropic_part', dest='plot_isotropic_part', nargs=0,
            action=action, help=SUPPRESS)

        action = _DeprecatedArgumentAction('--tension_colour',
                                           '--tension-colour')
        group_misc.add_argument(
            '--tension_colour', dest='plot_tension_colour',
            action=action, help=SUPPRESS)

        parser_plot.set_defaults(call=_call_plot, build=_build_plot_dict)

        # decompose
        help = 'decompose a given mechanism into its components'
        desc = """Returns a decomposition of the input moment tensor.\n

                    Different decompositions are available (following Jost &
        Herrmann). Either the complete decomposition or only parts are
        returned; in- and output basis systema can be chosen. The
        'fancy' option is available for better human reading.
        """

        parser_decompose = subparsers.add_parser('decompose', help=help,
                                                 description=desc,
                                                 parents=[mechanism])

        group_type = parser_decompose.add_argument_group(
            'Type of decomposition')
        group_part = parser_decompose.add_argument_group(
            'Partial decomposition')
        group_system = parser_decompose.add_argument_group(
            'Basis systems')

        helpstring11 = """
        Returns a list of the decomposition results.

        Order:
        1:
            basis of the provided input     (string);
        2:
            basis of  the representation    (string);
        3:
            chosen decomposition type      (integer);

        4:
            full moment tensor              (matrix);

        5:
            isotropic part                  (matrix);
        6:
            isotropic percentage             (float);
        7:
            deviatoric part                 (matrix);
        8:
            deviatoric percentage            (float);

        9:
            DC part                         (matrix);
        10:
            DC percentage                    (float);
        11:
            DC2 part                        (matrix);
        12:
            DC2 percentage                   (float);
        13:
            DC3 part                        (matrix);
        14:
            DC3 percentage                   (float);

        15:
            CLVD part                       (matrix);
        16:
            CLVD percentage                 (matrix);

        17:
            seismic moment                   (float);
        18:
            moment magnitude                 (float);

        19:
            eigenvectors                   (3-array);
        20:
            eigenvalues                       (list);
        21:
            p-axis                         (3-array);
        22:
            neutral axis                   (3-array);
        23:
            t-axis                         (3-array);
        24:
            faultplanes       (list of two 3-arrays).


        If option 'fancy' is set, only a small overview about geometry and
        strength is provided instead.
        """
        group_part.add_argument(
            '-c', '--complete', dest='decomp_out_complete',
            action='store_true', help=helpstring11)
        parser_decompose.add_argument(
            '-y', '--fancy', dest='decomp_out_fancy', action='store_true',
            help='key for a stylish output')
        group_part.add_argument(
            '-p', '--partial', dest='decomp_out_part',
            metavar='<part1,part2,... >',
            help='provide an argument, what part(s) shall be displayed (if '
                 'multiple, separate by commas): in, out, type, full, iso, '
                 'iso_perc, devi, devi_perc, dc, dc_perc, dc2, dc2_perc, dc3, '
                 'dc3_perc, clvd, clvd_perc, mom, mag, eigvals, eigvecs, t, '
                 'n, p, faultplanes')
        group_system.add_argument(
            '-i', '--input-system', dest='decomp_in_system',
            type=caps, choices=ALLOWED_BASES, default='NED',
            help='set to provide input in a system other than NED')
        group_system.add_argument(
            '-o', '--output-system', dest='decomp_out_system',
            type=caps, choices=ALLOWED_BASES, default='NED',
            help='set to return output in a system other than NED')
        group_type.add_argument(
            '-t', '--type', dest='decomp_key', metavar='<decomposition key>',
            type=int, choices=[20, 21, 31], default=20,
            help='integer key to choose the type of decomposition - 20: '
                 'ISO+DC+CLVD ; 21: ISO+major DC+ minor DC ; 31: ISO + 3 DCs')

        # Deprecated arguments
        action = _DeprecatedArgumentAction('--input_system', '--input-system')
        group_system.add_argument(
            '--input_system', dest='decomp_in_system',
            type=caps, choices=ALLOWED_BASES, default='NED',
            action=action, help=SUPPRESS)

        action = _DeprecatedArgumentAction('--output_system',
                                           '--output-system')
        group_system.add_argument(
            '--output_system', dest='decomp_out_system',
            type=caps, choices=ALLOWED_BASES, default='NED',
            action=action, help=SUPPRESS)

        parser_decompose.set_defaults(call=_call_decompose,
                                      build=_build_decompose_dict)

        return parser

    parser = _build_parsers()
    args = parser.parse_args(argv)

    try:
        M_raw = [float(xx) for xx in args.mechanism.split(',')]
    except:
        parser.error('invalid source mechanism')

    if not len(M_raw) in [3, 4, 6, 7, 9]:
        parser.error('invalid source mechanism')
    if len(M_raw) in [4, 6, 7, 9] and len(np.array(M_raw).nonzero()[0]) == 0:
        parser.error('invalid source mechanism')

    aa = _handle_input(M_raw, args)
    if aa is not None:
        try:
            print(aa)
        except:
            print(aa.encode("utf-8"))


if __name__ == '__main__':
    main()
