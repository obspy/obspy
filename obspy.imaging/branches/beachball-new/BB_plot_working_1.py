#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Code for direct plotting of beachballs
#
#  Version for test of concept


#  Sebastian & Lasse

#  Hamburg - April 2010



import pylab
import numpy as np

from matplotlib.nxutils import points_inside_poly
import StringIO


epsilon = 0.0000001

#-------------------------------------------------------------------

def plot_bb(M, borderline_points=360, outfile=None, format=None):

    n_curve_points = int(borderline_points)

    print '\nM in'
    print M
    print '\nTrace(M)'
    print np.trace(M)

    M = M - np.diag(np.array([1. / 3 * np.trace(M),
                              1. / 3 * np.trace(M),
                              1. / 3 * np.trace(M)]))

    print '\nreduced M'
    print M

    trace_M = np.trace(M)
    if abs(np.trace(M)) < epsilon:
        trace_M = 0

    EW, EV = np.linalg.eigh(M)
    for i, _ew in enumerate(EW):
        if abs(EW[i]) < epsilon:
            EW[i] = 0
    print '\neigenvalues: ', EW
    print '\neigenvectors:\n', EV, '\n'

    EW_order = np.argsort(EW)

    EW1 = EW[EW_order[2]]
    EW2 = EW[EW_order[1]]
    EW3 = EW[EW_order[0]]
    EV1 = EV[:, EW_order[2]]
    EV2 = EV[:, EW_order[1]]
    EV3 = EV[:, EW_order[0]]

    if trace_M > 0:
        EWs = EW3
        EVs = EV3
        EWh = EW1
        EVh = EV1
        print 'EXPLOSION - all axes pressure\n'
        clr = -1
    elif np.trace < 0:
        EWs = EW3
        EVs = EV3
        EWh = EW1
        EVh = EV1
        print 'IMPLOSION - all axes tension\n'
        clr = 1
    elif abs(EW3) > EW1:
        EWs = EW3
        EVs = EV3
        EWh = EW1
        EVh = EV1
        print 'SIGMA AXIS = tension axis\n'
        clr = -1
    else:
        EWs = EW1
        EVs = EV1
        EWh = EW3
        EVh = EV3
        print 'SIGMA AXIS = pressure axis\n'
        clr = 1
    if (EW1 < 0 and np.trace(M) >= 0):
        print 'Houston, we have had a problem  - check M !!!!!!'
        raise SystemExit
    EWn = EW2
    EVn = EV2

    chng_basis = np.matrix(np.zeros((3, 3)))

    #order of eigenvector's basis: (H,N,S)

    chng_basis[:, 0] = EVh
    chng_basis[:, 1] = EVn
    chng_basis[:, 2] = EVs

    # build the borderlines of positive/negative areas

    degree_steps = n_curve_points

    phi = (np.arange(degree_steps) / float(degree_steps) + 1. / degree_steps) * 2 * np.pi

    # analytcal/geometrical solution for separatrix curve - alpha is opening angle
    # between principal axis SIGMA and point of curve. alpha is 0, if
    # curve is on SIGMA axis

    RHS = -EWs / (EWn * np.cos(phi) ** 2 + EWh * np.sin(phi) ** 2)
    alpha = np.arctan(np.sqrt(RHS)) / np.pi * 180

    # set both curves
    r_hor = np.sin(alpha / 180. * np.pi)

    H_values = np.sin(phi) * r_hor
    N_values = np.cos(phi) * r_hor

    S_values_positive = np.cos(alpha / 180. * np.pi)
    S_values_negative = -np.cos(alpha / 180. * np.pi)

    #--------------------------------------------------------------------------
    # change basis back to original input reference system

    line_tuple_pos = np.zeros((3, degree_steps))
    line_tuple_neg = np.zeros((3, degree_steps))

    for ii in np.arange(degree_steps):
        pos_vec_in_EV_basis = np.array([H_values[ii], N_values[ii], S_values_positive[ii] ]).transpose()
        neg_vec_in_EV_basis = np.array([H_values[ii], N_values[ii], S_values_negative[ii] ]).transpose()
        line_tuple_pos[:, ii] = np.dot(chng_basis, pos_vec_in_EV_basis)
        line_tuple_neg[:, ii] = np.dot(chng_basis, neg_vec_in_EV_basis)


    all_EV = np.zeros((3, 6))

#    EVh_orig = np.dot(chng_basis, EVs)
    all_EV[:, 0] = EVh.transpose() #_orig.transpose()
#    EVn_orig = np.dot(chng_basis, EVn)
    all_EV[:, 1] = EVn.transpose() # _orig.transpose()
#    EVs_orig = np.dot(chng_basis, EVh)
    all_EV[:, 2] = EVs.transpose() # _orig.transpose()
#    EVh_orig_neg = np.dot(chng_basis, EVs)
    all_EV[:, 3] = -EVh.transpose() #_orig_neg.transpose()
#    EVn_orig_neg = np.dot(chng_basis, EVn)
    all_EV[:, 4] = -EVn.transpose() # _orig_neg.transpose()
#    EVs_orig_neg = np.dot(chng_basis, EVh)
    all_EV[:, 5] = -EVs.transpose() # _orig_neg.transpose()

    #-------------------------------------------------
    # bring full sphere to 2D
    # vertical stereographic projection. For vertical beachballs, this has to be turned by 90 degrees

    def stereo_vertical(coords, co_system='NED'):
        """ stereographic projection - lower hemisphere to unit
        circle, upper hemisphere to circle ring with radius between 1
        and 2"""

        if not co_system == 'NED':
            print 'other coordinate systems except NED not yet implemented'
            raise SystemExit

        n_points = len(coords[0, :])
        stereo_coords = np.zeros((2, n_points))

        for ll in np.arange(n_points):
            # second component is EAST
            co_x = coords[1, ll]

            # first component is NORTH
            co_y = coords[0, ll]

            # z given in DOWN
            co_z = -coords[2, ll]
            rho_hor = np.sqrt(co_x ** 2 + co_y ** 2)

            if co_z < 0:
                new_rho = rho_hor / (-co_z + 1.)
                new_x = co_x / rho_hor * new_rho
                new_y = co_y / rho_hor * new_rho

            else:
                new_rho = 2 - (rho_hor / (co_z + 1.))
                new_x = co_x / rho_hor * new_rho
                new_y = co_y / rho_hor * new_rho

            if rho_hor == 0:
                new_y = 0
                if  co_z < 0:
                    new_x = 0
                else:
                    new_x = 2

            stereo_coords[0, ll] = new_x
            stereo_coords[1, ll] = new_y

        return stereo_coords

    # build stereographic coordinates of curves and eigenvectors
    all_EV_stereographic = stereo_vertical(all_EV)
    pos_points_stereographic = stereo_vertical(line_tuple_pos)
    neg_points_stereographic = stereo_vertical(line_tuple_neg)

    #build edge of sphere projection
    x_sphere = np.cos(phi)
    y_sphere = np.sin(phi)

    #---------------------------
    # re-sorting coordinates should start
    # at point with largest radius, then follow curve inwards, either it is
    # already closed or it shall be closed by outer circle

    #
    #positive curve
    #
    points_pos_in_order = np.zeros((2, len(pos_points_stereographic[0, :])))

    #in polar coordinates
    #
    r_phi_positive_curve = np.zeros((len(pos_points_stereographic[0, :]), 2))
    for ii in np.arange(len(pos_points_stereographic[0, :])):
        r_phi_positive_curve[ii, 0] = np.sqrt(pos_points_stereographic[0, ii] ** 2 + pos_points_stereographic[1, ii] ** 2)
        r_phi_positive_curve[ii, 1] = np.arctan2(pos_points_stereographic[0, ii], pos_points_stereographic[1, ii]) % (2 * np.pi)

    #find index with highest r
    largest_r_idx = np.argmax(r_phi_positive_curve[:, 0])

    #check, if perhaps more values with same r - if so, take point with lowest phi
    other_idces = list(np.where(r_phi_positive_curve[:, 0] == r_phi_positive_curve[largest_r_idx, 0]))
    if len(other_idces) > 1:
        best_idx = np.argmin(r_phi_positive_curve[other_idces, 1])
        start_idx_pos = other_idces[best_idx]
    else:
        start_idx_pos = largest_r_idx

    if not start_idx_pos == 0:
        print 'redefined start point to %i for positive curve\n' % (start_idx_pos)

    #check orientation - want to go inwards

    start_r = r_phi_positive_curve[start_idx_pos, 0]
    next_idx = (start_idx_pos + 1) % len(r_phi_positive_curve[:, 0])
    prep_idx = (start_idx_pos - 1) % len(r_phi_positive_curve[:, 0])
    next_r = r_phi_positive_curve[next_idx, 0]

    keep_direction = True
    if next_r <= start_r:
        #check, if next R is on other side of area - look at total distance
        # if yes, reverse direction
        dist_first_next = (pos_points_stereographic[0, next_idx] - pos_points_stereographic[0, start_idx_pos]) ** 2\
                           + (pos_points_stereographic[1, next_idx] - pos_points_stereographic[1, start_idx_pos]) ** 2
        dist_first_other = (pos_points_stereographic[0, prep_idx] - pos_points_stereographic[0, start_idx_pos]) ** 2\
                           + (pos_points_stereographic[1, prep_idx] - pos_points_stereographic[1, start_idx_pos]) ** 2

        if  dist_first_next > dist_first_other:
            keep_direction = False

    if keep_direction:
        #direction is kept

        print 'positive curve with same direction as before\n'
        for jj in np.arange(len(pos_points_stereographic[0, :])):
            running_idx = (start_idx_pos + jj) % len(pos_points_stereographic[0, :])

            points_pos_in_order[0, jj] = pos_points_stereographic[0, running_idx]
            points_pos_in_order[1, jj] = pos_points_stereographic[1, running_idx]

    else:
        #direction  is reversed
        print 'positive curve with reverted direction\n'
        for jj in np.arange(len(pos_points_stereographic[0, :])):
            running_idx = (start_idx_pos - jj) % len(pos_points_stereographic[0, :])
            points_pos_in_order[0, jj] = pos_points_stereographic[0, running_idx]
            points_pos_in_order[1, jj] = pos_points_stereographic[1, running_idx]

    # check if step of first to second point does not have large angle
    # step (problem caused by projection of point (pole) onto whole
    # edge - if this first angle step is larger than the one between
    # points 2 and three, correct position of first point: keep R, but
    # take angle with same difference as point 2 to point 3

    angle_point_1 = (np.arctan2(points_pos_in_order[0, 0], points_pos_in_order[1, 0]) % (2 * np.pi))
    angle_point_2 = (np.arctan2(points_pos_in_order[0, 1], points_pos_in_order[1, 1]) % (2 * np.pi))
    angle_point_3 = (np.arctan2(points_pos_in_order[0, 2], points_pos_in_order[1, 2]) % (2 * np.pi))

    angle_diff_23 = (angle_point_3 - angle_point_2)
    if angle_diff_23 > np.pi :
        angle_diff_23 = (-angle_diff_23) % (2 * np.pi)

    angle_diff_12 = (angle_point_2 - angle_point_1)
    if angle_diff_12 > np.pi :
        angle_diff_12 = (-angle_diff_12) % (2 * np.pi)

    if abs(angle_diff_12) > abs(angle_diff_23):
        r_old = np.sqrt(points_pos_in_order[0, 0] ** 2 + points_pos_in_order[1, 0] ** 2)
        new_angle = (angle_point_2 - angle_diff_23) % (2 * np.pi)
        points_pos_in_order[0, 0] = r_old * np.sin(new_angle)
        points_pos_in_order[1, 0] = r_old * np.cos(new_angle)
        print 'corrected position of first point in positive curve to (%.2f,%.2f)\n' % (points_pos_in_order[0, 0], points_pos_in_order[1, 0])

    # check, if curve closed !!!!!!

    r_last_point = np.sqrt(points_pos_in_order[0, -1] ** 2 + points_pos_in_order[1, -1] ** 2)
    dist_last_first_point = np.sqrt((points_pos_in_order[0, -1] - points_pos_in_order[0, 0]) ** 2 + (points_pos_in_order[1, -1] - points_pos_in_order[1, 0]) ** 2)
    #print points_pos_in_order[0,-1], points_pos_in_order[0,0],points_pos_in_order[1,-1],points_pos_in_order[1,0]

    if dist_last_first_point > (2 - r_last_point):
        #add points on edge to polygon, if it is an open curve
        print 'positive curve not closed - closing over edge... '
        phi_end = np.arctan2(points_pos_in_order[0, -1], points_pos_in_order[1, -1]) % (2 * np.pi)
        phi_start = np.arctan2(points_pos_in_order[0, 0], points_pos_in_order[1, 0]) % (2 * np.pi)


        #add one point on the edge every 2pi/360 and two on the direct projections 

        phi_end_larger = np.sign(phi_end - phi_start)
        angle_smaller_pi = np.sign(np.pi - abs(phi_end - phi_start))

        if phi_end_larger * angle_smaller_pi > 0:
            go_ccw = True
            openangle = (phi_end - phi_start) % (2 * np.pi)
        else:
            go_ccw = False
            openangle = (phi_start - phi_end) % (2 * np.pi)

        n_edgepoints = int(openangle / np.pi * 180 * n_curve_points / 360) + 1
        #print phi_end/pi*180, phi_start/pi*180
        print 'open angle %.2f degrees - filling with %i points on the edge\n' % (openangle / np.pi * 180, n_edgepoints)

        #print phi_start, phi_end

        if go_ccw:
            points_pos_in_order = list(points_pos_in_order.transpose())
            for kk in np.arange(n_edgepoints):
                current_phi = phi_end - kk * openangle / (n_edgepoints - 1)
                points_pos_in_order.append([2 * np.sin(current_phi), 2 * np.cos(current_phi) ])
            points_pos_in_order = np.array(points_pos_in_order).transpose()
        else:
            points_pos_in_order = list(points_pos_in_order.transpose())
            for kk in np.arange(n_edgepoints):
                current_phi = phi_end + kk * openangle / (n_edgepoints - 1)
                points_pos_in_order.append([2 * np.sin(current_phi), 2 * np.cos(current_phi) ])
            points_pos_in_order = np.array(points_pos_in_order).transpose()

    #
    #negative curve
    #
    points_neg_in_order = np.zeros((2, len(neg_points_stereographic[0, :])))

    #in polar coordinates
    #
    r_phi_negative_curve = np.zeros((len(neg_points_stereographic[0, :]), 2))
    for ii in np.arange(len(neg_points_stereographic[0, :])):
        r_phi_negative_curve[ii, 0] = np.sqrt(neg_points_stereographic[0, ii] ** 2 + neg_points_stereographic[1, ii] ** 2)
        r_phi_negative_curve[ii, 1] = np.arctan2(neg_points_stereographic[0, ii], neg_points_stereographic[1, ii]) % (2 * np.pi)

    #find index with highest r
    largest_r_idx = np.argmax(r_phi_negative_curve[:, 0])

    #check, if perhaps more values with same r - if so, take point with lowest phi
    other_idces = list(np.where(r_phi_negative_curve[:, 0] == r_phi_negative_curve[largest_r_idx, 0]))
    if len(other_idces) > 1:
        best_idx = np.argmin(r_phi_negative_curve[other_idces, 1])
        start_idx_neg = other_idces[best_idx]
    else:
        start_idx_neg = largest_r_idx

    if not start_idx_pos == 0:
        print 'redefined start point to %i for negative curve\n' % (start_idx_pos)

    #check orientation - want to go inwards

    start_r = r_phi_negative_curve[start_idx_neg, 0]
    next_idx = (start_idx_neg + 1) % len(r_phi_negative_curve[:, 0])
    prep_idx = (start_idx_neg - 1) % len(r_phi_negative_curve[:, 0])
    next_r = r_phi_negative_curve[next_idx, 0]

    if next_r <= start_r:
        #check, if next R is on other side of area - look at total distance
        # if yes, reverse direction
        dist_first_next = (neg_points_stereographic[0, next_idx] - neg_points_stereographic[0, start_idx_neg]) ** 2\
                           + (neg_points_stereographic[1, next_idx] - neg_points_stereographic[1, start_idx_neg]) ** 2
        dist_first_other = (neg_points_stereographic[0, prep_idx] - neg_points_stereographic[0, start_idx_neg]) ** 2\
                           + (neg_points_stereographic[1, prep_idx] - neg_points_stereographic[1, start_idx_neg]) ** 2

        if  dist_first_next < dist_first_other:
            keep_direction = True
        else:
            keep_direction = False

    if keep_direction:
        #direction is kept

        print 'negative curve with same direction as before\n'
        for jj in np.arange(len(neg_points_stereographic[0, :])):
            running_idx = (start_idx_neg + jj) % len(neg_points_stereographic[0, :])
            points_neg_in_order[0, jj] = neg_points_stereographic[0, running_idx]
            points_neg_in_order[1, jj] = neg_points_stereographic[1, running_idx]

    else:
        #direction  is reversed
        print 'negative curve with reverted direction\n'
        for jj in np.arange(len(neg_points_stereographic[0, :])):
            running_idx = (start_idx_neg - jj) % len(neg_points_stereographic[0, :])
            points_neg_in_order[0, jj] = neg_points_stereographic[0, running_idx]
            points_neg_in_order[1, jj] = neg_points_stereographic[1, running_idx]


    # check if step of first to second point does not have large angle
    # step (problem caused by projection of point (pole) onto whole
    # edge - if this first angle step is larger than the one between
    # points 2 and three, correct position of first point: keep R, but
    # take angle with same difference as point 2 to point 3

    angle_point_1 = (np.arctan2(points_neg_in_order[0, 0], points_neg_in_order[1, 0]) % (2 * np.pi))
    angle_point_2 = (np.arctan2(points_neg_in_order[0, 1], points_neg_in_order[1, 1]) % (2 * np.pi))
    angle_point_3 = (np.arctan2(points_neg_in_order[0, 2], points_neg_in_order[1, 2]) % (2 * np.pi))

    angle_diff_23 = (angle_point_3 - angle_point_2)
    if angle_diff_23 > np.pi :
        angle_diff_23 = (-angle_diff_23) % (2 * np.pi)

    angle_diff_12 = (angle_point_2 - angle_point_1)
    if angle_diff_12 > np.pi :
        angle_diff_12 = (-angle_diff_12) % (2 * np.pi)

    if abs(angle_diff_12) > abs(angle_diff_23):
        r_old = np.sqrt(points_neg_in_order[0, 0] ** 2 + points_neg_in_order[1, 0] ** 2)
        new_angle = (angle_point_2 - angle_diff_23) % (2 * np.pi)
        points_neg_in_order[0, 0] = r_old * np.sin(new_angle)
        points_neg_in_order[1, 0] = r_old * np.cos(new_angle)
        print 'corrected position of first point of negative curve to (%.2f,%.2f)\n' % (points_pos_in_order[0, 0], points_pos_in_order[1, 0])


    # check, if curve closed !!!!!!
    r_last_point = np.sqrt(points_neg_in_order[0, -1] ** 2 + points_neg_in_order[1, -1] ** 2)
    dist_last_first_point = np.sqrt((points_neg_in_order[0, -1] - points_neg_in_order[0, 0]) ** 2 + (points_neg_in_order[1, -1] - points_neg_in_order[1, 0]) ** 2)


    if dist_last_first_point > (2 - r_last_point):
        #add points on edge to polygon, if it is an open curve
        print 'negative curve not closed - closing over edge...'
        phi_end = np.arctan2(points_neg_in_order[0, -1], points_neg_in_order[1, -1]) % (2 * np.pi)
        phi_start = np.arctan2(points_neg_in_order[0, 0], points_neg_in_order[1, 0]) % (2 * np.pi)

        #add one point on the edge every fraction of degree given by input parameter and two on the direct projections 

        phi_end_larger = np.sign(phi_end - phi_start)
        angle_smaller_pi = np.sign(np.pi - abs(phi_end - phi_start))

        if phi_end_larger * angle_smaller_pi > 0:
            go_ccw = True
            openangle = (phi_end - phi_start) % (2 * np.pi)
        else:
            go_ccw = False
            openangle = (phi_start - phi_end) % (2 * np.pi)

        n_edgepoints = int(openangle / np.pi * 180 * n_curve_points / 360) + 1
        print 'open angle %.2f degrees - filling with %i points on the edge\n' % (openangle / np.pi * 180, n_edgepoints)

        if go_ccw:
            points_neg_in_order = list(points_neg_in_order.transpose())
            for kk in np.arange(n_edgepoints):
                current_phi = phi_end - kk * openangle / (n_edgepoints - 1)
                points_neg_in_order.append([2 * np.sin(current_phi), 2 * np.cos(current_phi) ])
            points_neg_in_order = np.array(points_neg_in_order).transpose()
        else:
            points_neg_in_order = list(points_neg_in_order.transpose())
            for kk in np.arange(n_edgepoints):
                current_phi = phi_end + kk * openangle / (n_edgepoints - 1)
                points_neg_in_order.append([2 * np.sin(current_phi), 2 * np.cos(current_phi) ])
            points_neg_in_order = np.array(points_neg_in_order).transpose()

    #---------------------------------------------------------------------
    # smooth curves
    #
    # for avoiding strange behaviour of the filling
    # polygon (intersecting the circle in visible amount) curve on an
    # arc must have at least n_curve_points per degree - otherwise
    # taking one point per degree

    def smooth_curve(curve, points_per_degree=1):

        coord_array = curve.transpose()

        smoothed_array = np.zeros((1, 2))
        smoothed_array[0, :] = coord_array[0]

        #now in shape (n_points,2)
        for idx, val in enumerate(coord_array[:-1]):
            r1 = np.sqrt(val[0] ** 2 + val[1] ** 2)
            r2 = np.sqrt(coord_array[idx + 1][0] ** 2 + coord_array[idx + 1][1] ** 2)
            phi1 = np.arctan2(val[0], val[1])
            phi2 = np.arctan2(coord_array[idx + 1][0], coord_array[idx + 1][1])

            phi2_larger = np.sign(phi2 - phi1)
            angle_smaller_pi = np.sign(np.pi - abs(phi2 - phi1))

            if phi2_larger * angle_smaller_pi > 0:
                go_cw = True
                openangle = (phi2 - phi1) % (2 * np.pi)
            else:
                go_cw = False
                openangle = (phi1 - phi2) % (2 * np.pi)

            openangle_deg = openangle / np.pi * 180
            radius_diff = r2 - r1

            if openangle_deg > 1. / points_per_degree:

                n_fillpoints = int(openangle_deg * points_per_degree)
                fill_array = np.zeros((n_fillpoints, 2))
                if go_cw:
                    angles = ((np.arange(n_fillpoints) + 1) * openangle / (n_fillpoints + 1) + phi1) % (2 * np.pi)
                else:
                    angles = (phi1 - (np.arange(n_fillpoints) + 1) * openangle / (n_fillpoints + 1)) % (2 * np.pi)
                radii = (np.arange(n_fillpoints) + 1) * radius_diff / (n_fillpoints + 1) + r1

                fill_array[:, 0] = radii * np.sin(angles)
                fill_array[:, 1] = radii * np.cos(angles)

                smoothed_array = np.append(smoothed_array, fill_array, axis=0)

            smoothed_array = np.append(smoothed_array, [coord_array[idx + 1]], axis=0)


        return smoothed_array.transpose()

    points_neg_in_order = smooth_curve(points_neg_in_order, points_per_degree=(n_curve_points / 360.))
    points_pos_in_order = smooth_curve(points_pos_in_order, points_per_degree=(n_curve_points / 360.))

    #--------------------------------------------------------------------------
    # outer circle

    outer_circle_points = np.zeros((2, len(phi)))
    outer_circle_points[0, :] = 2 * np.sin(phi)
    outer_circle_points[1, :] = 2 * np.cos(phi)

    #--------------------------------------------------------------------------
    # check if one curve contains the other completely
    #if yes, colours and order of plotting have to be changed

    lo_points_in_pos_curve = list(points_pos_in_order.transpose())
    lo_points_in_neg_curve = list(points_neg_in_order.transpose())

    neg_in_pos = 0
    pos_in_neg = 0

    #print lo_points_in_pos_curve

    # check , if negative in positive
    mask_neg_in_pos = points_inside_poly(lo_points_in_neg_curve, lo_points_in_pos_curve)
    if np.prod(mask_neg_in_pos):
        print 'negative curve completely within positive curve'
        neg_in_pos = 1

    # check , if positive in negative
    mask_pos_in_neg = points_inside_poly(lo_points_in_pos_curve, lo_points_in_neg_curve)
    if np.prod(mask_pos_in_neg):
        print 'positive curve completely within negative curve'
        pos_in_neg = 1

    #--------------------------------------------------------------------------
    # stereographic plot

    pylab.close('all')

    plotfig = pylab.figure(88, figsize=(10, 10))
    plotfig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
    ax = plotfig.add_subplot(111, aspect='equal')
    pylab.axis([-1.3, 1.3, -1.3, 1.3], 'equal')
    ax.axison = True

    if clr > 0 :

        ax.fill(outer_circle_points[0, :], outer_circle_points[1, :], 'w')
        ax.fill(points_pos_in_order[0, :], points_pos_in_order[1, :], 'r')
        ax.fill(points_neg_in_order[0, :], points_neg_in_order[1, :], 'r')

        if pos_in_neg :
            ax.fill(outer_circle_points[0, :], outer_circle_points[1, :], 'r')
            ax.fill(points_neg_in_order[0, :], points_neg_in_order[1, :], 'w')
            ax.fill(points_pos_in_order[0, :], points_pos_in_order[1, :], 'r')
            pass

        if neg_in_pos:
            ax.fill(outer_circle_points[0, :], outer_circle_points[1, :], 'r')
            ax.fill(points_pos_in_order[0, :], points_pos_in_order[1, :], 'w')
            ax.fill(points_neg_in_order[0, :], points_neg_in_order[1, :], 'r')
            pass

        ax.plot([all_EV_stereographic[0, 0]], [all_EV_stereographic[1, 0]], 'm^', ms=10)
        ax.plot([all_EV_stereographic[0, 3]], [all_EV_stereographic[1, 3]], 'mv', ms=10)
        ax.plot([all_EV_stereographic[0, 1]], [all_EV_stereographic[1, 1]], 'b^', ms=10.)
        ax.plot([all_EV_stereographic[0, 4]], [all_EV_stereographic[1, 4]], 'bv', ms=10.)
        ax.plot([all_EV_stereographic[0, 2]], [all_EV_stereographic[1, 2]], 'g^', ms=10.)
        ax.plot([all_EV_stereographic[0, 5]], [all_EV_stereographic[1, 5]], 'gv', ms=10)

    else:
        ax.fill(outer_circle_points[0, :], outer_circle_points[1, :], 'r')
        ax.fill(points_pos_in_order[0, :], points_pos_in_order[1, :], 'w')
        ax.fill(points_neg_in_order[0, :], points_neg_in_order[1, :], 'w')

        if pos_in_neg :
            ax.fill(outer_circle_points[0, :], outer_circle_points[1, :], 'w')
            ax.fill(points_neg_in_order[0, :], points_neg_in_order[1, :], 'r')
            ax.fill(points_pos_in_order[0, :], points_pos_in_order[1, :], 'w')
            pass
        if neg_in_pos:
            ax.fill(outer_circle_points[0, :], outer_circle_points[1, :], 'w')
            ax.fill(points_pos_in_order[0, :], points_pos_in_order[1, :], 'r')
            ax.fill(points_neg_in_order[0, :], points_neg_in_order[1, :], 'w')
            pass

        ax.plot([all_EV_stereographic[0, 0]], [all_EV_stereographic[1, 0]], 'g^', ms=10, lw=3)
        ax.plot([all_EV_stereographic[0, 3]], [all_EV_stereographic[1, 3]], 'gv', ms=10)
        ax.plot([all_EV_stereographic[0, 1]], [all_EV_stereographic[1, 1]], 'b^', ms=10)
        ax.plot([all_EV_stereographic[0, 4]], [all_EV_stereographic[1, 4]], 'bv', ms=10)
        ax.plot([all_EV_stereographic[0, 2]], [all_EV_stereographic[1, 2]], 'm^', ms=10)
        ax.plot([all_EV_stereographic[0, 5]], [all_EV_stereographic[1, 5]], 'mv', ms=10)


    ax.plot(points_neg_in_order[0, :] , points_neg_in_order[1, :], 'bo', lw=2)
    ax.plot(points_pos_in_order[0, :] , points_pos_in_order[1, :], 'go' , lw=2)

    ax.plot(x_sphere, y_sphere, 'k:', lw=2)
    ax.plot(2 * x_sphere, 2 * y_sphere, 'k:')
    ax.plot([0, 2.03, 0, -2.03], [2.03, 0, -2.03, 0], ',', alpha=0.)

    ax.autoscale_view(tight=False, scalex=True, scaley=True)


    plotfig.savefig('88' + outfile, dpi=100, transparent=True, format=format)

    pylab.interactive(True)
    pylab.show()

    #--------------------------------------------------------------------------
    # projection to unit sphere ('US')

    points_pos_sorted_US = points_pos_in_order[:]
    points_neg_sorted_US = points_neg_in_order[:]

    for idx, val in enumerate(points_pos_in_order.transpose()):
        old_radius = np.sqrt(val[0] ** 2 + val[1] ** 2)
        if old_radius > 1:
            points_pos_sorted_US[0, idx] = val[0] / old_radius
            points_pos_sorted_US[1, idx] = val[1] / old_radius

    for idx, val in enumerate(points_neg_in_order.transpose()):
        old_radius = np.sqrt(val[0] ** 2 + val[1] ** 2)
        if old_radius > 1:
            points_neg_sorted_US[0, idx] = val[0] / old_radius
            points_neg_sorted_US[1, idx] = val[1] / old_radius


    # set list with eigenvector-axes positions within projection area
    lo_visible_EV = []

    for idx, val in enumerate(all_EV_stereographic.transpose()):
        r_ev = np.sqrt(val[0] ** 2 + val[1] ** 2)
        if r_ev <= 1:
            lo_visible_EV.append([val[0], val[1], idx])
    visible_EVs = np.array(lo_visible_EV)

    #--------------------------------------------------------------------------
    # plot projection to unit sphere ('US')

    plotfig = pylab.figure(99, figsize=(3, 3), dpi=100)
    plotfig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax = plotfig.add_subplot(111, aspect='equal')
    # disable axis
    ax.axison = False

    if clr > 0 :
        ax.fill(x_sphere, y_sphere, 'w')
        ax.fill(points_neg_sorted_US[0, :] , points_neg_sorted_US[1, :], 'r')
        ax.fill(points_pos_sorted_US[0, :] , points_pos_sorted_US[1, :], 'r')

        if pos_in_neg :
            ax.fill(x_sphere, y_sphere, 'r')
            ax.fill(points_neg_in_order[0, :], points_neg_in_order[1, :], 'w')
            ax.fill(points_pos_in_order[0, :], points_pos_in_order[1, :], 'r')
            pass

        if neg_in_pos:
            ax.fill(x_sphere, y_sphere, 'r')
            ax.fill(points_pos_in_order[0, :], points_pos_in_order[1, :], 'w')
            ax.fill(points_neg_in_order[0, :], points_neg_in_order[1, :], 'r')
            pass

        EV_sym = ['m^', 'b^', 'g^', 'mv', 'bv', 'gv']
        for val in visible_EVs:
            ax.plot([val[0]], [val[1]], EV_sym[int(val[2])], ms=10, lw=3, alpha=0.5)

    else:
        ax.fill(x_sphere, y_sphere, 'r')
        ax.fill(points_neg_sorted_US[0, :] , points_neg_sorted_US[1, :], 'w')
        ax.fill(points_pos_sorted_US[0, :] , points_pos_sorted_US[1, :], 'w')

        if pos_in_neg :
            ax.fill(x_sphere, y_sphere, 'w')
            ax.fill(points_neg_in_order[0, :], points_neg_in_order[1, :], 'r')
            ax.fill(points_pos_in_order[0, :], points_pos_in_order[1, :], 'w')
            pass
        if neg_in_pos:
            ax.fill(x_sphere, y_sphere, 'w')
            ax.fill(points_pos_in_order[0, :], points_pos_in_order[1, :], 'r')
            ax.fill(points_neg_in_order[0, :], points_neg_in_order[1, :], 'w')
            pass

        EV_sym = ['g^', 'b^', 'm^', 'gv', 'bv', 'mv']
        for val in visible_EVs:
            ax.plot([val[0]], [val[1]], EV_sym[int(val[2])], ms=10, lw=3, alpha=0.5)

    ax.plot(points_neg_sorted_US[0, :] , points_neg_sorted_US[1, :], 'k-', lw=3)
    ax.plot(points_pos_sorted_US[0, :] , points_pos_sorted_US[1, :], 'k-' , lw=3)
    ax.plot(x_sphere, y_sphere, 'k-', lw=3)

    # create margin around figure
    ax.plot([0, 1.03, 0, -1.03], [1.03, 0, -1.03, 0], ',', alpha=0.)
    ax.autoscale_view(tight=True, scalex=True, scaley=True)

    # handle result
    if outfile:
        # create a file
        if format:
            # with given format
            plotfig.savefig('99' + outfile, dpi=100, transparent=True, format=format)
        else:
            # default format
            plotfig.savefig('99' + outfile, dpi=100, transparent=True)
        return
    elif format and not outfile:
        # create a string
        imgdata = StringIO.StringIO()
        plotfig.savefig(imgdata, format=format, dpi=100, transparent=True)
        imgdata.seek(0)
        return imgdata.read()
    else:
        # show the figure directly
        pylab.interactive(True)
        pylab.show()
        return plotfig


def runplot(a, b, c, d, e, f, outfile=None, format=None):
    m_in = [a, b, c, d, e, f]
    M = np.matrix(np.array([m_in[0], m_in[3], m_in[4],
                            m_in[3], m_in[1], m_in[5],
                            m_in[4], m_in[5], m_in[2] ]).reshape(3, 3))
    plot_bb(M, outfile=outfile, format=format)


runplot(0.91, -0.89, -0.02, 1.78, -1.55, 0.47, 'beachball-sumatra-mt.png')
