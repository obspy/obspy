# -*- coding: utf-8 -*-
"""
Routines for error ellipses
 - Calculate an ellipse from covariance matrix
 - See if a point is inside an ellipse
 - Calculate the angle subtended by an ellipse (for back-azimuth uncert)
 - Plot an ellipse

TODO:
 - ellipsoids (3D ellipses)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport


import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings


class _ellipse:
    def __init__(self, a, b, theta=0, center=(0, 0)):
        """Defines an ellipse

        The ellipse is assumed to be centered at zero with its semi-major axis
        axis aligned along the NORTH axis (geographic standard, not math
        standard!) unless orientation and/or center are set otherwise
        You can think of it as theta (geographic angle) = 90-phi (math angle)

        :param a: length of semi-major axis
        :type a: float
        :param b: length of semi-minor axis
        :type b: float
        :param theta: azimuth (degrees CW from 0=N (y positive))
        :type b: float
        :param center: x,y coordinates of ellipse center
        :type center: tuple of numeric
        :return: ellipse
        :rtype: :class: `~obspy.io.nordic.ellipse._ellipse`
        """
        if a < b:
            warnings.warn('Semi-major smaller than semi-minor! Switching...')
        self.a = max(a, b)
        self.b = min(a, b)
        self.theta = theta
        self.x = center[0]
        self.y = center[1]

    @classmethod
    def from_cov(cls, cov, center=(0, 0)):
        """Set error ellipse using covariance matrix

        Sources:
            http://www.visiondummy.com/2014/04/
                   draw-error-ellipse-representing-covariance-matrix/
            https://blogs.sas.com/content/iml/2014/07/23/
                    prediction-ellipses-from-covariance.html

        :param cov: covariance matrix [[c_xx, c_xy], [c_xy, c_yy]]
        :type cov: 2-list of 2-lists
        :param center: center position (x,y)
        :type center: 2-tuple of numeric
        :return: ellipse
        :rtype: :class: `~obspy.io.nordic.ellipse._ellipse`
        """
        evals, evecs = np.linalg.eig(cov)
        evals = np.sqrt(evals)
        # Sort eigenvalues in decreasing order
        sort_indices = np.argsort(evals)[::-1]
        # Select semi-major and semi-minor axes
        a, b = evals[sort_indices[0]], evals[sort_indices[1]]
        # Calculate angle of semi-major axis
        x_v1, y_v1 = evecs[:, sort_indices[0]]
        if y_v1 == 0.:
            theta = 90.
        else:
            theta = (np.degrees(np.arctan((x_v1)/(y_v1))) + 180) % 180
        return cls(a, b, theta, center)

    @classmethod
    def from_uncerts(cls, x_err, y_err, c_xy, center=(0, 0)):
        """Set error ellipse using Nordic epicenter uncertainties

        Call as e=_ellipse.from_uncerts(x_err,y_err,c_xy,center)

        :param x_err: x error (dist_units)
        :type x_err: float
        :param y_err: y error (dist_units)
        :type y_err: float
        :param c_xy:  x-y covariance (dist_units^2)
        :type c_xy: float
        :param center: center position (x,y)
        :type center: 2-tuple of numeric
        :return: ellipse
        :rtype: :class: `~obspy.io.nordic.ellipse._ellipse`
        """
        cov = [[x_err**2, c_xy], [c_xy, y_err**2]]
        return cls.from_cov(cov, center)

    @classmethod
    def from_uncerts_baz(cls, x_err, y_err, c_xy, dist, baz,
                         viewpoint=(0, 0)):
        """Set error ellipse using uncertainties, distance and back-azimuth

        Call as e=_ellipse.from_uncerts_baz(xerr,yerr,c_xy,dist,baz[,viewpt])

        Inputs:
        :param x_err: x error (dist_units)
        :type x_err: float
        :param y_err: y error (dist_units)
        :type y_err: float
        :param c_xy:  x-y covariance (dist_units^2)
        :type c_xy: float
        :param dist:  distance of center from observer
        :type dist: float
        :param baz:   back-azimuth from observer (degrees)
        :type baz: float
        :param viewpoint: observer's position
        :type viewpoint: 2-tuple of floats
        :return: ellipse
        :rtype: :class: `~obspy.io.nordic.ellipse._ellipse`
        """
        x = viewpoint[0] + dist*np.sin(np.radians(baz))
        y = viewpoint[1] + dist*np.cos(np.radians(baz))
        return cls.from_uncerts(x_err, y_err, c_xy, (x, y))

    def to_cov(self):
        """Convert to covariance matrix notation

        Sources:
            https://stackoverflow.com/questions/41807958/
                    convert-position-confidence-ellipse-to-covariance-matrix

        :returns: covariance_matrix [[c_xx, c_xy], [c_xy, c_yy]],
                 center_position (x,y)
        :rtype: 2-tuple
        """
        sin_theta = np.sin(np.radians(self.theta))
        cos_theta = np.cos(np.radians(self.theta))
        # The following is BACKWARDs from math standard
        # but consistent with geographic notation
        c_yy = self.a**2 * cos_theta**2 + self.b**2 * sin_theta**2
        c_xx = self.a**2 * sin_theta**2 + self.b**2 * cos_theta**2
        c_xy = (self.a**2 - self.b**2) * sin_theta * cos_theta
        return [[c_xx, c_xy], [c_xy, c_yy]],  (self.x, self.y)

    def to_uncerts(self):
        """Convert to Nordic uncertainty values

        Call as x_err, y_err, c_xy, center =myellipse.to_uncerts()

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse._ellipse`
        :returns: x_error (m), y_error (m),
                  x-y covariance (dist^2), center (x,y)
        :rtype: 4-tuple
        """
        cov, center = self.to_cov()
        assert cov[0][1] == cov[1][0]
        x_err = np.sqrt(cov[0][0])
        y_err = np.sqrt(cov[1][1])
        c_xy = cov[0][1]
        return x_err, y_err, c_xy, center

    def __repr__(self):
        """String describing the ellipse

        >>> str(_ellipse(20,10))
        '<a=20, b=10, theta=  0.0>'
        >>>str(ellipse._ellipse(20,10,45,(3,4)))
        '<a=20, b=10, theta= 45.0, center=(3,4)>'
        """
        s = f'<a={self.a:.3g}, b={self.b:.3g}'
        s += f', theta={self.theta:5.1f}'
        if self.x != 0 or self.y != 0:
            s += f', center=({self.x:.3g},{self.y:.3g})'
        s += '>'
        return s

    def __ROT_CCW(theta):
        """counter-clockwise rotation matrix for theta in DEGREES"""
        c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        return np.array(((c, -s), (s, c)))

    def __ROT_CW(theta):
        """clockwise rotation matrix for theta in DEGREES"""
        c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        return np.array(((c, s), (-s, c)))

    def is_inside(self, pt):
        """ Is the given point inside the ellipse?

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse._ellipse`
        :param pt: coordinates of the point (x,y)
        :type pt: 2-tuple of floats
        :return: True or False
        :rtype: bool
        """
        pt1 = self._relative_viewpoint(pt)
        x1, y1 = pt1
        value = ((y1**2)/(self.a**2)) + ((x1**2)/(self.b**2))
        if value < 1:
            return True
        return False

    def is_on(self, pt):
        """ Is the given point on the ellipse?

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse._ellipse`
        :param pt: coordinates of the point (x,y)
        :type pt: 2-tuple of floats
        :return: True or False
        :rtype: bool
        """
        pt1 = self._relative_viewpoint(pt)
        x1, y1 = pt1
        value = ((y1**2)/(self.a**2)) + ((x1**2)/(self.b**2))
        if abs(value - 1) < 2*np.finfo(float).eps:
            return True
        return False

    def _relative_viewpoint(self, pt):
        """ Coordinates of the viewpoint relative to a 'centered' ellipse

        A centered ellipse has its center at 0,0 and its semi-major axis
        along the y-axis

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse._ellipse`
        :param pt: original coordinates of the point (x,y)
        :type pt: 2-tuple of floats
        :return: new coordinates of the viewpoint
        :rtype: 2-tuple of floats
        """
        # Translate
        pt1 = (pt[0]-self.x, pt[1]-self.y)

        # Rotate
        R_rot = _ellipse.__ROT_CCW(self.theta)
        rotated = np.dot(R_rot, pt1)
        return rotated

    def _absolute_viewpoint(self, pt):
        """ Coordinates of a point after 'uncentering' ellipse

        Assume that the ellipse was "centered" for calculations, now
        put the point back in its true position

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse._ellipse`
        :param pt: original coordinates of the point (x,y)
        :type pt: 2-tuple of floats
        :return: new coordinates of the viewpoint
        :rtype: 2-tuple of floats
        """
        # Unrotate
        R_rot = _ellipse.__ROT_CW(self.theta)
        unrot = np.dot(R_rot, pt)
        # Untranslate
        pt1 = (unrot[0] + self.x, unrot[1]+self.y)
        return pt1

    def _get_tangents(self, pt=(0, 0)):
        """ Return tangent intersections for a point and the ellipse

        Equation is from http://www.nabla.hr/Z_MemoHU-029.htm
        P = (-a**2 * m/c, b**2 / c)

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse._ellipse`
        :param pt: coordinates of the point (x,y)
        :type pt: 2-tuple of floats
        :return: coordinates of both tangent intersections
        :rtype: 2-tuple of 2-tuples of floats
        """
        if self.is_inside(pt):
            print('No tangents, point is inside ellipse')
            return [], []
        elif self.is_on(pt):
            print('No tangents, point is on ellipse')
            return [], []

        # for calculations, assume ellipse is centered at zero and pointing S-N
        (x1, y1) = self._relative_viewpoint(pt)
        coeffs = [self.a**2 - y1**2, 2*y1*x1, self.b**2 - x1**2]
        if coeffs[0] == 0:
            coeffs[0] = (abs(coeffs[1]) + abs(coeffs[2])) / 1.e5
        ms = np.roots(coeffs)
        cs = -(y1 * ms) + x1
        # Determine the tangent intersect with ellipse
        # Rotated from equation because ellipse theta=0 is N-S
        T0 = (self.b**2 / cs[0],         -self.a**2 * ms[0] / cs[0])
        T1 = (self.b**2 / cs[1],         -self.a**2 * ms[1] / cs[1])

        # Rotate back to true coords
        T0 = self._absolute_viewpoint(T0)
        T1 = self._absolute_viewpoint(T1)
        return T0, T1

    def subtended_angle(self, pt=(0, 0)):
        """ Find the angle subtended by an ellipse when viewed from x,y

        Equations are from http://www.nabla.hr/IA-EllipseAndLine2.htm
        For a "centered" ellipse
            y=mx+c
            a^2*m^2 + b^2 = c^2
            where   x,y are the viewpoint coordinates,
                    a,b are the semi-* axis
                    m and c are unknown
            => (a^2 - x^2)*m^2 + 2*y*x*m + (b^2 - y^2) = 0  [Solve for m]
            and then c  = y - mx

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse._ellipse`
        :param pt: coordinates of the point (x,y)
        :type pt: 2-tuple of floats
        :return: subtended angle (degrees)
        :rtype: float

        """
        # If point is on or inside the ellipse, no need to calculate tangents
        if self.is_on(pt):
            return 180.
        elif self.is_inside(pt):
            return 360.
        # Move point to origin
        temp = copy.copy(self)
        temp.x -= pt[0]
        temp.y -= pt[1]
        T0, T1 = temp._get_tangents((0, 0))
        cosang = np.dot(T0, T1)
        sinang = np.linalg.norm(np.cross(T0, T1))
        return np.degrees(np.arctan2(sinang, cosang))

    def plot(self):
        """ Plot the ellipse

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse._ellipse`
        """
        t = np.linspace(0, 2*np.pi, 100)
        Ell = np.array([self.b*np.sin(t), self.a*np.cos(t)])
        R_rot = _ellipse.__ROT_CW(self.theta)
        Ell_rot = np.zeros((2, Ell.shape[1]))
        for i in range(Ell.shape[1]):
            Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
        plt.plot(self.x + Ell_rot[0, :], self.y + Ell_rot[1, :])
        plt.axis('equal')
        return plt.gca()

    def plot_tangents(self, pt=(0, 0)):
        """ Plot tangents to an ellipse when viewed from x,y

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse._ellipse`
        :param pt: coordinates of the viewpoint (x,y)
        :type pt: 2-tuple of floats
        """
        ax = self.plot()
        ax.plot(pt[0], pt[1], '+')
        T0, T1 = self._get_tangents(pt)
        if T0:
            ax.plot([pt[0], T0[0]], [pt[1], T0[1]])
            ax.plot([pt[0], T1[0]], [pt[1], T1[1]])
        plt.show()
        return


if __name__ == "__main__":
    import doctest
    doctest.testmod()
