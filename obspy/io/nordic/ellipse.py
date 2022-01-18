# -*- coding: utf-8 -*-
"""
Routines for error ellipses in seismological coordinates (N=0, W=90)

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

- Calculate an ellipse from the covariance matrix
- See if a point is inside or on an ellipse
- Calculate the angle subtended by an ellipse (for back-azimuth uncertainty)
- Plot an ellipse
"""
from math import sin, cos, radians

import numpy as np
import copy
import io
import warnings


class Ellipse:
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
        :type center: tuple(:class:`numpy.ndarray)
        :return: ellipse
        :rtype: :class: `~obspy.io.nordic.ellipse.Ellipse`
        """
        if a < b:
            warnings.warn('Semi-major smaller than semi-minor! Switching...')
        self.a = max(a, b)
        self.b = min(a, b)
        self.theta = theta
        self.x = center[0]
        self.y = center[1]
        self.rtol = 1e-5    # tolerance for __eq__ method

    @classmethod
    def from_origin_uncertainty(cls, uncert, center=(0, 0)):
        """Set Ellipse from obspy origin_uncertainty

        :param uncert: obspy origin_uncertainty
        :type uncert: :class: `~obspy.origin.origin_uncertainty`
        :param center: center position (x,y)
        :type center: tuple
        :return: ellipse
        :rtype: :class: `~obspy.io.nordic.ellipse.Ellipse`
        """
        a = uncert.max_horizontal_uncertainty
        b = uncert.min_horizontal_uncertainty
        theta = uncert.azimuth_max_horizontal_uncertainty
        return cls(a, b, theta, center)

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
        :type center: tuple
        :return: ellipse
        :rtype: :class: `~obspy.io.nordic.ellipse.Ellipse`
        """
        cov = np.array(cov)
        if _almost_good_cov(cov):
            cov = _fix_cov(cov)
        evals, evecs = np.linalg.eig(cov)
        if np.any(evals < 0):
            cov_factor = cov[0][1]
            cov_base = cov/cov_factor
            warnings.warn("Can not make data ellipse because covariance "
                          "matrix is not positive definite: "
                          "{:g}x[{:.2f} {:g}][{:g} {:.2f}]. ".format(
                            cov_factor, cov_base[0][0], cov_base[0][1],
                            cov_base[1][0], cov_base[1][1]))
            # return cls(None, None, None, center)
            return None
        # Sort eigenvalues in decreasing order
        sort_indices = np.argsort(evals)[::-1]
        # Select semi-major and semi-minor axes
        a, b = np.sqrt(evals[sort_indices[0]]), np.sqrt(evals[sort_indices[1]])
        # Calculate angle of semi-major axis
        x_v1, y_v1 = evecs[:, sort_indices[0]]
        if y_v1 == 0.:
            theta = 90.
        else:
            theta = (np.degrees(np.arctan((x_v1) / (y_v1))) + 180) % 180
        return cls(a, b, theta, center)

    @classmethod
    def from_uncerts(cls, x_err, y_err, c_xy, center=(0, 0)):
        """Set error ellipse using Nordic epicenter uncertainties

        Call as e=Ellipse.from_uncerts(x_err,y_err,c_xy,center)

        :param x_err: x error (dist_units)
        :type x_err: float
        :param y_err: y error (dist_units)
        :type y_err: float
        :param c_xy:  x-y covariance (dist_units^2)
        :type c_xy: float
        :param center: center position (x,y)
        :type center: tuple
        :return: ellipse
        :rtype: :class: `~obspy.io.nordic.ellipse.Ellipse`
        """
        cov = [[x_err**2, c_xy], [c_xy, y_err**2]]
        return cls.from_cov(cov, center)

    @classmethod
    def from_uncerts_baz(cls, x_err, y_err, c_xy, dist, baz,
                         viewpoint=(0, 0)):
        """Set error ellipse using uncertainties, distance and back-azimuth

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
        :type viewpoint: tuple
        :return: ellipse
        :rtype: :class: `~obspy.io.nordic.ellipse.Ellipse`
        """
        x = viewpoint[0] + dist * sin(radians(baz))
        y = viewpoint[1] + dist * cos(radians(baz))
        return cls.from_uncerts(x_err, y_err, c_xy, (x, y))

    def __repr__(self):
        """String describing the ellipse

        >>> str(Ellipse(20, 10))
        'Ellipse(20, 10, 0)'
        >>> str(Ellipse(20, 10, 45, (3,4)))
        'Ellipse(20, 10, 45, (3,4))'
        """
        s = 'Ellipse({:.3g}, {:.3g}'.format(self.a, self.b)
        s += ', {:.3g}'.format(self.theta)
        if self.x != 0 or self.y != 0:
            s += ', ({:.3g},{:.3g})'.format(self.x, self.y)
        s += ')'
        return s

    def __eq__(self, other):
        """
        Returns true if two Ellipses are equal

        :param other: second Ellipse
        :type other:  :class: `~ellipsoid.Ellipse`
        :return: equal
        :rtype: bool
        """
        if not abs((self.a - other.a) / self.a) < self.rtol:
            return False
        if not abs((self.b - other.b) / self.b) < self.rtol:
            return False
        if not self.x == other.x:
            return False
        if not self.y == other.y:
            return False
        theta_diff = (self.theta - other.theta) % 180
        if not ((abs(theta_diff) < self.rtol)
                or (abs(theta_diff - 180) < self.rtol)):
            return False
        return True

    def to_cov(self):
        """Convert to covariance matrix notation

        Sources:
            https://stackoverflow.com/questions/41807958/
                    convert-position-confidence-ellipse-to-covariance-matrix

        :returns: covariance_matrix [[c_xx, c_xy], [c_xy, c_yy]],
                 center_position (x,y)
        :rtype: tuple
        """
        sin_theta = sin(radians(self.theta))
        cos_theta = cos(radians(self.theta))
        # The following is BACKWARDs from math standard
        # but consistent with geographic notation
        c_yy = self.a**2 * cos_theta**2 + self.b**2 * sin_theta**2
        c_xx = self.a**2 * sin_theta**2 + self.b**2 * cos_theta**2
        c_xy = (self.a**2 - self.b**2) * sin_theta * cos_theta
        return [[c_xx, c_xy], [c_xy, c_yy]], (self.x, self.y)

    def to_uncerts(self):
        """Convert to Nordic uncertainty values

        Call as x_err, y_err, c_xy, center =myellipse.to_uncerts()

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse.Ellipse`
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

    def is_inside(self, pt):
        """ Is the given point inside the ellipse?

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse.Ellipse`
        :param pt: coordinates of the point (x,y)
        :type pt: tuple(float, float)
        :return: True or False
        :rtype: bool
        """
        pt1 = self._relative_viewpoint(pt)
        x1, y1 = pt1
        value = ((y1**2) / (self.a**2)) + ((x1**2) / (self.b**2))
        if value < 1:
            return True
        return False

    def is_on(self, pt):
        """ Is the given point on the ellipse?

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse.Ellipse`
        :param pt: coordinates of the point (x,y)
        :type pt: tuple(float, float)
        :return: True or False
        :rtype: bool
        """
        pt1 = self._relative_viewpoint(pt)
        x1, y1 = pt1
        value = ((y1**2) / (self.a**2)) + ((x1**2) / (self.b**2))
        if abs(value - 1) < 2 * np.finfo(float).eps:
            return True
        return False

    def _relative_viewpoint(self, pt):
        """ Coordinates of the viewpoint relative to a 'centered' ellipse

        A centered ellipse has its center at 0,0 and its semi-major axis
        along the y-axis

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse.Ellipse`
        :param pt: original coordinates of the point (x,y)
        :type pt: tuple(float, float)
        :return: new coordinates of the viewpoint
        :rtype: tuple(float, float)
        """
        # Translate
        pt1 = (pt[0] - self.x, pt[1] - self.y)

        # Rotate
        r_rot = _rot_ccw(self.theta)
        rotated = np.dot(r_rot, pt1)
        return rotated

    def _absolute_viewpoint(self, pt):
        """ Coordinates of a point after 'uncentering' ellipse

        Assume that the ellipse was "centered" for calculations, now
        put the point back in its true position

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse.Ellipse`
        :param pt: original coordinates of the point (x,y)
        :type pt: tuple(float, float)
        :return: new coordinates of the viewpoint
        :rtype: tuple(float, float)
        """
        # Unrotate
        r_rot = _rot_cw(self.theta)
        unrot = np.dot(r_rot, pt)
        # Untranslate
        pt1 = (unrot[0] + self.x, unrot[1] + self.y)
        return pt1

    def _get_tangents(self, pt=(0, 0)):
        """ Return tangent intersections for a point and the ellipse

        Equation is from http://www.nabla.hr/Z_MemoHU-029.htm
        P = (-a**2 * m/c, b**2 / c)

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse.Ellipse`
        :param pt: coordinates of the point (x,y)
        :type pt: tuple(float, float)
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
        coeffs = [self.a**2 - y1**2, 2 * y1 * x1, self.b**2 - x1**2]
        if coeffs[0] == 0:
            coeffs[0] = (abs(coeffs[1]) + abs(coeffs[2])) / 1.e5
        ms = np.roots(coeffs)
        cs = -(y1 * ms) + x1
        # Determine the tangent intersect with ellipse
        # Rotated from equation because ellipse theta=0 is N-S
        t0 = (self.b**2 / cs[0], -self.a**2 * ms[0] / cs[0])
        t1 = (self.b**2 / cs[1], -self.a**2 * ms[1] / cs[1])

        # Rotate back to true coords
        t0 = self._absolute_viewpoint(t0)
        t1 = self._absolute_viewpoint(t1)
        return t0, t1

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
        :type self: :class: `~obspy.io.nordic.ellipse.Ellipse`
        :param pt: coordinates of the point (x,y)
        :type pt: tuple(float, float)
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
        t0, t1 = temp._get_tangents((0, 0))
        cosang = np.dot(t0, t1)
        sinang = np.linalg.norm(np.cross(t0, t1))
        return np.degrees(np.arctan2(sinang, cosang))

    def plot(self, linewidth=2, color='k',
             npts=100, alpha=1.0, zorder=100,
             outfile=None, format=None, fig=None, show=False):
        """ Plot the ellipse

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse.Ellipse`
        :param color: Color of the edges. Defaults to ``'k'`` (black).
        :param npts: Controls the number of interpolation points for the
            curves. Minimum is automatically set to ``100``.
        :param alpha: The alpha level of the beach ball. Defaults to ``1.0``
            (opaque).
        :param zorder: Set zorder. Artists with lower zorder values are drawn
            first.
        :param outfile: Output file string. Also used to automatically
            determine the output format. Supported file formats depend on your
            matplotlib backend. Most backends support png, pdf, ps, eps and
            svg. Defaults to ``None``.
        :param format: Format of the graph picture. If no format is given the
            outfile parameter will be used to try to automatically determine
            the output format. If no format is found it defaults to png output.
            If no outfile is specified but a format is, than a binary
            imagestring will be returned.
            Defaults to ``None``.
        :param fig: Give an existing figure instance to plot into.
            New Figure if set to ``None``.
        :param show: If no outfile/format, sets plt.show()
        """
        import matplotlib.pyplot as plt
        t = np.linspace(0, 2 * np.pi, npts)
        ell = np.array([self.b * np.sin(t), self.a * np.cos(t)])
        r_rot = _rot_cw(self.theta)
        ell_rot = np.zeros((2, ell.shape[1]))
        for i in range(ell.shape[1]):
            ell_rot[:, i] = np.dot(r_rot, ell[:, i])

        # plot the figure
        if not fig:
            fig = plt.figure(figsize=(3, 3), dpi=100)
            fig.add_subplot(111, aspect='equal')
            # ax = fig.add_subplot(111, aspect='equal')
        # else:
        #    ax = fig.gca()
        plt.plot(self.x + ell_rot[0, :], self.y + ell_rot[1, :],
                 linewidth=linewidth, color=color,
                 alpha=alpha, zorder=zorder)
        # export
        if outfile:
            if format:
                fig.savefig(outfile, dpi=100, transparent=True, format=format)
            else:
                fig.savefig(outfile, dpi=100, transparent=True)
        elif format and not outfile:
            imgdata = io.BytesIO()
            fig.savefig(imgdata, format=format, dpi=100, transparent=True)
            imgdata.seek(0)
            return imgdata.read()
        else:
            if show:
                plt.show()
            return fig

    def plot_tangents(self, pt=(0, 0), linewidth=2, color='k',
                      print_angle=False, ellipse_name=None, pt_name=None,
                      npts=100, alpha=1.0, zorder=100,
                      outfile=None, format=None, fig=None, show=False):
        """ Plot tangents to an ellipse when viewed from x,y

        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse.Ellipse`
        :param pt: coordinates of the viewpoint (x,y)
        :type pt: tuple(float, float)
        :parm self: ellipse
        :type self: :class: `~obspy.io.nordic.ellipse.Ellipse`
        :param color: Color of the edges. Defaults to ``'k'`` (black).
        :param print_angle: print the subtended angle on the plot (False)
        :param ellipse_name: text to print in the middle of the ellipse (None)
        :param pt_name: text to print next to the pt (None)
        :param npts: Controls the number of interpolation points for the
            curves. Minimum is automatically set to ``100``.
        :param alpha: The alpha level of the beach ball. Defaults to ``1.0``
            (opaque).
        :param zorder: Set zorder. Artists with lower zorder values are drawn
            first.
        :param outfile: Output file string. Also used to automatically
            determine the output format. Supported file formats depend on your
            matplotlib backend. Most backends support png, pdf, ps, eps and
            svg. Defaults to ``None``.
        :param format: Format of the graph picture. If no format is given the
            outfile parameter will be used to try to automatically determine
            the output format. If no format is found it defaults to png output.
            If no outfile is specified but a format is, than a binary
            imagestring will be returned.
            Defaults to ``None``.
        :param fig: Give an existing figure instance to plot into.
            New Figure if set to ``None``.
        :param show: If no outfile/format, sets plt.show()
        """
        import matplotlib.pyplot as plt
        # plot the figure
        fig = self.plot(linewidth=linewidth, color=color,
                        npts=npts, alpha=alpha, zorder=zorder,
                        fig=fig, show=False)
        ax = fig.gca()
        ax.plot(pt[0], pt[1], '+')
        t0, t1 = self._get_tangents(pt)
        if t0:
            ax.plot([pt[0], t0[0]], [pt[1], t0[1]], color=color)
            ax.plot([pt[0], t1[0]], [pt[1], t1[1]], color=color)
            if print_angle:
                sub_angle = self.subtended_angle(pt)
                ax.text(np.mean([pt[0], t0[0], t1[0]]),
                        np.mean([pt[1], t0[1], t1[1]]),
                        '{:.1f}'.format(sub_angle),
                        fontsize=6, color=color,
                        va='center', ha='center')
        if isinstance(ellipse_name, str):
            ax.text(self.x, self.y, ellipse_name, fontsize=8, color=color,
                    va='center', ha='center')
        if isinstance(pt_name, str):
            ax.text(pt[0], pt[1], pt_name, fontsize=8, color=color,
                    va='center', ha='center')

        # export
        if outfile:
            if format:
                fig.savefig(outfile, dpi=100, transparent=True, format=format)
            else:
                fig.savefig(outfile, dpi=100, transparent=True)
        elif format and not outfile:
            imgdata = io.BytesIO()
            fig.savefig(imgdata, format=format, dpi=100, transparent=True)
            imgdata.seek(0)
            return imgdata.read()
        else:
            if show:
                plt.show()
            return fig


def _rot_ccw(theta):
    """counter-clockwise rotation matrix for theta in DEGREES"""
    c, s = cos(radians(theta)), sin(radians(theta))
    return np.array(((c, -s), (s, c)))


def _rot_cw(theta):
    """clockwise rotation matrix for theta in DEGREES"""
    c, s = cos(radians(theta)), sin(radians(theta))
    return np.array(((c, s), (-s, c)))


def _almost_good_cov(cov):
    """Checks if a covariance matrix is "almost good" (c_xy is greater
    than c_xx and/or c_yy, but only by a little bit)
    """
    if cov[0][1] != cov[1][0]:
        return False
    ratio = cov[0][1]**2 / (cov[0][0] * cov[1][1])
    if (ratio > 1) and (ratio < 1.1):
        return True
    return False


def _fix_cov(cov):
    """Correct an "almost good" covariance matrix by making c_xx*c_yy
    > c_xy^2
    """
    if not _almost_good_cov(cov):
        return None
    ratio = cov[0][1]**2 / (cov[0][0] * cov[1][1])
    cov[0][0] *= np.sqrt(ratio) * 1.01
    cov[1][1] *= np.sqrt(ratio) * 1.01
    return cov


if __name__ == "__main__":
    import doctest
    doctest.testmod()
