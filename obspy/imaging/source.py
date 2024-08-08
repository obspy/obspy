# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: source.py
#  Purpose: Computes and plots radiation patterns
# ---------------------------------------------------------------------
"""
Functions to compute and plot radiation patterns

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import numpy as np

from itertools import chain

from obspy.core.event.source import farfield
from obspy.imaging.scripts.mopad import MomentTensor, BeachBall
from obspy.imaging.mopad_wrapper import beach
from obspy.core.util import CARTOPY_VERSION
if CARTOPY_VERSION:
    HAS_CARTOPY = True
else:
    HAS_CARTOPY = False


def _setup_figure_and_axes(kind, fig=None, subplot_size=4.0, **kwargs):
    """
    Setup figure for Event plot.

    :param kind: A list of strings or nested list of strings, see
        :meth:`obspy.core.event.event.Event.plot`.
    :type kind: list[str] or list[list[str]]
    :type subplot_size: float
    :param subplot_size: Width/height of one single subplot cell in inches.
    :rtype: tuple
    :returns: A 3-tuple with a :class:`~matplotlib.figure.Figure`, a list of
        :class:`~matplotlib.axes.Axes` and a list of strings with corresponding
        plotting options for each axes (see
        :meth:`obspy.core.event.event.Event.plot`, parameter `kind`).
    """
    import matplotlib.pyplot as plt
    # restrict potential fails on matplotlib 1.5.1 (e.g. Ubuntu xenial) owed to
    # matplotlib/matplotlib#6537 to routines that actually use Axes3D, for that
    # reason do the Axes3D import inside this routine.
    from mpl_toolkits.mplot3d import Axes3D  # NOQA
    # make 2d layout of kind parameter
    if isinstance(kind[0], (list, tuple)):
        nrows = len(kind)
        ncols = max([len(k) for k in kind])
    else:
        nrows, ncols = 1, len(kind)
        kind = [kind]
    figsize = (ncols * subplot_size, nrows * subplot_size)
    if not fig:
        fig = plt.figure(figsize=figsize, facecolor='white')
    # get tuple of "kind" options and corresponding axes
    kind_ = []
    axes = []
    for i, row in enumerate(kind):
        ncols_ = len(row)
        for j, kind__ in enumerate(row):
            kind_.append(kind__)
            kwargs["adjustable"] = "datalim"
            if kind__ in ("p_quiver", "p_sphere", "s_quiver", "s_sphere"):
                kwargs["projection"] = "3d"
                kwargs["aspect"] = "auto"
            if kind__ in ("ortho", "local", "global"):
                import cartopy.crs as ccrs
                lats = []
                lons = []
                if "events" in kwargs:
                    _cat = kwargs.pop("events")
                    for event in _cat:
                        origin = event.preferred_origin() or event.origins[0]
                        lats.append(origin.latitude)
                        lons.append(origin.longitude)
                    lat_0 = round(np.mean(lats), 4)
                    lon_0 = round(np.mean(lons), 4)
                else:
                    lat_0 = 0.0
                    lon_0 = 0.0
                if kind__ == "ortho":
                    kwargs["projection"] = ccrs.Orthographic(
                        central_longitude=lon_0,
                        central_latitude=lat_0)
                elif kind__ == "global":
                    kwargs["projection"] = ccrs.Mollweide(
                        central_longitude=lon_0
                    )
                else:
                    kwargs["projection"] = ccrs.AlbersEqualArea(
                        central_longitude=lon_0,
                        central_latitude=lat_0
                    )
                kwargs["aspect"] = "equal"
            else:  # equal aspect never worked on 3d plot, see mpl #13474
                kwargs["aspect"] = "auto"
            ax = fig.add_subplot(nrows, ncols_, i * ncols_ + j + 1, **kwargs)
            axes.append(ax)
    return fig, axes, kind_


def plot_radiation_pattern(
        mt, kind=['p_sphere', 'beachball'], coordinate_system='RTP',
        p_sphere_direction='inwards', fig=None, show=True):
    """
    Plot the P/S farfield radiation pattern on a unit sphere grid.

    The calculations are based on [Aki1980]_ eq. 4.29.

    :param mt: Focal mechanism NM x 6 (M11, M22, M33, M12, M13, M23 - the
        six independent components of the moment tensor, where the coordinate
        system is 1,2,3 = Up,South,East which equals r,theta,phi -
        Harvard/Global CMT convention). The relation to [Aki1980]_
        x,y,z equals North,East,Down convention is as follows: Mrr=Mzz,
        Mtt=Mxx, Mpp=Myy, Mrt=Mxz, Mrp=-Myz, Mtp=-Mxy.
    :param kind: One of:

        * **(A)** A list of strings or nested list of strings for a matplotlib
          plot (for details see :meth:`obspy.core.event.event.Event.plot`).
        * **(B)** ``"mayavi"``: uses the mayavi library.
        * **(C)** ``"vtk"``: This vtk option writes two vtk files to the
          current working directory. ``rpattern.vtk`` contains the p and s
          wave farfield vector field. ``beachlines.vtk`` contains the nodal
          lines of the radiation pattern. A vtk glyph filter should be applied
          to the vector field (e.g. in ParaView) to visualize it.

    :type fig: :class:`matplotlib.figure.Figure`
    :param fig: Figure instance to use.
    :type show: bool
    :param show: Whether to show the figure after plotting or not. Can be
        used to do further customization of the plot before showing it.
    :returns: Matplotlib figure or ``None`` (if ``kind`` is ``"mayavi"`` or
        ``"vtk"``)
    """
    import matplotlib.pyplot as plt

    # reoorder all moment tensors to NED and RTP convention
    # name : COMPONENT              : NED sign and index
    # NED  : NN, EE, DD, NE, ND, ED : [0, 1, 2, 3, 4, 5]
    # USE  : UU, SS, EE, US, UE, SE : [1, 2, 0, -5, 3, -4]
    # RTP  : RR, TT, PP, RT, RP, TP : [1, 2, 0, -5, 3, -4]
    # DSE  : DD, SS, EE, DS, DE, SE : [1, 2, 0, -5, -3, 4]
    if coordinate_system == 'RTP' or coordinate_system == 'USE':
        signs = [1, 1, 1, -1, 1, -1]
        indices = [1, 2, 0, 5, 3, 4]
        ned_mt = [sign * mt[ind] for sign, ind in zip(signs, indices)]
        rtp_mt = mt
    # the moment tensor has to be converted to
    # RTP/USE coordinates as well because the beachball routine relies
    # on it.
    # elif coordinate_system == 'DSE':
    #     signs = [1, 1, 1, -1, -1, 1]
    #     indices = [1, 2, 0, 5, 3, 4]
    #     ned_mt = [sign * mt[ind] for sign, ind in zip(signs, indices)]
    # elif coordinate_system == 'NED':
    #     ned_mt = mt
    else:
        msg = 'moment tensor in {:s} coordinates not implemented yet'
        raise NotImplementedError(msg.format(coordinate_system))

    # matplotlib plotting is triggered when kind is a list of strings
    if isinstance(kind, (list, tuple)):
        if not fig:
            fig, axes, kind = _setup_figure_and_axes(kind, fig=fig)
        else:
            axes = fig.axes
            if len(kind) == 1:
                kind = kind
            else:
                kind = list(chain(*kind))
        for ax, kind_ in zip(axes, kind):
            if kind_ is None:
                continue
            elif kind_ == 'p_quiver':
                _plot_radiation_pattern_quiver(ax, ned_mt, type="P")
            elif kind_ == 'p_sphere':
                _plot_radiation_pattern_sphere(
                    ax, ned_mt, type="P",
                    p_sphere_direction=p_sphere_direction)
            elif kind_ == 's_quiver':
                _plot_radiation_pattern_quiver(ax, ned_mt, type="S")
            elif kind_ == 's_sphere':
                _plot_radiation_pattern_sphere(ax, ned_mt, type="S")
            elif kind_ == 'beachball':
                ax.spines['left'].set_position('center')
                ax.spines['right'].set_color('none')
                ax.spines['bottom'].set_position('center')
                ax.spines['top'].set_color('none')
                _plot_beachball(ax, rtp_mt)
        # see https://github.com/SciTools/cartopy/issues/1207
        fig.canvas.draw()
        fig.tight_layout(pad=0.1)
        if show:
            plt.show()
        return fig

    elif kind == 'mayavi':
        _plot_radiation_pattern_mayavi(ned_mt)

    elif kind == 'vtk':
        # this saves two files, one with the vector field and one
        # with the nodal lines of the beachball
        fname_rpattern = 'rpattern.vtk'
        fname_beachlines = 'beachlines.vtk'
        _write_radiation_pattern_vtk(
            ned_mt, fname_rpattern=fname_rpattern,
            fname_beachlines=fname_beachlines)

    else:
        raise NotImplementedError('{:s} not implemented yet'.format(kind))


def _plot_radiation_pattern_sphere(
        ax3d, ned_mt, type, p_sphere_direction='inwards'):
    """
    Private function that plots a radiation pattern sphere into an
    :class:`~mpl_toolkits.mplot3d.axes3d.Axes3D`.

    :type ax3d: :class:`mpl_toolkits.mplot3d.axes3d.Axes3D`
    :param ax3d: matplotlib Axes3D object
    :param ned_mt: moment tensor in NED convention
    :param p_sphere_direction: If this is 'inwards', the tension regions of the
        beachball deform the radiation sphere inwards. If 'outwards' it deforms
        outwards.
    :param type: 'P' or 'S' (P or S wave).
    """
    import matplotlib.pyplot as plt
    type = type.upper()
    if type not in ("P", "S"):
        msg = ("type must be 'P' or 'S'")
        raise ValueError(msg)
    is_p_wave = type == "P"

    # generate spherical mesh that is aligned with the moment tensor null
    # axis. MOPAD should use NED coordinate system to avoid internal
    # coordinate transformations
    mtensor = MomentTensor(ned_mt, system='NED')

    # use the most isolated eigenvector as axis of symmetry
    evecs = mtensor.get_eigvecs()
    evals = np.abs(mtensor.get_eigvals())**2
    evals_dev = np.abs(evals - np.mean(evals))
    if is_p_wave:
        if p_sphere_direction == 'outwards':
            evec_max = evecs[np.argmax(evals_dev)]
        elif p_sphere_direction == 'inwards':
            evec_max = evecs[np.argmax(evals)]
    else:
        evec_max = evecs[np.argmax(evals_dev)]
    orientation = np.ravel(evec_max)

    # get a uv sphere that is oriented along the moment tensor axes
    ntheta, nphi = 100, 100
    points = _oriented_uv_sphere(ntheta=ntheta, nphi=nphi,
                                 orientation=orientation)
    sshape = (ntheta, nphi)

    # get radiation pattern
    if is_p_wave:
        disp = farfield(ned_mt, points, type="P")
        magn = np.sum(disp * points, axis=0)
        cmap = plt.get_cmap('bwr')
        norm = plt.Normalize(-1, 1)
    else:
        disp = farfield(ned_mt, points, type="S")
        magn = np.sqrt(np.sum(disp * disp, axis=0))
        cmap = plt.get_cmap('Greens')
        norm = plt.Normalize(0, 1)
    magn /= np.max(np.abs(magn))

    # compute colours and displace points along normal
    if is_p_wave:
        if p_sphere_direction == 'outwards':
            points *= (1. + np.abs(magn) / 2.)
        elif p_sphere_direction == 'inwards':
            points *= (1. + magn / 2.)
    else:
        points *= (1. + magn / 2.)
    colors = np.array([cmap(norm(val)) for val in magn])
    colors = colors.reshape(ntheta, nphi, 4)

    x = points[0].reshape(sshape)
    y = points[1].reshape(sshape)
    z = points[2].reshape(sshape)

    # plot 3d radiation pattern
    ax3d.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=colors)
    ax3d.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5),
             xticks=[-1, 1], yticks=[-1, 1], zticks=[-1, 1],
             xticklabels=['South', 'North'],
             yticklabels=['West', 'East'],
             zticklabels=['Up', 'Down'],
             title='{} wave farfield'.format(type))
    ax3d.view_init(elev=-110., azim=0.)


def _plot_radiation_pattern_quiver(ax3d, ned_mt, type):
    """
    Private routine that plots the wave farfield into an
    :class:`~mpl_toolkits.mplot3d.axes3d.Axes3D` object

    :type ax3d: :class:`mpl_toolkits.mplot3d.axes3d.Axes3D`
    :param ax3d: matplotlib Axes3D object
    :param ned_mt: the 6 comp moment tensor in NED orientation
    :type type: str
    :param type: 'P' or 'S' (P or S wave).
    """
    import matplotlib.pyplot as plt

    type = type.upper()
    if type not in ("P", "S"):
        msg = ("type must be 'P' or 'S'")
        raise ValueError(msg)
    is_p_wave = type == "P"

    # precompute even spherical grid and directional cosine array
    points = _equalarea_spherical_grid(nlat=14)

    if is_p_wave:
        # get radiation pattern
        disp = farfield(ned_mt, points, type="P")
        # normalized magnitude:
        magn = np.sum(disp * points, axis=0)
        magn /= np.max(np.abs(magn))
        cmap = plt.get_cmap('bwr')
    else:
        # get radiation pattern
        disp = farfield(ned_mt, points, type="S")
        # normalized magnitude (positive only):
        magn = np.sqrt(np.sum(disp * disp, axis=0))
        magn /= np.max(np.abs(magn))
        cmap = plt.get_cmap('Greens')

    # plot
    # there is a mlab3d bug that quiver vector colors and lengths
    # can only be changed if we plot each arrow independently
    for loc, vec, mag in zip(points.T, disp.T, magn.T):
        norm = plt.Normalize(-1., 1.)
        color = cmap(norm(mag))
        if is_p_wave:
            loc *= (1. + mag / 2.)
            length = abs(mag) / 2.0
        else:
            length = abs(mag) / 5.0
        ax3d.quiver(loc[0], loc[1], loc[2], vec[0], vec[1], vec[2],
                    length=length, color=color)
    ax3d.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5),
             xticks=[-1, 1], yticks=[-1, 1], zticks=[-1, 1],
             xticklabels=['South', 'North'],
             yticklabels=['West', 'East'],
             zticklabels=['Up', 'Down'],
             title='{} wave farfield'.format(type))
    ax3d.view_init(elev=-110., azim=0.)


def _plot_beachball(ax2d, rtp_mt):
    """
    Private function that plots a beachball into a 2d matplotlib
    :class:`~matplotlib.axes.Axes`.

    :type ax2d: :class:`matplotlib.axes.Axes`
    :param ax2d: 2d matplotlib Axes
    :param ax2d: matplotlib Axes3D object
    :param rtp_mt: moment tensor in RTP convention
    """
    import matplotlib.pyplot as plt
    norm = plt.Normalize(-1., 1.)
    cmap = plt.get_cmap('bwr')
    bball = beach(rtp_mt, xy=(0, 0), width=50, facecolor=cmap(norm(0.7)),
                  bgcolor=cmap(norm(-0.7)))

    ax2d.add_collection(bball)
    ax2d.set(xlim=(-50, 50), ylim=(-50, 50),
             xticks=(-40, 40), yticks=(-40, 40),
             xticklabels=('West', 'East'),
             yticklabels=('South', 'North'),
             title='lower hemisphere stereographical projection')


def _plot_radiation_pattern_mayavi(ned_mt):
    """
    Plot the radiation pattern using MayaVi.

    This private function uses the mayavi (vtk) library to plot the radiation
    pattern to screen. Note that you might have to set the QT_API environmental
    variable to e.g. export QT_API=pyqt that mayavi works properly.

    :param ned_mt: moment tensor in NED convention
    """
    # use mayavi if possible.
    try:
        from mayavi import mlab
    except Exception as err:
        print(err)
        msg = ("ObsPy failed to import MayaVi. "
               "You need to install the mayavi module "
               "(e.g. 'conda install mayavi', 'pip install mayavi'). "
               "If it is installed and still doesn't work, "
               "try setting the environmental variable QT_API to "
               "pyqt (e.g. export QT_API=pyqt) before running the "
               "code. Another option is to avoid mayavi and "
               "directly use kind='vtk' for vtk file output of the "
               "radiation pattern that can be used by external "
               "software like ParaView")
        raise ImportError(msg)

    # get mopad moment tensor
    mopad_mt = MomentTensor(ned_mt, system='NED')
    bb = BeachBall(mopad_mt, npoints=200)
    bb._setup_BB(unit_circle=False)

    # extract the coordinates of the nodal lines
    neg_nodalline = bb._nodalline_negative
    pos_nodalline = bb._nodalline_positive

    # add the first point to the end to close the nodal line
    neg_nodalline = np.hstack((neg_nodalline, neg_nodalline[:, 0][:, None]))
    pos_nodalline = np.hstack((pos_nodalline, pos_nodalline[:, 0][:, None]))

    # plot radiation pattern and nodal lines
    points = _equalarea_spherical_grid(nlat=20)
    dispp = farfield(ned_mt, points, type="P")
    disps = farfield(ned_mt, points, type="S")

    # get vector lengths
    normp = np.sum(dispp * points, axis=0)
    normp /= np.max(np.abs(normp))

    norms = np.sqrt(np.sum(disps * disps, axis=0))
    norms /= np.max(np.abs(norms))

    # make sphere to block view to the other side of the beachball
    rad = 0.8
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

    x = rad * sin(phi) * cos(theta)
    y = rad * sin(phi) * sin(theta)
    z = rad * cos(phi)

    # p wave radiation pattern
    mlab.figure(size=(800, 800), bgcolor=(0, 0, 0))
    pts1 = mlab.quiver3d(points[0], points[1], points[2],
                         dispp[0], dispp[1], dispp[2],
                         scalars=normp, vmin=-1., vmax=1.)
    pts1.glyph.color_mode = 'color_by_scalar'
    mlab.plot3d(*neg_nodalline, color=(0, 0.5, 0), tube_radius=0.01)
    mlab.plot3d(*pos_nodalline, color=(0, 0.5, 0), tube_radius=0.01)
    mlab.mesh(x, y, z, color=(0, 0, 0))

    # s wave radiation pattern
    mlab.figure(size=(800, 800), bgcolor=(0, 0, 0))
    pts2 = mlab.quiver3d(points[0], points[1], points[2],
                         disps[0], disps[1], disps[2], scalars=norms,
                         vmin=-0., vmax=1.)
    pts2.glyph.color_mode = 'color_by_scalar'
    mlab.plot3d(*neg_nodalline, color=(0, 0.5, 0), tube_radius=0.01)
    mlab.plot3d(*pos_nodalline, color=(0, 0.5, 0), tube_radius=0.01)
    mlab.mesh(x, y, z, color=(0, 0, 0))

    mlab.show()


def _write_radiation_pattern_vtk(
        ned_mt, fname_rpattern='rpattern.vtk',
        fname_beachlines='beachlines.vtk'):
    # output a vtkfile that can for exampled be displayed by ParaView
    mtensor = MomentTensor(ned_mt, system='NED')
    bb = BeachBall(mtensor, npoints=200)
    bb._setup_BB(unit_circle=False)

    # extract the coordinates of the nodal lines
    neg_nodalline = bb._nodalline_negative
    pos_nodalline = bb._nodalline_positive

    # plot radiation pattern and nodal lines
    points = _equalarea_spherical_grid()
    ndim, npoints = points.shape
    dispp = farfield(ned_mt, points, type="P")
    disps = farfield(ned_mt, points, type="S")

    # write vector field
    with open(fname_rpattern, 'w') as vtk_file:
        vtk_header = '# vtk DataFile Version 2.0\n' + \
                     'radiation pattern vector field\n' + \
                     'ASCII\n' + \
                     'DATASET UNSTRUCTURED_GRID\n' + \
                     'POINTS {:d} float\n'.format(npoints)

        vtk_file.write(vtk_header)
        # write point locations
        for x, y, z in np.transpose(points):
            vtk_file.write('{:.3e} {:.3e} {:.3e}\n'.format(x, y, z))
        # write vector field
        vtk_file.write('POINT_DATA {:d}\n'.format(npoints))
        vtk_file.write('VECTORS s_radiation float\n')
        for x, y, z in np.transpose(disps):
            vtk_file.write('{:.3e} {:.3e} {:.3e}\n'.format(x, y, z))
        vtk_file.write('VECTORS p_radiation float\n')
        for x, y, z in np.transpose(dispp):
            vtk_file.write('{:.3e} {:.3e} {:.3e}\n'.format(x, y, z))

    # write nodal lines
    with open(fname_beachlines, 'w') as vtk_file:
        npts_neg = neg_nodalline.shape[1]
        npts_pos = pos_nodalline.shape[1]
        npts_tot = npts_neg + npts_pos
        vtk_header = '# vtk DataFile Version 2.0\n' + \
                     'beachball nodal lines\n' + \
                     'ASCII\n' + \
                     'DATASET UNSTRUCTURED_GRID\n' + \
                     'POINTS {:d} float\n'.format(npts_tot)

        vtk_file.write(vtk_header)
        # write point locations
        for x, y, z in np.transpose(neg_nodalline):
            vtk_file.write('{:.3e} {:.3e} {:.3e}\n'.format(x, y, z))
        for x, y, z in np.transpose(pos_nodalline):
            vtk_file.write('{:.3e} {:.3e} {:.3e}\n'.format(x, y, z))

        # write line segments
        vtk_file.write('\nCELLS 2 {:d}\n'.format(npts_tot + 4))

        ipoints = list(range(0, npts_neg)) + [0]
        vtk_file.write('{:d} '.format(npts_neg + 1))
        for ipoint in ipoints:
            if ipoint % 30 == 29:
                vtk_file.write('\n')
            vtk_file.write('{:d} '.format(ipoint))
        vtk_file.write('\n')

        ipoints = list(range(0, npts_pos)) + [0]
        vtk_file.write('{:d} '.format(npts_pos + 1))
        for ipoint in ipoints:
            if ipoint % 30 == 29:
                vtk_file.write('\n')
            vtk_file.write('{:d} '.format(ipoint + npts_neg))
        vtk_file.write('\n')

        # cell types. 4 means cell type is a poly_line
        vtk_file.write('\nCELL_TYPES 2\n')
        vtk_file.write('4\n4')


# ===== SUPPORT FUNCTIONS FOR SPHERICAL MESHES ETC STARTING HERE:
def _oriented_uv_sphere(ntheta=100, nphi=100, orientation=[0., 0., 1.]):
    """
    Returns a uv sphere (equidistant lat/lon grid) with its north-pole rotated
    to the input axis. It returns the spherical grid points that can be used to
    generate a QuadMesh on the sphere for surface plotting.

    :param nlat: number of latitudinal grid points (default = 100)
    :param nphi: number of longitudinal grid points (default = 100)
    :param orientation: axis of the north-pole of the sphere
                        (default = [0, 0, 1])
    """
    # make rotation matrix (after numpy mailing list)
    zaxis = np.array([0., 0., 1.])
    raxis = np.cross(orientation, zaxis)  # rotate z axis to null
    raxis_norm = np.linalg.norm(raxis)
    if raxis_norm < 1e-10:  # check for zero or 180 degree rotation
        rotmtx = np.eye(3, dtype=np.float64)
    else:
        raxis /= raxis_norm

        # angle between z and null
        angle = np.arccos(np.dot(zaxis, orientation))

        eye = np.eye(3, dtype=np.float64)
        raxis2 = np.outer(raxis, raxis)
        skew = np.array([[0, raxis[2], -raxis[1]],
                         [-raxis[2], 0, raxis[0]],
                         [raxis[1], -raxis[0], 0]])

        rotmtx = (raxis2 + np.cos(angle) * (eye - raxis2) +
                  np.sin(angle) * skew)

    # make uv sphere that is aligned with z-axis
    ntheta, nphi = 100, 100
    u = np.linspace(0, 2 * np.pi, nphi)
    v = np.linspace(0, np.pi, ntheta)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # ravel point array and rotate them to the null axis
    points = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    points = np.dot(rotmtx, points)
    return points


def _equalarea_spherical_grid(nlat=30):
    """
    Generates a simple spherical equalarea grid that adjust the number of
    longitude samples to the latitude. This grid is useful to plot vectors on
    the sphere but not surfaces.

    :param nlat: number of nodes in lat direction. The number of
                 nodes in lon direction is 2*nlat+1 at the equator
    """

    ndim = 3
    colats = np.linspace(0., np.pi, nlat)
    norms = np.sin(colats)
    # Scale number of point with latitude.
    nlons = (2 * nlat * norms + 1).astype(np.int_)

    # make colat/lon grid
    colatgrid, longrid = [], []
    for ilat in range(nlat):
        nlon = nlons[ilat]
        dlon = 2. * np.pi / nlon
        lons = np.linspace(0. + dlon / 2., 2. * np.pi - dlon / 2., nlon)
        for ilon in range(nlon):
            colatgrid.append(colats[ilat])
            longrid.append(lons[ilon])
    npoints = len(longrid)

    # get cartesian coordinates of spherical grid
    points = np.empty((ndim, npoints))
    points[0] = np.sin(colatgrid) * np.cos(longrid)
    points[1] = np.sin(colatgrid) * np.sin(longrid)
    points[2] = np.cos(colatgrid)

    return points


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
