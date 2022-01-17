=====================================
Python Introduction for Seismologists
=====================================

Here we want to give a small, incomplete introduction to the Python_
programming language, with links to useful packages and further resources. The
key features are explained via the following Python script:

.. code-block:: python
   :linenos:

   #!/usr/bin/env python
   import glob
   from obspy.core import read
   
   for file in glob.glob('*.z'):
       st = read(file)
       tr = st[0]
       msg = "%s %s %f %f" % (tr.stats.station, str(tr.stats.starttime),
                              tr.data.mean(), tr.data.std())
       print(msg)

Description of each line of the example above:

*Line 1*
    Shebang, specifying the location of the Python interpreter for Unix-like
    operating systems.
*Lines 2-3*
    Import modules/libraries/packages in the current namespace. The :mod:`glob`
    module, which allows wildcard matching on filenames, is imported here. All
    functions inside this module can be accessed via ``glob.function()``
    afterwards, such as :func:`glob.glob()`.
    Furthermore, a single function :func:`~obspy.core.stream.read()` from the
    :mod:`obspy.core` module is imported, which is used to read various
    different seismogram file formats.
*Line 5*
    Starts a ``for``-loop using the :func:`~glob.glob` function of the module
    :mod:`glob` on all files ending with ``'.z'``.

    .. note::

        The length of all loops in Python is determined by the indentation level.
        Do not mix spaces and tabs in your program code for indentation, this
        produces bugs that are not easy to identify.

*Line 6*
    Uses the :func:`~obspy.core.stream.read()` function from the
    :mod:`obspy.core` module to read in the seismogram to a
    :class:`~obspy.core.stream.Stream` object named ``st``.
*Line 7*
    Assigns the first :class:`~obspy.core.trace.Trace` object of the
    list-like :class:`~obspy.core.stream.Stream` object to the variable ``tr``.
*Line 8-9*
    A Python counterpart for the well-known C function ``sprintf`` is the ``%``
    operator acting on a format string. Here we print the header attributes
    ``station`` and ``starttime`` as well as the return value of the methods
    :func:`~numpy.mean` and :func:`~numpy.std` acting on the data sub-object
    of the :class:`~obspy.core.trace.Trace` (which are of type
    :class:`numpy.ndarray`).
*Line 10*
    Prints content of variable ``msg`` to the screen.

As Python_ is an interpreter language, we recommend to use the IPython_ shell
for rapid development and trying things out. It supports tab completion,
history expansion and various other features. E.g.
type ``help(glob.glob)`` or ``glob.glob?`` to see the help of the
:func:`~glob.glob` function (the module must be imported beforehand).

.. rubric:: Further Resources

* https://docs.python.org/3/tutorial/
    Official Python tutorial.
* https://docs.python.org/3/library/index.html
    Python library reference
* https://software-carpentry.org/
    Very instructive video lectures on various computer related topics. A good
    starting point for learning Python and Version Control with Git.
* https://ipython.org/
    An enhanced interactive Python shell.
* https://docs.scipy.org/doc/
   NumPy and SciPy are the matrix based computation modules of Python. The
   allow fast array manipulation (functions in C). NumPy and SciPy provide
   access to FFTW, LAPACK, ATLAS or BLAS. That is svd, eigenvalues...
   ObsPy uses the numpy.ndarrays for storing the data (e.g. tr.data).
* https://matplotlib.org/gallery.html
   matplotlib is the 2-D plotting package for Python. The gallery is the market
   place which allows you to go shopping for all kind of figures. The source
   code for each figure is linked. Note matplotlib has even its own latex
   renderer.
* https://scitools.org.uk/cartopy/docs/latest/
   Package plotting 2D data on maps in Python. Similar to GMT.
* https://trac.osgeo.org/gdal/wiki/GdalOgrInPython
   Package which allows to directly read a GeoTiff which then can be plotted
   with the Cartopy package.
* https://svn.geophysik.uni-muenchen.de/trac/mtspecpy
   Multitaper spectrum bindings for Python


.. _Python: https://www.python.org
.. _IPython: https://ipython.org
