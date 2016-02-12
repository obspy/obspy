.. _coding-style-guide:

ObsPy Coding Style Guide
========================

Like most Python projects, we try to adhere to :pep:`8` (Style Guide for Python
Code) and :pep:`257` (Docstring Conventions) with the modifications documented
here. Be sure to read all documents if you intend to contribute code to ObsPy.

Import Conventions
------------------

Like the Python projects NumPy_, SciPy_ and matplotlib_, we try to improve
readability of the code by importing the following modules in an unified
manner:

>>> import numpy as np
>>> import matplotlib.pylab as plt

.. _NumPy: http://www.numpy.org/
.. _SciPy: https://scipy.scipy.org/
.. _matplotlib: http://matplotlib.org/

Naming
------

**Names to Avoid**

* single character names except for counters or iterators
* dashes (``-``) in any package/module name
* ``__double_leading_and_trailing_underscore__`` names (reserved by Python)

**Naming Convention**

* Use meaningful variable/function/method names; these will help other people a
  lot when reading your code.
* Prepending a single underscore (``_``) means an object is "internal" /
  "private", which means that it is not supposed to be used by end-users and
  the API might change internally without notice to users (in contrast to API
  changes in public objects which get handled with deprecation warnings for one
  release cycle).
* Prepending a double underscore (``__``) to an instance variable or method
  effectively serves to make the variable or method private to its class (using
  name mangling).
* Place related classes and top-level functions together in a module. Unlike
  Java, there is no need to limit yourself to one class per module.
* Use ``CamelCase`` for class names, but ``snake_case`` for module
  names, variables and functions/methods.

======================  ===================  ====================
Type                    Public               Internal / Private
======================  ===================  ====================
Packages                ``snake_case``
Modules                 ``snake_case.py``    ``_snake_case``
Classes / Exceptions    ``CamelCase``        ``_CamelCase``
Functions / Methods     ``snake_case()``     ``_snake_case()``
Variables / Attributes  ``snake_case``       ``_snake_case``
Constants               ``CAPS_WITH_UNDER``  ``_CAPS_WITH_UNDER``
======================  ===================  ====================

Doc Strings / Comments
----------------------

* One-liner Doc Strings: both ``"""`` are in new lines

  .. code-block:: python

      def someMethod():
          """
          This is a one line doc string.
          """
          print("test")

* Multiple line Doc Strings: both ``"""`` are in new lines - also you should
  try provide a meaningful one-liner description at the top, followed by two
  linebreaks with further text.

  .. code-block:: python

      def someMethod():
          """
          This is just the short story.

          The long story is, this docstring would not have been able to fit in
          one line. Therefore we have to break lines.
          """
          print("test")

* Comments at the end of code lines should come after (at least) two spaces:

  .. code-block:: python

      x = x + 1  # Compensate for border

* Comments start with a single # followed by a single space. The same goes for
  multi-line block comments:

  .. code-block:: python

      # Compensate for border
      x = x + 1
      # The next line needs some more longish explanation which does not fit
      # on a single line.
      foobar = (foo + bar) ** 3 - 1

Function/Method Definitions
---------------------------

In docstrings which annotate functions and methods, the following
reStructuredText_ fields are recognized and formatted nicely:

``param``
    Description of a parameter.
``type``
    Type of a parameter.
``raises``, ``raise``
    That (and when) a specific exception is raised.
``var``
    Description of a variable.
``returns``, ``return``
    Description of the return value.
``rtype``
    Return type.

The field names must consist of one of these keywords and an argument (except
for ``returns`` and ``rtype``, which do not need an argument). This is best
explained by an example:

.. code-block:: python

  def formatException(etype, value, tb, limit=None):
      """
      Format the exception with a traceback.

      :param etype: exception type
      :param value: exception value
      :param tb: traceback object
      :param limit: maximum number of stack frames to show
      :type limit: integer or None
      :rtype: list of strings
      :return: Traceback messages.
      """

which renders like this:

.. function:: formatException(etype, value, tb, limit=None)

   Format the exception with a traceback.

   :param etype: exception type
   :param value: exception value
   :param tb: traceback object
   :param limit: maximum number of stack frames to show
   :type limit: integer or None
   :rtype: list of strings
   :return: Traceback messages.

.. _reStructuredText: http://docutils.sourceforge.net/rst.html

Reference Conventions
---------------------

As with :class:`numpy.ndarrays <numpy.ndarray>` or Python ``lists``, we try to
reduce the memory consumption by using references where ever possible. In the
following example ``a`` is appended to ``b`` as reference, that is the reason
why ``b`` get changed when we change ``a``:

>>> a = [1, 2, 3, 4]
>>> b = [5, 6]
>>> b.append(a)
>>> a[0] = -99
>>> print(b)
[5, 6, [-99, 2, 3, 4]]

Tests
-----

* test methods names must start with ``test_`` followed by a mixedCase part
* Tests which are expected to fail, because there is a known/unfixed bug should
  be commented with an ``XXX:`` followed by an valid ticket number, e.g.

  .. code-block:: python

      def test_doSomething():
          """XXX: This test does something.

          But fails badly. See ticket #number.
          """
          print("test")
          ...
          # XXX: here it fails
          ...

Citations
---------

References to publications (journal articles, books, etc.) should be properly
reproducible. A bibtex entry in `obspy/misc/docs/source/bibliography` should be
made for each single publication (ideally with an URL or DOI), using first
author and year as article identifier::

    @article{Beyreuther2010,
    author = {Beyreuther, Moritz and Barsch, Robert and Krischer,
              Lion and Megies, Tobias and Behr, Yannik and Wassermann, Joachim},
    title = {ObsPy: A Python Toolbox for Seismology},
    volume = {81},
    number = {3},
    pages = {530-533},
    year = {May/June 2010},
    doi = {10.1785/gssrl.81.3.530},
    URL = {http://www.seismosoc.org/publications/SRL/SRL_81/srl_81-3_es/},
    eprint = {http://srl.geoscienceworld.org/content/81/3/530.full.pdf+html},
    journal = {Seismological Research Letters}
    }

This entry can then be referenced (using the bibtex article identifier) in
docstrings in the source code with the following Sphinx syntax to be converted
to a link to the bibliography section:

  .. code-block:: python

      def some_function():
          """
          Function to do something.

          See [Beyreuther2010]_ for details.
          """
          return None


Miscellaneous
-------------

* Lines shouldn't exceed a length of ``79`` characters. No, it's not because
  we're mainly using VT100 terminals while developing, rather because the diffs
  look nicer on short lines, especially in side-by-side mode.
* never use multiple statements on the same line, e.g. ``if check: a = 0``.
* Prefer `list comprehension` to the built-in functions :func:`filter()` and
  :func:`map()` when appropriate.
