.. _coding-style-guide:

ObsPy Coding Style Guide
========================

Like most Python projects, we try to adhere to :pep:`8` (Style Guide for Python
Code) and :pep:`257` (Docstring Conventions) with the modifications documented
here. Be sure to read all documents if you intend to contribute code to ObsPy.

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

Import Conventions
------------------

Like the Python projects NumPy_, SciPy_ and matplotlib_, we try to improve
readability of the code by importing the following modules in an unified
manner:

>>> import numpy as np
>>> import matplotlib.pylab as plt 

.. _NumPy: http://www.numpy.org/
.. _SciPy: http://scipy.scipy.org/
.. _matplotlib: http://matplotlib.org/

Naming
------

**Names to Avoid**

* single character names except for counters or iterators
* dashes (``-``) in any package/module name
* ``__double_leading_and_trailing_underscore__`` names (reserved by Python)

**Naming Convention**

* "Internal" means internal to a module or protected or private within a class.
* Prepending a single underscore (``_``) has some support for protecting module
  variables and functions (not included with ``import * from``). Prepending a
  double underscore (``__``) to an instance variable or method effectively
  serves to make the variable or method private to its class (using name
  mangling).
* Place related classes and top-level functions together in a module. Unlike
  Java, there is no need to limit yourself to one class per module.
* Use ``CamelCase`` for class names, but ``lower_with_under.py`` for module
  names.

==================  ======================  ===================================
Type                Public                  Internal
==================  ======================  ===================================
Packages            ``lower_with_under``      
Modules             ``lower_with_under``    ``_lower_with_under``
Classes             ``CamelCase``           ``_CamelCase``
Exceptions          ``CamelCase``    
Functions           ``lower_with_under()``  ``_lower_with_under()``
Constants           ``CAPS_WITH_UNDER``     ``_CAPS_WITH_UNDER``
Class Variables     ``lower_with_under``    ``_lower_with_under``
Instance Variables  ``lower_with_under``    ``_lower_with_under`` (protected)
                                            ``__lower_with_under`` (private)
Methods             ``lower_with_under()``  ``_lower_with_under()`` (protected)
                                            ``__lower_with_under()`` (private)
Attributes          ``lower_with_under``      
Local Variables     ``lower_with_under``      
==================  ======================  ===================================

Doc Strings
-----------

* One-liner: both ``"""`` are in new lines

  .. code-block:: python

      def someMethod():
          """
          This is a one line doc string.
          """
          print("test")

* Multiple lines: both ``"""`` are in new lines - also you should try provide
  a meaningful one-liner description at the top, followed by two linebreaks
  with further text.

  .. code-block:: python

      def someMethod():
          """
          This is just the short story. 

          The long story is, this docstring would not have been able to fit in
          one line. Therefore we have to break lines.
          """
          print("test")

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

Miscellaneous
-------------

* Lines shouldn't exceed a length of ``79`` characters. No, it's not because
  we're mainly using VT100 terminals while developing, rather because the diffs
  look nicer on short lines, especially in side-by-side mode.
* never use multiple statements on the same line, e.g. ``if check: a = 0``.
* Prefer `list comprehension` to the built-in functions :func:`filter()` and
  :func:`map()` when appropriate. 
          
