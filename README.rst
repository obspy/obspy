vcr - decorator for capturing and simulating network communication
==================================================================

|Build Status| |PyPI| |Gitter|

Any Python socket communication in unittests (decorated with the ```@vcr```)
and/or doctests (containing a ```# doctest: +VCR```) will be recorded
on the first run and saved into a special 'vcrtapes' directory as single
pickled file for each test case. Future test runs will reuse those recorded
network session allowing for faster tests without any network connection. In
order to create a new recording one just needs to remove/rename the pickled
session file(s).

So pretty similar to VCRPy (https://github.com/kevin1024/vcrpy) but on socket
level instead of HTTP/HTTPS level only - so therefore it should be more
powerful ...

Inspired by:
 * https://docs.python.org/3.6/howto/sockets.html
 * https://github.com/gabrielfalcao/HTTPretty
 * http://code.activestate.com/recipes/408859/


Why was it initiated?
---------------------

Network tests tend to fail sporadically, need usually a very long time 
(compared to other tests) and (surprise!) require a network connection (if
not mocked). This module tackles all three issues mentioned above.


Installation
------------

Install from PyPI_:

.. code:: sh

   pip install -U vcr


Usage
-----

Just decorate your unit tests with ``@vcr``:

.. code:: python

    import unittest
    from vcr import vcr
    import requests

    class MyTestCase(unittest.TestCase):
       @vcr
       def test_something(self):
           response = requests.get('http://example.com')

Doctests requires currently a monkey patch shown below and the ``+VCR`` keyword
somewhere within the unittest.

.. code:: python

    import doctest
    from vcr import vcr

    def test_something(url):
        """
        My test function

        Usage:
        >>> test_something('https://www.python.org')  # doctests: +VCR
        200
        """
        r = requests.get(url)
        return r.status_code

    if __name__ == "__main__":
        import doctest

        # monkey patch
        def runTest(self):  # NOQA
            if '+VCR' in self._dt_test.docstring:
                return vcr(self._runTest)()
            return self._runTest()
        doctest.DocTestCase._runTest = doctest.DocTestCase.runTest
        doctest.DocTestCase.runTest = runTest
        doctest.register_optionflag('VCR')

        # run doctests
        doctest.testmod()


License
-------

This library uses the LGPLv3 license. See `LICENSE.txt
<https://github.com/obspy/vcr/blob/master/LICENSE.txt>`__ for more
details.
