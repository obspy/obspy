vcr - decorator for capturing and simulating network communication
==================================================================

|TravisCI Status| |AppVeyor Status| |PyPI| |License| |Gitter|

Any Python socket communication in unittests (decorated with the ``@vcr``)
and/or doctests (containing a ``+VCR``) will be recorded on the first run
and saved into a special 'vcrtapes' directory as single pickled file for
each test case. Future test runs will reuse those recorded network session
allowing for faster tests without any network connection. In order to
create a new recording one just needs to remove/rename the pickled session
file(s).

So pretty similar to `VCR.py`_ but on socket level instead of HTTP/HTTPS
level only - so therefore it should be more powerful ...

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

    import requests
    from vcr import vcr

    class MyTestCase(unittest.TestCase):

       @vcr
       def test_something(self):
           response = requests.get('http://example.com')
           self.assertEqual(response.status_code, 200)

       @vcr(debug=True, overwrite=True, tape_file='python.vcr')
       def test_something_else(self):
           response = requests.get('http://python.org')
           self.assertEqual(response.status_code, 200)

VCR functionality within doctests requires currently a monkey patch and the
``+VCR`` keyword somewhere within the doctest as shown in
`test_doctest.py
<https://github.com/obspy/vcr/blob/master/tests/test_doctest.py>`__.


License
-------

This library uses the LGPLv3 license. See `LICENSE.txt
<https://github.com/obspy/vcr/blob/master/LICENSE.txt>`__ for more
details.

.. _PyPI: https://pypi.python.org/pypi/vcr
.. _VCR.py: https://github.com/kevin1024/vcrpy

.. |TravisCI Status| image:: https://travis-ci.org/obspy/vcr.svg?branch=master
   :target: https://travis-ci.org/obspy/vcr?branch=master
.. |AppVeyor Status| image:: https://ci.appveyor.com/api/projects/status/cbkyij3rcshvihuf?svg=true
   :target: https://ci.appveyor.com/project/obspy/vcr
.. |PyPI| image:: https://img.shields.io/pypi/v/vcr.svg
   :target: https://pypi.python.org/pypi/vcr
.. |Gitter| image:: https://badges.gitter.im/JoinChat.svg
   :target: https://gitter.im/obspy/obspy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |License| image:: https://img.shields.io/pypi/l/vcr.svg
   :target: https://pypi.python.org/pypi/vcr/
