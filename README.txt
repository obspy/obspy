VCR - decorator for capturing and simulating network communication
==================================================================

Any Python socket communication in unittests (decorated with the ```@vcr```
function) and/or doctests (containing a ```# doctest: +VCR```) will be recorded
on the first run and saved into a special 'vcrtapes' directory as single
pickled file for each test case. Future test runs will reuse those recorded
network session allowing for faster tests without any network connection. In
order to create a new recording one just needs to remove/rename the pickled
session file(s).

So pretty similar to [VCRPy](https://github.com/kevin1024/vcrpy) but on socket
level instead of HTTP/HTTPS level only (request, urllib3, etc.) - so therefore
it should be more powerful ...

Inspired by:
 * https://docs.python.org/3.6/howto/sockets.html
 * https://github.com/gabrielfalcao/HTTPretty
 * http://code.activestate.com/recipes/408859/


Why was it initiated?

Network tests tend to fail sporadically, need usually a very long time 
(compared to other tests) and (surprise!)  require a network connection (if
not mocked). This module tackles all three issues mentioned above.


