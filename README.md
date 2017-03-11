

Any Python socket communication in unittests (decorated with the ```@vcr``` function) and/or doctests (containing a ```# doctest: +VCR```) will be recorded on the first run and saved into a special 'vcrtapes' directory as single pickled file for each test case. Future test runs will reuse those recorded network session allowing for faster tests without any network connection. In order to create a new recording one just needs to remove/rename the pickled session file(s).

So pretty similar to [VCRPy](https://github.com/kevin1024/vcrpy) but on socket level instead of HTTP (request, urllib3, etc.) - so therefore it should be more powerful ...

### Why was it initiated?

Network tests tend to fail sporadically, need usually a very long time (compared to other tests) and (surprise!)  require a network connection (if not mocked).

This PR tackles all three issues mentioned above. Here some performance tests (network connection had been disabled during tests of the vcr branch):

```
(Python36) d:\Workspace\obspy>git checkout master
Already on 'master'
Your branch is up-to-date with 'origin/master'.

(Python36) d:\Workspace\obspy>obspy-runtests -d obspy.clients.syngine.tests.test_client
Running d:\workspace\obspy\obspy\scripts\runtests.py, ObsPy version '1.0.2.post0+902.gbfe6b3c90a.obspy.master'
..............
----------------------------------------------------------------------
Ran 14 tests in 9.743s
OK

(Python36) d:\Workspace\obspy>git checkout vcr
Switched to branch 'vcr'
Your branch is up-to-date with 'origin/vcr'.

(Python36) d:\Workspace\obspy>obspy-runtests -d obspy.clients.syngine.tests.test_client
Running d:\workspace\obspy\obspy\scripts\runtests.py, ObsPy version '1.0.2.post0+903.g151e12679f.obspy.vcr'
..............
----------------------------------------------------------------------
Ran 14 tests in 0.432s
OK
```

### ToDo
- [ ] ensure platform compatibility - so please check it out and try to break it! So far tested on
  - [X] Win10, Python 2.7/3.4/3.5/3.6,  32bit/64bit, all combinations
  - [ ] Linux
  - [ ] Mac
- [X] SSLSocket support (HTTPS)
- [ ] needs documentation

