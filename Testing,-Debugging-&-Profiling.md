Several buildbots are running to ensure the functionality of ObsPy. For current reports, see http://tests.obspy.org.

## Testing

All tests in ObsPy are located in the tests directory of the certain module. The ```__init__.py``` of the tests directory itself as well as every test has a function called suite. During installation with ```setup.py``` (```easy_install```, ```python setup.py ...```) the ```obspy-runtests``` script will be registered automatically.

To run all tests or a single test from the shell/cmd do one of the following:
```sh
obspy-runtests    # Run all tests
obspy-runtests -r -v # Run all tests in verbose mode an post output to tests.obspy.org

python -m 'obspy.core.scripts.runtests.__init__'             # Run all tests
obspy-runtests -v obspy.core.tests.test_stream.StreamTestCase.test_adding
python obspy/core/tests/test_stats.py -v
python obspy/core/tests/test_stats.py -v StatsTestCase.test_pickleStats

python __init__.py -v test_mseed_util.MSEEDUtilTestCase.test_unpackSteim2
```

To run all tests or a single test inside python do one of the following:
```python
import obspy.core
obspy.core.scripts.runtests.main()

from unittest import TextTestRunner
from obspy.core.tests import suite
TextTestRunner().run(suite())            # Run all tests
TextTestRunner(verbosity=2).run(suite()) # Verbose output
    
from unittest import TextTestRunner
from obspy.core.tests.test_stats import suite
TextTestRunner().run(suite())            # Run all tests
TextTestRunner(verbosity=2).run(suite()) # Verbose output
```

Running the test verbose exposes the available tests.

## Debugging

### Debugging Python

Debugging in Python is best with [pdb](http://docs.python.org/library/pdb.html). In the interesting place of your Python source code set the debugger by

```python
import pdb; pdb.set_trace()
```

Running your program will now bring you to a ```pdb``` prompt. Type ```help``` or see [http://docs.python.org/library/pdb.html](pdb-doc) for available ```pdb``` commands.

From IPython 0.11 (http://ipython.org) you can alternatively use the more interactive ```ipdb``` debugger:

```python
from IPython.core.debugger import Tracer; Tracer()()
```

### Debugging C Extensions with ddd

In order to debug C extensions [ddd](http://www.gnu.org/software/ddd) can be used. Possibly its also good to activate the ```MALLOC_CHECK_``` environment variable before hand such the memory leaks are handled in a better way ([Read more ...](http://manpages.ubuntu.com/manpages/jaunty/man3/malloc.3.html)). E.g. for debugging libmseed you need to the following.

 * Compile the C extension with the debug option `-g`
 * Change in the directory where all the C source files are
 * Do do the following:

```sh
export MALLOC_CHECK_=2
ddd python
```

Run your test in the following `(gdb)` prompt by (from source directory):

```
run ../../../tests/test_libmseed.py LibMSEEDTestCase.test_readAndWriteTrace
```


### Debugging C Extensions with gdb

ddd is a graphical frontend for gdb. In order to debug via the network (e.g. ssh) it might be better to use gdb directly. gdb is used in the identical way like ddd in the example above.

There is also the way to debug explicitly written Python commands by gdb. Here is an example ([Read more ...](http://www.scipy.org/BugReport))

```sh 
$ gdb python
(gdb) run -c "import scipy; scipy.test(10,10)"
```

### Debugging Memory Leaks with Valgrind

Possible memory leaks can be located by [valgrind](http://www.valgrind.org). As Python allocates the memory differently form C a suppression file is needed to exclude messages due to this fact ([Read more ...](http://svn.python.org/view/python/trunk/Misc/README.valgrind?view=markup)).

```sh
valgrind --tool=memcheck \
    --leak-check=yes \
    --suppressions=/usr/lib/valgrind/python.supp \
    python mseed/tests/test_libmseed.py
```

Memory leaks do only show up in the ''definitely lost'' section if there is no reference to the Python object any more. Therefor explicitly delete objects in the tests --- the garbage collector seems to delete the object too late.

## Profiling

### Profiling C-Extensions
For profiling C-Extensions, valgrind in combination with kcachgrind can be used. For running it, do e.g.:

```sh
valgrind --tool=callgrind python tests/test_invsim.py InvSimTestCase.test_evalrespVsObsPy
kcachegrind 
```