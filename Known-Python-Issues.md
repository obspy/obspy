## Performance Issues

### np.arange numerically unstable

For floating point numbers and given increment numpy's ```np.arange``` is dangerous / numerical unstable ([further reading](http://www.google.de/url?sa=t&rct=j&q=&esrc=s&source=web&cd=6&ved=0CFEQ6AEwBQ&url=http%3A%2F%2Fbooks.google.de%2Fbooks%3Fid%3DYEoiYr4H2A0C%26pg%3DPA133%26lpg%3DPA133%26dq%3Dnumpy%2Barange%2Bfloat%2Bincrement%26source%3Dbl%26ots%3DownYCLYAjV%26sig%3DXrKDycJZDHAkV05wiB7H52zIB20%26hl%3Dde&ei=ORdLUM6bAof0sgaO_YCQAQ&usg=AFQjCNF__O6h4bqiMqKrjQrgsDCEQ38nwg&cad=rja), or #395). Use ```np.linspace``` instead.

### Allocating numpy arrays

Allocation of numpy arrays is much faster using the ```numpy.empty``` instead of the ```numpy.zeros``` function. The by far worst method is to use the ```numpy.array``` function on an list object. For lists the data type must be stored for each element, in contrast to numpy arrays where the data type is stored only once for the whole array. The following ```ipython``` lines show some performance tests. The differences in speed are in the order of 100.

```python
In [1]: import numpy as np
In [2]: N = int(1e7)
In [3]: %timeit np.empty(N)
100000 loops, best of 3: 12.2 µs per loop
In [4]: %timeit np.zeros(N)
10 loops, best of 3: 70.4 ms per loop

In [5]: %timeit x = np.empty(N); x[:] = np.NaN
10 loops, best of 3: 86.1 ms per loop
In [6]: %timeit x = np.array([np.NaN]*N)
10 loops, best of 3: 1.57 s per loop
```


### NumPy vs. math

numpy is designed for vector operations. Thus for each expression it needs to check whether the argument is a vector or not. This consumes time and is one of the reasons that the library math is faster for non-vector arguments.

```python
In [1]: import math as M
In [2]: import numpy as np
In [3]: %timeit M.floor(3.5)
1000000 loops, best of 3: 396 ns per loop
In [4]: %timeit np.floor(3.5)
100000 loops, best of 3: 3.72 us per loop
```

Note: math cannot handle vectors

## CTypes, Tips and Tricks ===
 * NULL pointers in ctypes are generally of `None` type. However sometimes the C functions need directly given addresses. 
   In such cases a NULL pointer to an e.g. integer must be passed with the construct `ctypes.POINTER(ctypes.c_int)()`
 * For cross platform compilation of C code the `#ifdef` statement can be really useful. To see all available variables for `#ifdef`
   type in a shell 
```sh
echo "" | gcc -E -dM -c -
```

### 64 Bit Platforms
In order to compile the C extensions on 64bit platforms add the compiler option `-m64 -fPIC`. The data types on 64bit platforms have a different type. Avoid using the type `long` in Python, better use `int32`, `int64` or for variable size `int`. `long` changes a lot between 32bit and 64bit platforms.
[http://www.unix.org/whitepapers/64bit.html Read more ...]
 
## Formats floats in a fixed exponential format 

Different operation systems are delivering different output for the exponential format of floats. Here we ensure to deliver in a for SEED valid format independent of the OS. For speed issues we simple cut any 
number ending with E+0XX or E-0XX down to E+XX or E-XX. This fails for numbers XX>99, but should not occur, because the SEED standard does not allow this values either.

```python
Python 2.5.2 (r252:60911, Feb 21 2008, 13:11:45) 
[MSC v.1310 32 bit (Intel)] on win32
>>> '%E' % 2.5
'2.500000E+000'
```

```python   
Python 2.5.2 (r252:60911, Apr  2 2008, 18:38:52)
[GCC 4.1.2 20061115 (prerelease) (Debian 4.1.1-21)] on linux2
>>> '%E' % 2.5
'2.500000E+00'
```

## Default Parameter Values in Python

Python’s handling of default parameter values is one of a few things that tends to trip up most new Python programmers (but usually only once). 

What causes the confusion is the behaviour you get when you use a “mutable” object as a default value; that is, a value that can be modified in place, like a list or a dictionary.

An example:
```python
>>> def function(data=[]):
...     data.append(1)
...     return data
...
>>> function()
[1]
>>> function()
[1, 1]
>>> function()
[1, 1, 1]
```

[Read more ...](http://effbot.org/zone/default-values.htm)

## Calculating Micro Seconds
Micro seconds cannot be calculated by the modulo operator, as the modulo operator returns only positive results.

An example:
```python
>>> -0.5 % 1
0.5
>>> -0.2 % 1
0.80000000000000004
```

A preferred way is to use the modf function:

```python
>>> from math import modf
>>> sec = 1.5
>>> msec, dsec = modf(sec)
>>> msec *= 1000
>>> print dsec, msec
1 500.0
```

## Writing and Reading Sampling Intervals

The following example shows that the effect of casting (of the sampling interval 0.01) can be circumvented by dividing with a number which got the same casting effect.

```python
>>> import numpy as np
>>> np.float32(0.01)
0.0099999998
>>> 1.0 / np.float32(0.01)
100.00000223517424
>>> np.float32(1.0) / np.float32(0.01)
100.0
```

## floor vs int
Just to avoid problems with negative numbers: 

```python
>>> import math as M
>>> M.floor(3.5)
3.0
>>> int(3.5)
3
>>> M.floor(-3.5)
-4.0
>>> int(-3.5)
-3
```


## Comma & Dot; Locale Settings
A known problem are locale settings so that the Python shell uses comma instead of dot as decimal separator. 
In this case the ctypes library could cause problems ([Read more ...](http://www.seismic-handler.org/portal/browser/SHX/trunk/src/sandbox/sscanf.py)). 
As soon as this problem occurs with !ObsPy please let us know.

## Writing Data from Numpy Arrays after Indexing Operations (#192, #193)

Using convenient indexing operations on Numpy `ndarray`s can lead to problems when writing data via external functions with ctypes (C/Fortran). Consider the following example...

```python
import numpy as np
from obspy.core import read, Trace

x = np.arange(10)
y = x[:5]
z = x[::2]
tr1 = Trace(data=y)
tr2 = Trace(data=z)
print tr1.data
print tr2.data
```

...which shows that in Python the data of Traces 1 and 2 is:

```python
[0 1 2 3 4]
[0 2 4 6 8]
```

But after writing and reading the data again...

```python
tr1.write("/tmp/tr1.tmp", "MSEED")
tr2.write("/tmp/tr2.tmp", "MSEED")
tr1 = read("/tmp/tr1.tmp")[0]
tr2 = read("/tmp/tr2.tmp")[0]
print tr1.data
print tr2.data
```

...it is obvious that there was a problem with using the reference to the original data array. During the write operation not the correct data got written to file:

```python
[0 1 2 3 4]
[0 1 2 3 4]
```

It seems that this can only be avoided by creating a fresh array from the array that was created as a new view on the original data:

```python
z_safe = np.array(z)
tr1 = Trace(data=y)
tr2 = Trace(data=z_safe)
tr1.write("/tmp/tr1.tmp", "MSEED")
tr2.write("/tmp/tr2.tmp", "MSEED")
tr1 = read("/tmp/tr1.tmp")[0]
tr2 = read("/tmp/tr2.tmp")[0]
print tr1.data
print tr2.data
```

which gives the expected result:

```python
[0 1 2 3 4]
[0 2 4 6 8]
```

Whether the data is safe for operations with ctypes or not can be checked looking at the ndarray flags:

```python
print y.flags
print z.flags
print z_safe.flags
```
```python
True
False
True
```

To summarize: When data arrays are created via e.g. indexing operations on other arrays it should be checked if the correct data get passed on during `ctypes` calls. Also refer to bugs #192 and #193.

## Floating Point Arithmetic Issues

 * [Python documentation page on floating point arithmetic](http://docs.python.org/tutorial/floatingpoint.html)
 * Numpy information on machine limits for floating point types: ```numpy.finfo()```
