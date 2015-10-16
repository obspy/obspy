# PySAC

Python interface to the [Seismic Analysis
Code](http://ds.iris.edu/files/sac-manual/) (SAC) file format.  

File-type support:

* little and big-endian binary format
* alphanumeric format
* evenly-sampled data
* time-series, not spectra

[Project page](https://lanl-seismoacoustics.github.io/pysac)  
[Repository](https://github.com/lanl-seismoacoustics/pysac)  
[LANL Seismoacoustics](https://lanl-seismoacoustics.github.io/)


## Goals

1. Expose the file format in a way that is intuitive to SAC users and to Python
   programmers
2. Maintaining header validity when converting between ObsPy Traces.


## Features

1. **Read and write SAC binary or ASCII**
    - autodetect or specify expected byteorder
    - optional file size checking and/or header consistency checks
    - header-only reading and writing
    - "overwrite OK" checking ('lovrok' header)
2. **Convenient access and manipulation of relative and absolute time headers**
3. **User-friendly header printing/viewing**
4. **Fast access to header values from attributes**
    - With type checking, null handling, and enumerated value checking
5. **Convert to/from ObsPy Traces**
    - Conversion from ObsPy Trace to SAC trace retains detected previous SAC header values.
    - Conversion to ObsPy Trace retains the *complete* SAC header.

## Usage examples

### Read/write SAC files
```python
# read from a binary file
sac = SACTrace.read(filename)

# read header only
sac = SACTrace.read(filename, headonly=True)

# write header-only, file must exist
sac.write(filename, headonly=True)

# read from an ASCII file
sac = SACTrace.read(filename, ascii=True)

# write a binary SAC file for a Sun machine
sac.write(filename, byteorder='big')
```

### Reference-time and relative time headers
```python
sac = SACTrace(nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
               t1=23.5, data=numpy.arange(100))

sac.reftime
sac.b, sac.e, sac.t1
```

```
2000-01-01T00:00:00.000000Z
(0.0, 99.0, 23.5)
```

Move reference time by relative seconds, relative time headers are preserved.
```python
sac.reftime -= 2.5
sac.b, sac.e, sac.t1
```

```
(2.5, 101.5, 26.0)
```

Set reference time to new absolute time, relative time headers are preserved.
```python
sac.reftime = UTCDateTime(2000, 1, 1, 0, 2, 0, 0)
sac.b, sac.e
```

```
(-120.0, -21.0, -96.5)
```

### Quick header viewing

Print non-null header values.
```python
sac = SACTrace()
print sac
```

```
Reference Time = 01/01/2000 (001) 00:00:00.000000
	iztype IB: begin time
b          = 0.0
cmpaz      = 0.0
cmpinc     = 0.0
delta      = 1.0
e          = 99.0
iftype     = itime
internal0  = 2.0
iztype     = ib
kcmpnm     = Z
lcalda     = False
leven      = True
lovrok     = True
lpspol     = True
npts       = 100
nvhdr      = 6
nzhour     = 0
nzjday     = 1
nzmin      = 0
nzmsec     = 0
nzsec      = 0
nzyear     = 2000
```

Print relative time header values.
```python
sac.lh('picks')
```

```
Reference Time = 01/01/1970 (001) 00:00:00.000000
    iztype IB: begin time
    a          = None
    b          = 0.0
    e          = 0.0
    f          = None
    o          = None
    t0         = None
    t1         = None
    t2         = None
    t3         = None
    t4         = None
    t5         = None
    t6         = None
    t7         = None
    t8         = None
    t9         = None
```

### Header values as attributes

Great for interactive use, with (ipython) tab-completion...
```python
sac.<tab>
```

```
sac.a                 sac.kevnm             sac.nzsec
sac.az                sac.kf                sac.nzyear
sac.b                 sac.khole             sac.o
sac.baz               sac.kinst             sac.odelta
sac.byteorder         sac.knetwk            sac.read
sac.cmpaz             sac.ko                sac.reftime
sac.cmpinc            sac.kstnm             sac.scale
sac.copy              sac.kt0               sac.stdp
sac.data              sac.kt1               sac.stel
sac.delta             sac.kt2               sac.stla
sac.depmax            sac.kt3               sac.stlo
sac.depmen            sac.kt4               sac.t0
sac.depmin            sac.kt5               sac.t1
sac.dist              sac.kt6               sac.t2
sac.e                 sac.kt7               sac.t3
sac.evdp              sac.kt8               sac.t4
sac.evla              sac.kt9               sac.t5
sac.evlo              sac.kuser0            sac.t6
sac.f                 sac.kuser1            sac.t7
sac.from_obspy_trace  sac.kuser2            sac.t8
sac.gcarc             sac.lcalda            sac.t9
sac.idep              sac.leven             sac.to_obspy_trace
sac.ievreg            sac.lh                sac.unused23
sac.ievtyp            sac.listhdr           sac.user0
sac.iftype            sac.lovrok            sac.user1
sac.iinst             sac.lpspol            sac.user2
sac.imagsrc           sac.mag               sac.user3
sac.imagtyp           sac.nevid             sac.user4
sac.internal0         sac.norid             sac.user5
sac.iqual             sac.npts              sac.user6
sac.istreg            sac.nvhdr             sac.user7
sac.isynth            sac.nwfid             sac.user8
sac.iztype            sac.nzhour            sac.user9
sac.ka                sac.nzjday            sac.validate
sac.kcmpnm            sac.nzmin             sac.write
sac.kdatrd            sac.nzmsec
```

...and documentation!
```python
sac.iztype?
```

```
Type:        property
String form: <property object at 0x106404940>
Docstring:
I    Reference time equivalence:
* IUNKN (5): Unknown
* IB (9): Begin time
* IDAY (10): Midnight of reference GMT day
* IO (11): Event origin time
* IA (12): First arrival time
* ITn (13-22): User defined time pick n, n=0,9
```

### Convert to/from ObsPy Traces

```python
from obspy import read
tr = read()[0]
print tr.stats
```
```
         network: BW
         station: RJOB
        location: 
         channel: EHZ
       starttime: 2009-08-24T00:20:03.000000Z
         endtime: 2009-08-24T00:20:32.990000Z
   sampling_rate: 100.0
           delta: 0.01
            npts: 3000
           calib: 1.0
    back_azimuth: 100.0
     inclination: 30.0
```

```python
sac = SACTrace.from_obspy_trace(tr)
print sac
```

```
Reference Time = 08/24/2009 (236) 00:20:03.000000
	iztype IB: begin time
b          = 0.0
cmpaz      = 0.0
cmpinc     = 0.0
delta      = 0.00999999977648
depmax     = 1293.77099609
depmen     = -4.49556303024
depmin     = -1515.81311035
e          = 29.9899993297
iftype     = itime
internal0  = 2.0
iztype     = ib
kcmpnm     = EHZ
knetwk     = BW
kstnm      = RJOB
lcalda     = False
leven      = True
lovrok     = True
lpspol     = True
npts       = 3000
nvhdr      = 6
nzhour     = 0
nzjday     = 236
nzmin      = 20
nzmsec     = 0
nzsec      = 3
nzyear     = 2009
scale      = 1.0
```

```python
tr2 = sac.to_obspy_trace()
print tr2.stats
```

```
         network: BW
         station: RJOB
        location:
         channel: EHZ
       starttime: 2009-08-24T00:20:03.000000Z
         endtime: 2009-08-24T00:20:32.990000Z
   sampling_rate: 100.0
           delta: 0.01
            npts: 3000
           calib: 1.0
             sac: AttribDict({'cmpaz': 0.0, 'nzyear': 2009, 'nzjday': 236,
             'iztype': 9, 'evla': 0.0, 'nzhour': 0, 'lcalda': 0, 'evlo': 0.0,
             'scale': 1.0, 'nvhdr': 6, 'depmin': -1515.8131, 'kcmpnm': 'EHZ',
             'nzsec': 3, 'internal0': 2.0, 'depmen': -4.495563, 'cmpinc': 0.0,
             'depmax': 1293.771, 'iftype': 1, 'delta': 0.0099999998, 'nzmsec':
             0, 'lpspol': 1, 'b': 0.0, 'e': 29.99, 'leven': 1, 'kstnm': 'RJOB',
             'nzmin': 20, 'lovrok': 1, 'npts': 3000, 'knetwk': 'BW'})
```


## License

Copyright 2015. Los Alamos National Security, LLC under LA-CC-15-051. This
material was produced under U.S. Government contract DE-AC52-06NA25396 for Los
Alamos National Laboratory (LANL), which is operated by Los Alamos National
Security, LLC for the U.S. Department of Energy. The U.S. Government has rights
to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to
produce derivative works, such modified software should be clearly marked, so
as not to confuse it with the version available from LANL.

Additionally, this library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License, v. 3., as
published by the Free Software Foundation. Accordingly, this library is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
