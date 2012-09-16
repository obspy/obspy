## Python

Get and run the latest official Python 2.7.x installer from  http://www.python.org/download/ (Python 3.x is NOT supported for now).

The Python binary and the Scripts directory should be set within the system PATH variable. Therefore append the following to the system PATH
```
;C:\path\to\Python2.7;C:\path\to\Python2.7\Scripts
```
See  http://docs.python.org/using/windows.html#excursus-setting-environment-variables for a short tutorial on changing environment variables in Windows.
Dependencies

 1. Using pre-compiled executable installers are the easiest way to install the needed packages
    NumPy, SciPy and matplotlib.
      * The official 32-bit installers are available at:
        * http://sourceforge.net/projects/numpy/files/
        * http://sourceforge.net/projects/scipy/files/
        * http://sourceforge.net/projects/matplotlib/files/matplotlib/
      * or fetch the unofficial Windows 64 bit releases from
        http://www.lfd.uci.edu/~gohlke/pythonlibs/ (get the MKL builds for NumPy).

 1. ObsPy and further dependencies can be downloaded via easy_install of the
    Distribute package. Download and run from the windows command line the 
    Python script  http://python-distribute.org/distribute_setup.py. You may
    also find the file in the ObsPy repository via 
    https://svn.obspy.org/trunk/obspy.core/distribute_setup.py.
    ```bash
    python.exe distribute_setup.py
    ```

 1. For obspy.xseed and obspy.arclink the Python module  lxml is needed.
    Run from windows command line:
    ```bash
    easy_install lxml
    ```

 1. We strongly recommend the enhanced Python shell  IPython which can be obtained via:
    ```bash
    easy_install pyreadline
    easy_install ipython
    ```

## ObsPy

For installing basic packages able to handle GSE2, SAC and MSEED files run on command line:
```bash
easy_install obspy.core
easy_install obspy.mseed
easy_install obspy.sac
easy_install obspy.gse2
easy_install obspy.imaging
```

Additional modules of ObsPy may be installed by calling easy_install followed by the module name.

```bash
easy_install obspy.signal
easy_install obspy.arclink
easy_install obspy.xseed
easy_install obspy.seishub
easy_install obspy.seisan
easy_install obspy.wav
easy_install obspy.sh
easy_install obspy.iris
easy_install obspy.neries
...
```

Installed packages may be updated at any time using the -U option, e.g.:
```bash
easy_install -U obspy.core
```