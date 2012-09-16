## _**Note: Information on this page might be outdated**_

## Python

Get and run the latest official Python 2.7.x installer from  http://www.python.org/download/ (Python 3.x is NOT supported for now).

The Python binary and the Scripts directory should be set within the system PATH variable. Therefore append the following to the system PATH
```
;C:\path\to\Python2.7;C:\path\to\Python2.7\Scripts
```
See  http://docs.python.org/using/windows.html#excursus-setting-environment-variables for a short tutorial on changing environment variables in Windows.

## C Compiler

Building the modules obspy.gse2, obspy.mseed and obspy.signal require a C compiler. Those modules have been successfully built using the free GNU compiler of the MinGW project (32- and 64-bit) and Microsoft Visual Studio (32- and 64-bit).

## Git Client

In order to check out the latest developer packages from ObsPy you need a Git command line client. You may find a binary installers e.g. at http://git-scm.com/downloads.

Please make sure that the directory to the Git client is also included in the PATH settings (for details see step 1), e.g. add
```
;C:\path\to\GitClient\bin;%PATH%
```

## Dependencies

  1. Using pre-compiled executable installers are the easiest way to install the needed packages  NumPy,  SciPy and  matplotlib.
    * The official 32-bit installers are available at:
      *   http://sourceforge.net/projects/numpy/files/
      *   http://sourceforge.net/projects/scipy/files/
      *   http://sourceforge.net/projects/matplotlib/files/matplotlib/
    * or fetch the unofficial Windows 64 bit releases from  http://www.lfd.uci.edu/~gohlke/pythonlibs/ (get the MKL builds for NumPy?).

  1.  ObsPy and further dependencies can be downloaded via easy_install of the  Distribute package. Download and run from the windows command line the Python script  http://python-distribute.org/distribute_setup.py.
      ```bash
python.exe distribute_setup.py
      ```
  1.  For obspy.xseed and obspy.arclink the Python module  lxml is needed. Run from windows command line:
      ```bash
easy_install lxml
      ```
  1.  We strongly recommend the enhanced Python shell  IPython which can be obtained via:
      ```bash
easy_install pyreadline
easy_install ipython
      ```

## ObsPy

Checkout the latest sources using a Git client on command line:
```
git clone https://github.com/obspy/obspy.git obspy
```
All ObsPy packages must now be linked to the current Python instance by running the following command within each module:
```
python setup.py develop
```
Note: Calling python setup.py develop in each module is quite a workout. Therefore we provide a simple helper script develop.bat (Windows) in the trunk/misc/scripts directory. You may have to modify the Path to your Python Installation in that script before running it. 