Developers might be interested in directly linking their git checkout to the python site-packages. First, clone the ObsPy git repository:
```bash
git clone https://github.com/obspy/obspy.git /path/to/my/obspy
```
Make sure all of [[ObsPy's dependencies are installed|Installation on Linux: Dependencies]] and then install it in Python site-packages (will install for current Python interpreter, check with `which python`):
```bash
cd /path/to/my/obspy
cd misc/scripts
./develop.sh
```