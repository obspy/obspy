First, install ObsPy's dependencies as described [[here|Installation on Linux: Dependencies]].

Then, make sure the version of distribute is recent enough (the current developer version needs at least version 0.6.21):

```bash
easy_install -U distribute
```

To install ObsPy packages run the following command:

```bash
easy_install -N obspy
```

### Notes
 * ObsPy may be updated to a newer version using the **-U** option:
```bash
easy_install -N -U obspy
```
 * **-N**: Option will prevent easy_install to resolve the dependencies on its own (can be useful if dependencies are already installed and installing them via PyPI fails).
 * **==dev**: The latest development version can be obtained by adding ==dev to the package name, i.e. `easy_install -N obspy==dev`.