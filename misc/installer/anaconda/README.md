# How to build (Ana)Conda packages

Install the necessary tools with

```bash
$ conda install anaconda-client conda-build
```

Turn off automatic uploading to anaconda.org:

```bash
conda config --set anaconda_upload no
```

You also need to login to anaconda once thus you need an account and you have to be part of the ObsPy organization on anaconda.

```bash
$ anaconda login
```

Once in a while its also a good idea to clean the `conda-bld` directory in the root conda path. `$ conda info` shows the location of that paths. Lots of things are cached there and if the version number of the conda package does not change it will reuse previously downloaded files.


## For Linux Using Docker

**NOTE:** Linux 64 bit conda packages are now in general built via conda-forge:
https://github.com/conda-forge/obspy-feedstock/

This must be done on a Linux with a fairly old `libc` version. We currently do
it for `libc 2.5`, thus it is done in Docker containers (Centos5).

Make sure Docker is installed and running (also that some space is left on your disc or boot2docker's VM!). Execute

```bash
$ ./build_conda_packages_linux.sh
```

This will take a while but it will build packages for all Python versions on 32
bit (uncomment 64 bit section in build script if needed). It will copy the
final packages to `conda_builds` so as a last step you have to upload them from
your local machine.

```bash
$ cd conda_builds
$ anaconda upload -u obspy linux-32/obspy*
$ anaconda upload -u obspy linux-64/obspy*
```

## For OSX

**NOTE:** OSX conda packages are now in general built via conda-forge:
https://github.com/conda-forge/obspy-feedstock/

On OSX just execute (from this directory)

```bash
$ conda build --py 27 --py 33 --py 34 --py 35 obspy
```

Afterwards check (on of the last lines in the output) where the packages are stored and

```bash
$ cd /path/to/packages
$ anaconda upload -u obspy --channel docker --channel main obspy*
```

## For Windows

**NOTE:** OSX conda packages are now in general built via conda-forge:
https://github.com/conda-forge/obspy-feedstock/
