# ObsPy Test Runner on Docker Images

This directory contains a collections of scripts and Dockerfiles enabling
developers to test ObsPy on various Unix based operating systems.

It requires a working installation of [Docker](https://www.docker.com/). It has
been designed to work with a remote Docker installation so it works fine on OSX
and I guess Windows as well.

The `run_obspy_tests.sh` script will test the **current state of the
repository** on the specified Docker image. The following command will execute
the ObsPy test suite on a CentOS 7 image.


```bash
$ ./run_obspy_tests.sh centos_7
```

Running it without any commands will execute the test suite on all available images.

```bash
$ ./run_obspy_tests.sh
```

Additionally arguments can be passed to `obspy-runtests` in the Docker images
with the `-e` argument. To for example only the MiniSEED test suite on CentOS
7, do

```bash
$ ./run_obspy_tests.sh -eio.mseed centos_7
```

A specific commit from a specific obspy fork (or main repo) can be tested using
the `-t` (for "target") argument.

Make sure to use the `-e` argument before the list of images to run on.

```bash
$ # test a commit that is in the obspy main repo
$ ./run_obspy_tests.sh -tobspy:bdc6dd855c00c831bcc007b607d83f6070b5b1c0
$ # test a commit that only exists in a fork (but might be the tip of a PR)
$ ./run_obspy_tests.sh -tclaudiodsf:2fa3d3bdaded126a9ebdaf73cf60403c1acb3457
```

Make sure to use the `-t` argument before the list of images to run on.

If the image is not yet available it will be created automatically. The
`base_images` directory contains all available images receipts.

```bash
$ ls base_images
centos_7                debian_8_jessie   opensuse_13_2          ubuntu_14_04_trusty
debian_7_wheezy         fedora_22         opensuse_leap          ubuntu_15_10_wily
debian_7_wheezy_32bit   fedora_23         ubuntu_12_04_precise
```

Each directory in `base_images` contains a `Dockerfile` with instructions to
install all ObsPy dependencies without actually installing ObsPy itsself. This
makes it easy to add more test images.

Each execution of the `run_obspy_tests.sh` script creates a new subdirectory
under `logs` containing logs for each tested image.

```bash
$ tree logs
logs
└── 2014-06-22T20:35:25Z
    ├── centos_7
    │   ├── INSTALL_LOG.txt
    │   └── TEST_LOG.txt
    ├── debian_7_wheezy
    │   ├── INSTALL_LOG.txt
    │   └── TEST_LOG.txt
    ├── debian_8_jessie
    │   ├── INSTALL_LOG.txt
    │   └── TEST_LOG.txt
    ├── fedora_22
    │   ├── INSTALL_LOG.txt
    │   └── TEST_LOG.txt
    ...
```
