# ObsPy Docker Utilities

This directory contains a collections of scripts and Dockerfiles enabling
developers to..

 * test ObsPy on various Unix based operating systems, and
 * build ObsPy deb packages on Debian/Ubuntu distributions

It requires a working installation of [Docker](https://www.docker.com/). It has
been designed to work with a remote Docker installation so it works fine on OSX
and I guess Windows as well.


## 1. ObsPy Test Runner on Docker Images

The `run_obspy_tests.sh` script will test either

 * the **current state of the repository** on the specified Docker image, or..
 * test a clean github clone at a certain commit from either the main
   repository or a fork.

Test reports will be sent automatically to
[http://tests.obspy.org](http://tests.obspy.org).

If the respective image is not yet available it will be created automatically.
The `base_images` directory contains all available images receipts.

```bash
$ ls base_images
centos_7               debian_8_jessie           fedora_24
ubuntu_12_04_precise   ubuntu_14_04_trusty_32bit debian_7_wheezy
debian_8_jessie_32bit  opensuse_13_2             ubuntu_12_04_precise_32bit
ubuntu_16_04_xenial    debian_7_wheezy_32bit     fedora_23
opensuse_leap_42_1     ubuntu_14_04_trusty       ubuntu_16_04_xenial_32bit
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

### a) Testing the Current State of the local Repository

The following command will execute the ObsPy test suite on a CentOS 7 image.


```bash
$ ./run_obspy_tests.sh centos_7
```

Running it without any commands will execute the test suite on all available images.

```bash
$ ./run_obspy_tests.sh
```

Additionally arguments can be passed to `obspy-runtests` in the Docker images
with the `-e` argument. To for example only the MiniSEED test suite on CentOS
7, do:

```bash
$ ./run_obspy_tests.sh -eio.mseed centos_7
```
Or to also provide a pull request url to `obspy-runtests` with the `-e`
argument and using quotes (including bash variable replacement) to mark the
start and end of the `-e` option, do (again with just a single distribution)
the following:

```bash
$ export PR=1460
$ ./run_obspy_tests.sh -e"io.ndk --pr-url=https://github.com/obspy/obspy/pull/${PR}" centos_7
```

### b) Testing a Certain Clean Commit State of a Remote Repository

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

The `-t` and `-e` options can also be combined, e.g.:

```bash
$ # see http://tests.obspy.org/46691/ for the result of this command:
$ ./run_obspy_tests.sh -ttrichter:e6da3ddb5 -e"io.ndk --pr-url=https://github.com/obspy/obspy/pull/${PR}" fedora_24
```

## 2. ObsPy Deb Packaging based on Docker

The `package_debs.sh` script can be used to build `deb` packages for Debian/Ubuntu.

`deb` packages should ideally be built from a dedicated branch that fixes the
version number lookup, see e.g.
[megies deb_1.0.2 branch](https://github.com/megies/obspy/commits/deb_1.0.2).

```bash
$ ./package_debs.sh -tmegies:deb_1.0.2
```

Each execution of the script creates a new subdirectory under `logs` containing
logs for each image used to build `deb` packages, and also containing the built
packages. The resulting built images are also automatically tested inside the
Docker images and test reports are sent to
[http://tests.obspy.org](http://tests.obspy.org).

```bash
logs
└── package_debs
    └── 2016-08-29T21-30-48Z
        ├── debian_7_wheezy
        │   ├── BUILD_LOG.txt
        │   ├── packages
        │   │   ├── obspy_1.0.2-1~wheezy_amd64.changes
        │   │   ├── python-obspy_1.0.2-1~wheezy_amd64.deb
        │   │   └── python-obspy-dbg_1.0.2-1~wheezy_amd64.deb
        │   ├── success
        │   └── TEST_LOG.txt
        ├── debian_7_wheezy_32bit
        │   ├── BUILD_LOG.txt
        │   ├── packages
        │   │   ├── obspy_1.0.2-1~wheezy_i386.changes
        │   │   ├── python-obspy_1.0.2-1~wheezy_i386.deb
        │   │   └── python-obspy-dbg_1.0.2-1~wheezy_i386.deb
        │   ├── success
        │   └── TEST_LOG.txt
        ├── docker.log
        ├── ubuntu_12_04_precise
        │   ├── BUILD_LOG.txt
        │   ├── packages
        │   │   ├── obspy_1.0.2-1~precise_amd64.changes
        │   │   ├── python-obspy_1.0.2-1~precise_amd64.deb
        │   │   └── python-obspy-dbg_1.0.2-1~precise_amd64.deb
        │   ├── success
        │   └── TEST_LOG.txt
        ...
```

### Preparation of a new base image for Debian/Ubuntu

Creation of a new docker base image can be done on a recent Debian/Ubuntu
(especially with a recent debootstrap installed). The following commands were
used on a Debian 8 Jessie host.

`debootstrap` has no dependencies worth mentioning and can be updated without
problems to a newer version. Its version has to be updated from backports or
`unstable` if a new Ubuntu/Debian version is not yet known to the installed
`debootstrap` version.

```bash
$ cd /tmp
$ CODENAME=`lsb_release -cs`
$ sudo aptitude install -t ${CODENAME}-backports debootstrap  # optional, if the following doesn't work
```

When `debootstrap`ping on a Debian host, the Ubuntu archive keyring should be
used. To create an Ubuntu docker base image:

```bash
$ cd /tmp
$ sudo aptitude install ubuntu-archive-keyring
$ DISTRO=xenial
$ DISTRO_FULL=ubuntu_16_04_xenial_32bit
$ sudo debootstrap --arch=i386 --variant=minbase --components=main,universe --keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg ${DISTRO} ${DISTRO_FULL} http://archive.ubuntu.com/ubuntu 2>&1 | tee ${DISTRO_FULL}.debootstrap.log
$ sudo tar -C ${DISTRO_FULL} -c . | docker import - obspy/base-images:${DISTRO_FULL}
$ docker login  # docker hub user needs write access to "obspy/base-images" of organization "obspy"
$ docker push obspy/base-images:${DISTRO_FULL}
```

To create a Debian docker base image:

```bash
$ cd /tmp
$ DISTRO=jessie
$ DISTRO_FULL=debian_8_jessie_32bit
$ sudo debootstrap --arch=i386 --variant=minbase ${DISTRO} ${DISTRO_FULL} http://httpredir.debian.org/debian/ 2>&1 | tee ${DISTRO_FULL}.debootstrap.log
$ sudo tar -C ${DISTRO_FULL} -c . | docker import - obspy/base-images:${DISTRO_FULL}
$ docker login  # docker hub user needs write access to "obspy/base-images" of organization "obspy"
$ docker push obspy/base-images:${DISTRO_FULL}
```

To create a Raspbian (Debian ``armhf`` for Raspberry Pi) docker base image (using ``qemu``, https://wiki.debian.org/ArmHardFloatChroot).
This imports the public key for the Raspbian repository at the start, so the package integrity can be verified later on.

```bash
$ cd /tmp
$ wget https://archive.raspbian.org/raspbian.public.key -O - | sudo gpg --import -
$ DISTRO=stretch
$ DISTRO_FULL=debian_9_stretch_armhf
$ sudo qemu-debootstrap --arch=armhf --keyring /root/.gnupg/pubring.kbx ${DISTRO} ${DISTRO_FULL} http://archive.raspbian.org/raspbian 2>&1 | tee ${DISTRO_FULL}.debootstrap.log
$ sudo tar -C ${DISTRO_FULL} -c . | docker import - obspy/base-images:${DISTRO_FULL}
$ docker login  # docker hub user needs write access to "obspy/base-images" of organization "obspy"
$ docker push obspy/base-images:${DISTRO_FULL}
```

To run ``armhf`` docker images/containers built this way on non-ARM Linux machines,
it seems that it's necessary to install ``qemu``, ``qemu-user-static`` and
``qemu-system-arm`` packages (Debian/Ubuntu).
Furthermore, qemu multiarch support has to be registered with docker (see
https://hub.docker.com/r/multiarch/qemu-user-static/):

```bash
$ docker run --rm --privileged multiarch/qemu-user-static:register --reset
```

### Setting up `docker-testbot` to automatically test PRs and branches and send commit statuses

##### Install docker

Well.. install it: https://docs.docker.com/engine/installation/

##### Set up a dedicated Python environment

Set up a dedicated Anaconda Python environment and install
[`obspy_github_api`](https://github.com/obspy/obspy_github_api). Activate that
environment before running the docker testbot (in the last step of the
instructions).

##### Set up a dedicated ObsPy clone

Set up a dedicated ObsPy git clone. This clone should not be used for anything
else than running the docker testbot. `git clean -fdx` will be run, razing any
local changes. Set the location to the dedicated obspy repository:

```bash
$ export OBSPY_DOCKER_BASE=/path/to/dedicated/obspy
```

Remote 'origin' should point to obspy/obspy (obspy main repository).

##### Register an OAuth token on github

Login to https://github.com and create a dedicated OAuth token. It should only
have rights for "repo:status" on obspy/obspy. Set token as env variable:

```bash
$ export OBSPY_COMMIT_STATUS_TOKEN=abcdefgh123456789
```

##### Run docker testbot

To run docker testbot, simply do:

```bash
$ bash cronjob_docker_tests.sh -t -d -b -p
```

This will run both..

 - docker testing (-t)
 - docker deb packaging/testing  (-d)

..on both:

 - main branches, like master and maintenance_1.0.x (-b)
 - pull requests (-p)

Since runtime is rather high, ideally these jobs should be distributed on
separate docker testrunners (scripts might have to be adjusted, e.g. naming of
docker temporary containers..)

To run four workers embarassingly parallel (by separating the different build
types), set up four dedicated obspy github repository clones and run the jobs
in parallel:

```bash
$ git clone git://github.com/obspy/obspy /path/to/obspy/dockers/test-pr
$ git clone git://github.com/obspy/obspy /path/to/obspy/dockers/test-branches
$ git clone git://github.com/obspy/obspy /path/to/obspy/dockers/deb-pr
$ git clone git://github.com/obspy/obspy /path/to/obspy/dockers/deb-branches
$ OBSPY_DOCKER_BASE=/path/to/obspy/dockers/test-pr ./cronjob_docker_tests.sh -t -p &
$ OBSPY_DOCKER_BASE=/path/to/obspy/dockers/test-branches ./cronjob_docker_tests.sh -t -b &
$ OBSPY_DOCKER_BASE=/path/to/obspy/dockers/deb-pr ./cronjob_docker_tests.sh -d -p &
$ OBSPY_DOCKER_BASE=/path/to/obspy/dockers/deb-branches ./cronjob_docker_tests.sh -d -b &
```

# Release Lifecycle Information

 * Debian: https://wiki.debian.org/DebianReleases#Production_Releases
 * Ubuntu: https://wiki.ubuntu.com/Releases#Current
 * openSUSE: https://en.opensuse.org/Lifetime#Maintained_Regular_distributions
 * CentOS: https://en.wikipedia.org/wiki/CentOS#End-of-support_schedule
 * Fedora: https://fedoraproject.org/wiki/Releases#Current_Supported_Releases

# Lookup available packages:

 * Debian: https://www.debian.org/distrib/packages
 * Ubuntu: https://packages.ubuntu.com/
 * openSUSE: https://software.opensuse.org/find
 * CentOS: http://mirror.centos.org/centos/7/os/x86_64/Packages/
 * Fedora: https://fedoraproject.org/wiki/Releases#Current_Supported_Release://admin.fedoraproject.org/pkgdb/packages/
