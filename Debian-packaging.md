Here is a short walkthrough for doing the Debian packaging of ObsPy. It uses chroot/schroot and debootstrap to build packages for all distributions and architectures on a 64bit Debian host.

 - Create schroot/debootstrap environments:
```sh
# add testing to sources.list, we need a more recent debootstrap that knows
# the more recent ubuntu releases...
sudo cat >> /etc/apt/sources.list <<'EOT'

deb http://ftp.de.debian.org/debian/ testing main # REMOVE AGAIN
EOT
sudo aptitude update
sudo aptitude install debootstrap -t testing
# remove our sources.list entry again, otherwise aptitude wants to update
# all our packages...
sudo ex /etc/apt/sources.list <<'EOT'
g/REMOVE AGAIN/d
wq
EOT
sudo aptitude update
sudo aptitude install schroot debootstrap

# comment out line that mounts /home in schroot,
# no need to mount that in the guest systems:
sudo ex /etc/schroot/mount-defaults <<'EOT'
g/^\/home/s/^/#/
wq
EOT

# make the config files for schroot
sudo cat > /etc/schroot/chroot.d/squeeze_i386.conf << EOT
[squeeze_i386]
description=Debian 6.0 Squeeze for i386
directory=/srv/chroot/squeeze_i386
root-users=root
type=directory
users=obspy
EOT
sudo cat > /etc/schroot/chroot.d/squeeze_amd64.conf << EOT
[squeeze_amd64]
description=Debian 6.0 Squeeze for amd64
directory=/srv/chroot/squeeze_amd64
root-users=root
type=directory
users=obspy
EOT
sudo cat > /etc/schroot/chroot.d/wheezy_i386.conf << EOT
[wheezy_i386]
description=Debian 7.0 Wheezy for i386
directory=/srv/chroot/wheezy_i386
root-users=root
type=directory
users=obspy
EOT
sudo cat > /etc/schroot/chroot.d/wheezy_amd64.conf << EOT
[wheezy_amd64]
description=Debian 7.0 Wheezy for amd64
directory=/srv/chroot/wheezy_amd64
root-users=root
type=directory
users=obspy
EOT
sudo cat > /etc/schroot/chroot.d/lucid_i386.conf << EOT
[lucid_i386]
description=Ubuntu 10.04 LTS Lucid Lynx for i386
directory=/srv/chroot/lucid_i386
root-users=root
type=directory
users=obspy
EOT
sudo cat > /etc/schroot/chroot.d/lucid_amd64.conf << EOT
[lucid_amd64]
description=Ubuntu 10.04 LTS Lucid Lynx for amd64
directory=/srv/chroot/lucid_amd64
root-users=root
type=directory
users=obspy
EOT
sudo cat > /etc/schroot/chroot.d/natty_i386.conf << EOT
[natty_i386]
description=Ubuntu 11.04 Natty Narwhal for i386
directory=/srv/chroot/natty_i386
root-users=root
type=directory
users=obspy
EOT
sudo cat > /etc/schroot/chroot.d/natty_amd64.conf << EOT
[natty_amd64]
description=Ubuntu 11.04 Natty Narwhal for amd64
directory=/srv/chroot/natty_amd64
root-users=root
type=directory
users=obspy
EOT
sudo cat > /etc/schroot/chroot.d/oneiric_i386.conf << EOT
[oneiric_i386]
description=Ubuntu 11.10 Oneiric Ocelot for i386
directory=/srv/chroot/oneiric_i386
root-users=root
type=directory
users=obspy
EOT
sudo cat > /etc/schroot/chroot.d/oneiric_amd64.conf << EOT
[oneiric_amd64]
description=Ubuntu 11.10 Oneiric Ocelot for amd64
directory=/srv/chroot/oneiric_amd64
root-users=root
type=directory
users=obspy
EOT
sudo cat > /etc/schroot/chroot.d/precise_i386.conf << EOT
[precise_i386]
description=Ubuntu 12.04 LTS Precise Pangolin for i386
directory=/srv/chroot/precise_i386
root-users=root
type=directory
users=obspy
EOT
sudo cat > /etc/schroot/chroot.d/precise_amd64.conf << EOT
[precise_amd64]
description=Ubuntu 12.04 LTS Precise Pangolin for amd64
directory=/srv/chroot/precise_amd64
root-users=root
type=directory
users=obspy
EOT

# debootstrap all currently supported debian/ubuntu distros
# this takes ~500MB per distro and architecture (after all additional installs later)
for ARCH in i386 amd64; do for DISTRO in squeeze wheezy; do
DIR=/srv/chroot/${DISTRO}_${ARCH}
sudo mkdir -p $DIR
sudo debootstrap --arch $ARCH --variant=buildd $DISTRO $DIR http://ftp.debian.org/debian/
done; done
for ARCH in i386 amd64; do for DISTRO in lucid natty oneiric precise; do
DIR=/srv/chroot/${DISTRO}_${ARCH}
sudo mkdir -p $DIR
sudo debootstrap --arch $ARCH --variant=buildd --components=main,universe $DISTRO $DIR http://archive.ubuntu.com/ubuntu/
done; done

# install additional packages necessary for building the deb files
for ARCH in i386 amd64; do for DISTRO in squeeze wheezy lucid natty oneiric precise; do
sudo cat <<'EOT'| schroot -c ${DISTRO}_${ARCH} -u root
apt-get update
apt-get install debian-archive-keyring -y --force-yes
apt-get update
apt-get install aptitude --no-install-recommends -y
aptitude update
aptitude install python -y
PYVERS=`pyversions -s`
aptitude install vim-common $PYVERS python-setuptools python-support python-numpy lsb-release gfortran -y
aptitude install ${PYVERS/ /-dev }-dev subversion fakeroot equivs lintian git git-core -R -y
aptitude clean
EOT
done; done
```
 - clone github repository and call build script
```sh
git clone https://github.com/obspy/obspy.git $HOME/obspy
$HOME/obspy/misc/scripts/build_all_debs.sh
```
 - now either..
   - create a new apt repository from scratch
```sh
$HOME/obspy/misc/debian/deb__build_repo.sh
```
   - or download the current apt repository
```sh
$HOME/obspy/misc/debian/deb__download_repo.sh
```
 - add built packages to repository
```sh
$HOME/obspy/misc/debian/deb__add_debs_to_repo.sh
```
 - upload repository
```sh
$HOME/obspy/misc/debian/deb__upload_repo.sh
```