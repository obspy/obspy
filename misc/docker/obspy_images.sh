# all base images of obspy
OBSPY_DOCKER_BASE_IMAGES="
centos:centos7
debian:jessie
debian:stretch
debian:wheezy
fedora:24
fedora:25
obspy/base-images:debian_7_wheezy_32bit
obspy/base-images:debian_8_jessie_32bit
obspy/base-images:debian_9_stretch_32bit
obspy/base-images:ubuntu_14_04_trusty_32bit
obspy/base-images:ubuntu_16_04_xenial_32bit
obspy/base-images:ubuntu_16_10_yakkety_32bit
obspy/base-images:ubuntu_17_04_zesty_32bit
opensuse:42.1
opensuse:42.2
ubuntu:14.04
ubuntu:16.04
obspy/base-images:debian_7_wheezy_armhf
obspy/base-images:debian_8_jessie_armhf
obspy/base-images:debian_9_stretch_armhf"

# docker images used for testing, excluding Debian/Ubuntu and excluding ARM
OBSPY_DOCKER_IMAGES_WITHOUT_DEBUNTU="
obspy:centos_7
obspy:fedora_24
obspy:fedora_25
obspy:opensuse_leap_42_1
obspy:opensuse_leap_42_2"
# docker images used for testing and deb packaging, excluding ARM
OBSPY_DOCKER_IMAGES_DEBUNTU_WITHOUT_ARM="
obspy:debian_7_wheezy
obspy:debian_7_wheezy_32bit
obspy:debian_8_jessie
obspy:debian_8_jessie_32bit
obspy:debian_9_stretch
obspy:debian_9_stretch_32bit
obspy:ubuntu_14_04_trusty
obspy:ubuntu_14_04_trusty_32bit
obspy:ubuntu_16_04_xenial
obspy:ubuntu_16_04_xenial_32bit
obspy:ubuntu_16_10_yakkety
obspy:ubuntu_16_10_yakkety_32bit
obspy:ubuntu_17_04_zesty
obspy:ubuntu_17_04_zesty_32bit"
# docker images used for testing and deb packaging, ARM only
OBSPY_DOCKER_IMAGES_DEBUNTU_ARM="
obspy:debian_7_wheezy_armhf
obspy:debian_8_jessie_armhf
obspy:debian_9_stretch_armhf"

# all images used for deb package building, both non-ARM and ARM
OBSPY_DOCKER_IMAGES_DEBUNTU_WITH_ARM="$OBSPY_DOCKER_IMAGES_DEBUNTU_WITHOUT_ARM $OBSPY_DOCKER_IMAGES_DEBUNTU_ARM"

# all images used for testing and deb package building, excluding ARM
OBSPY_DOCKER_IMAGES_WITHOUT_ARM="$OBSPY_DOCKER_IMAGES_WITHOUT_DEBUNTU $OBSPY_DOCKER_IMAGES_DEBUNTU_WITHOUT_ARM"
# all images used for testing and deb package building, both non-ARM and ARM
OBSPY_DOCKER_IMAGES_WITH_ARM="$OBSPY_DOCKER_IMAGES_WITHOUT_DEBUNTU $OBSPY_DOCKER_IMAGES_DEBUNTU_WITH_ARM"

# all images involved in obspy docker actions, base images and actual images used in testing/deb packaging
OBSPY_DOCKER_ALL_IMAGES="$OBSPY_DOCKER_BASE_IMAGES $OBSPY_DOCKER_IMAGES_WITH_ARM"
