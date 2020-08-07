#!/bin/bash

CURDIR=`pwd`
DATETIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
LOG_DIR_ROOT=logs/package_debs
LOG_DIR_BASE=$LOG_DIR_ROOT/$DATETIME
mkdir -p $LOG_DIR_BASE

DOCKER_IMAGES="debian_8_jessie debian_8_jessie_32bit debian_8_jessie_armhf debian_9_stretch debian_9_stretch_32bit debian_9_stretch_armhf debian_10_buster debian_10_buster_32bit debian_10_buster_armhf ubuntu_14_04_trusty ubuntu_14_04_trusty_32bit ubuntu_16_04_xenial ubuntu_16_04_xenial_32bit ubuntu_18_04_bionic ubuntu_18_04_bionic_32bit"

# Parse the target for deb package building (e.g. "-tmegies:deb_1.0.2")
SET_COMMIT_STATUS=false
while getopts "t:s" opt; do
    case "$opt" in
    t)  TARGET=(${OPTARG//:/ })
        REPO=${TARGET[0]}
        GITTARGET=${TARGET[1]}
        TARGET=true
        extra_args=', "-f '$REPO' -t '$GITTARGET'"'
        ;;
    s)  SET_COMMIT_STATUS=true;;
    esac
done
shift $(expr $OPTIND - 1 )

# This bracket is closed at the very end and causes a redirection of everything
# to the logfile as well as stdout.
{
# Delete all but the last 3 log directories. The `+4` is intentional. Fully
# POSIX compliant version adapted from http://stackoverflow.com/a/34862475/1657047
ls -tp $LOG_DIR_ROOT | tail -n +4 | xargs -I % rm -rf -- $LOG_DIR_ROOT/%

OBSPY_PATH=$(dirname $(dirname $(pwd)))

DOCKERFILE_FOLDER=base_images
TEMP_PATH=temp/$RANDOM-$RANDOM-$RANDOM

# Determine the docker binary name. The official debian packages use docker.io
# for the binary's name due to some legacy docker package.
DOCKER=`which docker.io || which docker`

# Create temporary folder.
rm -rf $TEMP_PATH
mkdir -p $TEMP_PATH

# but we need to find out the SHA of the git target (e.g. if a branch name was
# specified..)
git clone git://github.com/$REPO/obspy $TEMP_PATH/obspy
cd $TEMP_PATH/obspy
git checkout $GITTARGET
SHA=`git log -1 --pretty=format:'%H'`
cd $CURDIR
# Deb packaging is always performed using the deb build script from misc/debian
# of the target branch
cp -a $TEMP_PATH/obspy/misc/debian/deb__build_debs.sh $TEMP_PATH/
rm -rf $TEMP_PATH/obspy

cd $CURDIR

# Copy the install script. Use current version (on master)
cp scripts/package_debs.sh $TEMP_PATH/package_debs.sh


# Helper function checking if an element is in an array.
list_not_contains() {
    for word in $1; do
        [[ $word == $2 ]] && return 1
    done
    return 0
}


# Function to check if an image exists
image_exists () {
    # "docker images" does not provide useful return codes.. so use a dummy format string and grep
    if $DOCKER images obspy:$1 --format 'EXISTS' | grep 'EXISTS' 2>&1 > /dev/null ; then
        return 0
    else
        return 1
    fi
}


# Function creating an image if it does not exist.
create_image () {
    image_name=$1;
    if image_exists $image_name ; then
        printf "\e[101m\e[30m  >>> Image '$image_name' already exists.\e[0m\n"
    else
        printf "\e[101m\e[30m  Image '$image_name' will be created.\e[0m\n"
        $DOCKER build -t obspy:$image_name ${DOCKERFILE_FOLDER}/${image_name}
    fi
}


# Function building deb packages on an image.
package_debs_on_image () {
    image_name=$1;
    printf "\n\e[101m\e[30m  >>> Packaging debs for image '"$image_name"'...\e[0m\n"
    # Copy dockerfile and render template.
    sed "s#{{IMAGE_NAME}}#$image_name#g; s#{{EXTRA_ARGS}}#$extra_args#g" scripts/Dockerfile_package_debs.tmpl > $TEMP_PATH/Dockerfile

    # Where to save the logs, and a random ID for the containers.
    LOG_DIR=${LOG_DIR_BASE}/$image_name
    mkdir -p $LOG_DIR
    ID=$RANDOM-$RANDOM-$RANDOM
    TAG=package_debs_$RANDOM-$RANDOM-$RANDOM

    $DOCKER build -t temp:$TAG $TEMP_PATH

    $DOCKER run --name=$ID temp:$TAG

    $DOCKER cp $ID:/BUILD_LOG.txt $LOG_DIR
    $DOCKER cp $ID:/TEST_LOG.txt $LOG_DIR
    $DOCKER cp $ID:/build.failure $LOG_DIR
    $DOCKER cp $ID:/build.success $LOG_DIR
    $DOCKER cp $ID:/test.failure $LOG_DIR
    $DOCKER cp $ID:/test.success $LOG_DIR

    $DOCKER cp $ID:/tmp/python-obspy_build/packages $LOG_DIR/packages

    $DOCKER rm $ID
    $DOCKER rmi temp:$TAG
}


# 1. Build all the base images if they do not yet exist.
printf "\e[44m\e[30mSTEP 1: CREATING BASE IMAGES\e[0m\n"

for image_name in ${DOCKER_IMAGES}; do
    if [ $# != 0 ]; then
        if list_not_contains "$*" $image_name; then
            continue
        fi
    fi
    create_image $image_name;
done


# 2. Build ObsPy Deb packages
printf "\n\e[44m\e[30mSTEP 2: BUILDING DEB PACKAGES\e[0m\n"

# Loop over all ObsPy Docker images.
for image_name in $($DOCKER images obspy --format '{{.Tag}}'); do
    # skip images that are not in our Debian/Ubuntu build targets
    if list_not_contains "${DOCKER_IMAGES}" $image_name; then
        continue
    fi
    # skip images that haven't been created yet
    if ! image_exists $image_name ; then
        continue
    fi
    # if any specific targets are specified on command line..
    if [ $# != 0 ]; then
        # skip image, if not in list specified on command line
        if list_not_contains "$*" $image_name; then
            continue
        fi
    fi
    package_debs_on_image $image_name;
done

# helper function to determine overall success/failure of packaging, across all images
overall_build_status() {
    ls ${LOG_DIR_BASE}/*/build.failure 2>&1 > /dev/null && return 1
    ls ${LOG_DIR_BASE}/*/build.success 2>&1 > /dev/null && return 0
    return 1
}

# helper function to determine overall success/failure of testsuite on packages, across all images
overall_test_status() {
    ls ${LOG_DIR_BASE}/*/test.failure 2>&1 > /dev/null && return 1
    ls ${LOG_DIR_BASE}/*/test.success 2>&1 > /dev/null && return 0
    return 1
}

COMMIT_STATUS_TARGET_URL="http://tests.obspy.org/?git=${SHA}&node=docker-deb-"
if overall_build_status ;
then
    if overall_test_status ;
    then
        COMMIT_STATUS=success
        COMMIT_STATUS_DESCRIPTION="Deb packaging and testing succeeded"
    else
        COMMIT_STATUS=failure
        COMMIT_STATUS_DESCRIPTION="Deb packaging succeeded but tests failed"
    fi
else
    COMMIT_STATUS=error
    COMMIT_STATUS_DESCRIPTION="Deb packaging failed"
fi
echo $COMMIT_STATUS_DESCRIPTION

if [ "$SET_COMMIT_STATUS" = true ] ; then
    (python -c 'from obspy_github_api import __version__; assert [int(x) for x in __version__.split(".")[:2]] >= [0, 5]' && python -c "from obspy_github_api import set_commit_status; set_commit_status(commit='${SHA}', status='${COMMIT_STATUS}', context='docker-deb-buildbot', description='${COMMIT_STATUS_DESCRIPTION}', target_url='${COMMIT_STATUS_TARGET_URL}')" && echo "Set commit status for '$SHA' to '$COMMIT_STATUS' with description '$COMMIT_STATUS_DESCRIPTION' and target url '$COMMIT_STATUS_TARGET_URL'") || echo "Failed to set commit status"
fi

rm -rf $TEMP_PATH

} 2>&1 | tee -a $LOG_DIR_BASE/docker.log
