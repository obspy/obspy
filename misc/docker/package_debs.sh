#!/bin/bash

# import list of images used by obspy docker scripts
source obspy_images.sh

CURDIR=`pwd`
DATETIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
LOG_DIR_ROOT=logs/package_debs
LOG_DIR_BASE=$LOG_DIR_ROOT/$DATETIME
mkdir -p $LOG_DIR_BASE

DOCKER_IMAGES=$OBSPY_DOCKER_IMAGES_DEBUNTU_NON_ARM

# Parse the target for deb package building (e.g. "-tmegies:deb_1.0.2")
SET_COMMIT_STATUS=false
while getopts "t:sa" opt; do
    case "$opt" in
    t)  TARGET=(${OPTARG//:/ })
        REPO=${TARGET[0]}
        GITTARGET=${TARGET[1]}
        TARGET=true
        extra_args=', "-f '$REPO' -t '$GITTARGET'"'
        ;;
    s)  SET_COMMIT_STATUS=true;;
    a)  DOCKER_IMAGES="$OBSPY_DOCKER_IMAGES_DEBUNTU"
        echo 'Setting up qemu for docker for ARM images.'
        $DOCKER run --rm --privileged multiarch/qemu-user-static:register --reset
        ;;
    esac
done
shift $(expr $OPTIND - 1 )

# override list of images to run if specified on command line
if [ $# != 0 ]; then
    DOCKER_IMAGES="$*"
fi

echo "Running the following images:"
echo $DOCKER_IMAGES

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

# Deb packaging is always performed using the deb build script from misc/debian
# from the current state of the local repository.
cp -a $OBSPY_PATH/misc/debian/deb__build_debs.sh $TEMP_PATH/
# but we need to find out the SHA of the git target (e.g. if a branch name was
# specified..)
git clone git://github.com/$REPO/obspy $TEMP_PATH/obspy
cd $TEMP_PATH/obspy
git checkout $GITTARGET
SHA=`git log -1 --pretty=format:'%H'`
cd $CURDIR
rm -rf $TEMP_PATH/obspy

cd $CURDIR

# Copy the install script.
cp scripts/package_debs.sh $TEMP_PATH/package_debs.sh


# Function creating an image if it does not exist.
create_image () {
    image_name=$1;
    image_path="${DOCKERFILE_FOLDER}/${image_name#obspy:}"
    has_image=$($DOCKER images --format '{{.Repository}}:{{.Tag}}' | grep $image_name)
    if [ "$has_image" ]; then
        printf "\e[101m\e[30m  >>> Image '$image_name' already exists.\e[0m\n"
    else
        printf "\e[101m\e[30m  Image '$image_name' will be created.\e[0m\n"
        $DOCKER build -t $image_name $image_path
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
    create_image $image_name;
done


# 2. Build ObsPy Deb packages
printf "\n\e[44m\e[30mSTEP 2: BUILDING DEB PACKAGES\e[0m\n"

# Loop over all ObsPy Docker images.
for image_name in ${DOCKER_IMAGES}; do
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
