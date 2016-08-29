#!/bin/bash

CURDIR=`pwd`
DATETIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
LOG_DIR_BASE=logs/$DATETIME
mkdir -p $LOG_DIR_BASE

DOCKER_REPOSITORY=obspy-deb-packaging
DOCKER_IMAGES="debian_7_wheezy debian_7_wheezy_32bit debian_8_jessie ubuntu_12_04_precise ubuntu_14_04_trusty ubuntu_16_04_xenial"

# Parse the target for deb package building (e.g. "-tmegies:deb_1.0.2")
while getopts "t:" opt; do
    case "$opt" in
    t)  TARGET=(${OPTARG//:/ })
        REPO=${TARGET[0]}
        SHA=${TARGET[1]}
        TARGET=true
        OBSPY_DOCKER_TEST_SOURCE_TREE="clone"
        extra_args=', "-f '$REPO' -t '$SHA'"'
        ;;
    esac
done
shift $(expr $OPTIND - 1 )

# This bracket is closed at the very end and causes a redirection of everything
# to the logfile as well as stdout.
{
# Delete all but the last 5 log directories. The `+6` is intentional. Fully
# POSIX compliant version adapted from http://stackoverflow.com/a/34862475/1657047
ls -tp logs | tail -n +6 | xargs -I % rm -rf -- logs/%

OBSPY_PATH=$(dirname $(dirname $(pwd)))

DOCKERFILE_FOLDER=base_images
TEMP_PATH=temp
NEW_OBSPY_PATH=$TEMP_PATH/obspy

# Determine the docker binary name. The official debian packages use docker.io
# for the binary's name due to some legacy docker package.
DOCKER=`which docker.io || which docker`

# Execute Python once and import ObsPy to trigger building the RELEASE-VERSION
# file.
python -c "import obspy"

# Create temporary folder.
rm -rf $TEMP_PATH
mkdir -p $TEMP_PATH

# Copy ObsPy to the temp path. This path is the execution context of the Docker images.
mkdir -p $NEW_OBSPY_PATH
# depending on env variable OBSPY_DOCKER_TEST_SOURCE_TREE ("cp" or "clone")
# we either copy the obspy tree (potentially with local changes) or
# `git clone` from it for a tree free of local changes
if [ ! "$OBSPY_DOCKER_TEST_SOURCE_TREE" ]
then
    # default to "cp" to not change default behavior
    OBSPY_DOCKER_TEST_SOURCE_TREE="cp"
fi
if [ "$OBSPY_DOCKER_TEST_SOURCE_TREE" == "cp" ]
then
    cp -r $OBSPY_PATH/obspy $NEW_OBSPY_PATH/obspy/
    cp $OBSPY_PATH/setup.py $NEW_OBSPY_PATH/setup.py
    cp $OBSPY_PATH/MANIFEST.in $NEW_OBSPY_PATH/MANIFEST.in
    rm -f $NEW_OBSPY_PATH/obspy/lib/*.so
    COMMIT=`cd $OBSPY_PATH && git log -1 --pretty=format:%H`
elif [ "$OBSPY_DOCKER_TEST_SOURCE_TREE" == "clone" ]
then
    git clone file://$OBSPY_PATH $NEW_OBSPY_PATH
    # we're cloning so we have a non-dirty version actually
    cat $OBSPY_PATH/obspy/RELEASE-VERSION | sed 's#\.dirty$##' > $NEW_OBSPY_PATH/obspy/RELEASE-VERSION
    if [ "$TARGET" = true ] ; then
        # get a fresh and clean obspy main repo clone (e.g. to avoid unofficial
        # tags tampering with version number lookup)
        rm -rf $NEW_OBSPY_PATH
        git clone git://github.com/obspy/obspy $NEW_OBSPY_PATH || exit 1
        # be nice, make sure to only run git commands when successfully changed
        # to new temporary clone, exit otherwise
        cd $NEW_OBSPY_PATH || exit 1
        if [ "$REPO" != "obspy" ]
        then
            git remote add $REPO git://github.com/$REPO/obspy
            git fetch $REPO
        fi
        # everything comes from a clean clone, so there should be no need to
        # git-clean the repo
        git checkout $SHA
        git status
        cd $CURDIR
        # write RELEASE-VERSION file in temporary obspy clone without
        # installation, same magic as done in setup.py
        python -c "import os, sys; sys.path.insert(0, os.path.join(\"${NEW_OBSPY_PATH}\", 'obspy', 'core', 'util')); from version import get_git_version; sys.path.pop(0); print(get_git_version())" > $NEW_OBSPY_PATH/obspy/RELEASE-VERSION
        cat $NEW_OBSPY_PATH/obspy/RELEASE-VERSION
    fi
    COMMIT=`cd $NEW_OBSPY_PATH && git log -1 --pretty=format:%H`
else
    echo "Bad value for OBSPY_DOCKER_TEST_SOURCE_TREE: $OBSPY_DOCKER_TEST_SOURCE_TREE"
    exit 1
fi
cd $CURDIR
FULL_VERSION=`cat $NEW_OBSPY_PATH/obspy/RELEASE-VERSION`

# Copy the install script.
cp scripts/package_debs.sh $TEMP_PATH/package_debs.sh


# Helper function checking if an element is in an array.
list_not_contains() {
    for word in $1; do
        [[ $word == $2 ]] && return 1
    done
    return 0
}


# Function creating an image if it does not exist.
create_image () {
    image_name=$1;
    has_image=$($DOCKER images | grep ${DOCKER_REPOSITORY} | grep $image_name)
    if [ "$has_image" ]; then
        printf "\e[101m\e[30m  >>> Image '$image_name' already exists.\e[0m\n"
    else
        printf "\e[101m\e[30m  Image '$image_name' will be created.\e[0m\n"
        $DOCKER build -t ${DOCKER_REPOSITORY}:$image_name $image_path
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

    $DOCKER build -t temp:temp $TEMP_PATH

    $DOCKER run --name=$ID temp:temp

    $DOCKER cp $ID:/LOG.txt $LOG_DIR
    $DOCKER cp $ID:/failure $LOG_DIR
    $DOCKER cp $ID:/success $LOG_DIR

    $DOCKER cp $ID:/tmp/python-obspy_build/packages $LOG_DIR/packages

    $DOCKER rm $ID
    $DOCKER rmi temp:temp
}


# 1. Build all the base images if they do not yet exist.
printf "\e[44m\e[30mSTEP 1: CREATING BASE IMAGES\e[0m\n"

for image_name in ${DOCKER_IMAGES}; do
    image_path=${DOCKERFILE_FOLDER}/${image_name}
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
for image_name in $($DOCKER images | grep ${DOCKER_REPOSITORY} | awk '{print $2}'); do
    if [ $# != 0 ]; then
        if list_not_contains "$*" $image_name; then
            continue
        fi
    fi
    package_debs_on_image $image_name;
done

# helper function to determine overall success/failure across all images
overall_status() {
    ls ${LOG_DIR_BASE}/*/failure 2>&1 > /dev/null && return 1
    ls ${LOG_DIR_BASE}/*/success 2>&1 > /dev/null && return 0
    return 1
}

if overall_status ;
then
    echo "Overall status: Deb packaging succeeded"
else
    echo "Overall status: Deb packaging failed"
fi

rm -rf $TEMP_PATH

} 2>&1 | tee -a $LOG_DIR_BASE/docker.log
