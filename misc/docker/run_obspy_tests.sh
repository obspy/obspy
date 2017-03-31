#!/bin/bash

CURDIR=`pwd`
DATETIME=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
LOG_DIR_ROOT=logs/run_tests
LOG_DIR_BASE=$LOG_DIR_ROOT/$DATETIME
mkdir -p $LOG_DIR_BASE

DOCKER_REPOSITORY=obspy

# Parse the additional args later passed to `obspy-runtests` in
# the docker images.
extra_args=""
SET_COMMIT_STATUS=false
while getopts "t:e:s" opt; do
    case "$opt" in
    e)  extra_args=', "'$OPTARG'"'
        ;;
    t)  TARGET=(${OPTARG//:/ })
        REPO=${TARGET[0]}
        GITTARGET=${TARGET[1]}
        TARGET=true
        OBSPY_DOCKER_TEST_SOURCE_TREE="clone"
        ;;
    s)  SET_COMMIT_STATUS=true;;
    esac
done
shift $(expr $OPTIND - 1 )

# This bracket is closed at the very end and causes a redirection of everything
# to the logfile as well as stdout.
{
# Delete all but the last 15 log directories. The `+16` is intentional. Fully
# POSIX compliant version adapted from http://stackoverflow.com/a/34862475/1657047
ls -tp $LOG_DIR_ROOT | tail -n +16 | xargs -I % rm -rf -- $LOG_DIR_ROOT/%

OBSPY_PATH=$(dirname $(dirname $(pwd)))

# Remove all test images stored locally. Otherwise they'll end up on the
# images.
rm -rf $OBSPY_PATH/obspy/core/tests/images/testrun
rm -rf $OBSPY_PATH/obspy/imaging/tests/images/testrun


DOCKERFILE_FOLDER=base_images
TEMP_PATH=temp/$RANDOM-$RANDOM-$RANDOM
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
    SHA=`cd $OBSPY_PATH && git log -1 --pretty=format:%H`
elif [ "$OBSPY_DOCKER_TEST_SOURCE_TREE" == "clone" ]
then
    git clone file://$OBSPY_PATH $NEW_OBSPY_PATH
    # we're cloning so we have a non-dirty version actually
    cat $OBSPY_PATH/obspy/RELEASE-VERSION | sed 's#\.dirty$##' > $NEW_OBSPY_PATH/obspy/RELEASE-VERSION
    if [ "$TARGET" = true ] ; then
        # get a fresh and clean obspy main repo clone (e.g. to avoid unofficial
        # tags tampering with version number lookup)
        rm -rf $NEW_OBSPY_PATH
        git clone git://github.com/$REPO/obspy $NEW_OBSPY_PATH || exit 1
        # be nice, make sure to only run git commands when successfully changed
        # to new temporary clone, exit otherwise
        cd $NEW_OBSPY_PATH || exit 1
        if [ "$REPO" != "obspy" ]
        then
            git remote add obspy git://github.com/obspy/obspy
            git fetch --tags obspy
        fi
        # everything comes from a clean clone, so there should be no need to
        # git-clean the repo
        git checkout $GITTARGET || exit 1
        git status
        cd $CURDIR
        # write RELEASE-VERSION file in temporary obspy clone without
        # installation, same magic as done in setup.py
        python -c "import os, sys; sys.path.insert(0, os.path.join(\"${NEW_OBSPY_PATH}\", 'obspy', 'core', 'util')); from version import get_git_version; sys.path.pop(0); print(get_git_version())" > $NEW_OBSPY_PATH/obspy/RELEASE-VERSION
        cat $NEW_OBSPY_PATH/obspy/RELEASE-VERSION
    fi
    SHA=`cd $NEW_OBSPY_PATH && git log -1 --pretty=format:%H`
else
    echo "Bad value for OBSPY_DOCKER_TEST_SOURCE_TREE: $OBSPY_DOCKER_TEST_SOURCE_TREE"
    exit 1
fi
cd $CURDIR
FULL_VERSION=`cat $NEW_OBSPY_PATH/obspy/RELEASE-VERSION`

# Copy the install script.
cp scripts/install_and_run_tests_on_image.sh $TEMP_PATH/install_and_run_tests_on_image.sh


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
    has_image=$($DOCKER images ${DOCKER_REPOSITORY} --format '{{.Tag}}' | grep $image_name)
    if [ "$has_image" ]; then
        printf "\e[101m\e[30m  >>> Image '$image_name' already exists.\e[0m\n"
    else
        printf "\e[101m\e[30m  Image '$image_name' will be created.\e[0m\n"
        $DOCKER build -t ${DOCKER_REPOSITORY}:$image_name $image_path
    fi
}


# Function running test on an image.
run_tests_on_image () {
    image_name=$1;
    printf "\n\e[101m\e[30m  >>> Running tests for image '"$image_name"'...\e[0m\n"
    # Copy dockerfile and render template.
    sed "s#{{IMAGE_NAME}}#$image_name#g; s#{{EXTRA_ARGS}}#$extra_args#g" scripts/Dockerfile_run_tests.tmpl > $TEMP_PATH/Dockerfile

    # Where to save the logs, and a random ID for the containers.
    LOG_DIR=${LOG_DIR_BASE}/$image_name
    mkdir -p $LOG_DIR
    ID=$RANDOM-$RANDOM-$RANDOM
    TAG=run_obspy_tests_$RANDOM-$RANDOM-$RANDOM

    $DOCKER build -t temp:$TAG $TEMP_PATH

    $DOCKER run --name=$ID temp:$TAG

    $DOCKER cp $ID:/INSTALL_LOG.txt $LOG_DIR
    $DOCKER cp $ID:/TEST_LOG.txt $LOG_DIR
    $DOCKER cp $ID:/failure $LOG_DIR
    $DOCKER cp $ID:/success $LOG_DIR

    $DOCKER cp $ID:/obspy/obspy/imaging/tests/images/testrun $LOG_DIR/imaging_testrun
    $DOCKER cp $ID:/obspy/obspy/core/tests/images/testrun $LOG_DIR/core_testrun
    $DOCKER cp $ID:/obspy/obspy/station/tests/images/testrun $LOG_DIR/station_testrun

    mkdir -p $LOG_DIR/test_images

    mv $LOG_DIR/imaging_testrun/testrun $LOG_DIR/test_images/imaging
    mv $LOG_DIR/core_testrun/testrun $LOG_DIR/test_images/core
    mv $LOG_DIR/station_testrun/testrun $LOG_DIR/test_images/station

    rm -rf $LOG_DIR/imaging_testrun
    rm -rf $LOG_DIR/core_testrun
    rm -rf $LOG_DIR/station_testrun

    $DOCKER rm $ID
    $DOCKER rmi temp:$TAG
}


# 1. Build all the base images if they do not yet exist.
printf "\e[44m\e[30mSTEP 1: CREATING BASE IMAGES\e[0m\n"

for image_path in $DOCKERFILE_FOLDER/*; do
    image_name=$(basename $image_path)
    if [ $# != 0 ]; then
        if list_not_contains "$*" $image_name; then
            continue
        fi
    fi
    create_image $image_name;
done


# 2. Execute the ObsPy
printf "\n\e[44m\e[30mSTEP 2: EXECUTING THE TESTS\e[0m\n"

# Loop over all ObsPy Docker images.
for image_name in $($DOCKER images ${DOCKER_REPOSITORY} --format '{{.Tag}}'); do
    if [ $# != 0 ]; then
        if list_not_contains "$*" $image_name; then
            continue
        fi
    fi
    run_tests_on_image $image_name;
done

# set commit status
# helper function to determine overall success/failure across all images
# env variable OBSPY_COMMIT_STATUS_TOKEN has to be set for authorization
overall_status() {
    ls ${LOG_DIR_BASE}/*/failure 2>&1 > /dev/null && return 1
    ls ${LOG_DIR_BASE}/*/success 2>&1 > /dev/null && return 0
    return 1
}
# encode parameter part of the URL, using requests as it is installed anyway..
# (since we use python to import obspy to generate RELEASE-VERSION above)
# it's just looking up the correct quoting function from urllib depending on
# py2/3 and works with requests >= 1.0 (which is from 2012)
FULL_VERSION_URLENCODED=`python -c "from requests.compat import quote; print(quote(\"${FULL_VERSION}\"))"`
COMMIT_STATUS_TARGET_URL="http://tests.obspy.org/?version=${FULL_VERSION_URLENCODED}&node=docker-"
if overall_status ;
then
    COMMIT_STATUS=success
    COMMIT_STATUS_DESCRIPTION="Docker tests succeeded"
else
    COMMIT_STATUS=failure
    COMMIT_STATUS_DESCRIPTION="Docker tests failed"
fi

if [ "$SET_COMMIT_STATUS" = true ] ; then
    (python -c 'from obspy_github_api import __version__; assert [int(x) for x in __version__.split(".")[:2]] >= [0, 5]' && python -c "from obspy_github_api import set_commit_status; set_commit_status(commit='${SHA}', status='${COMMIT_STATUS}', context='docker-testbot', description='${COMMIT_STATUS_DESCRIPTION}', target_url='${COMMIT_STATUS_TARGET_URL}')" && echo "Set commit status for '$SHA' to '$COMMIT_STATUS' with description '$COMMIT_STATUS_DESCRIPTION' and target url '$COMMIT_STATUS_TARGET_URL'") || echo "Failed to set commit status"
fi

rm -rf $TEMP_PATH

} 2>&1 | tee -a $LOG_DIR_BASE/docker.log
