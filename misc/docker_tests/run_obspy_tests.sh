#!/bin/bash
OBSPY_PATH=$(dirname $(dirname $(pwd)))

# 1. Build all the base images if they do not yet exist.
DOCKERFILE_FOLDER=base_images
TEMP_PATH=temp
NEW_OBSPY_PATH=$TEMP_PATH/obspy
DATETIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Create temporary folder.
rm -rf $TEMP_PATH
mkdir -p $TEMP_PATH

# Copy ObsPy to the temp path. This path is the execution context of the Docker images.
mkdir -p $NEW_OBSPY_PATH
cp -r $OBSPY_PATH/obspy $NEW_OBSPY_PATH/obspy/
cp $OBSPY_PATH/setup.py $NEW_OBSPY_PATH/setup.py
cp $OBSPY_PATH/MANIFEST.in $NEW_OBSPY_PATH/MANIFEST.in
rm -f $NEW_OBSPY_PATH/obspy/lib/*.so

# Copy the install script.
cp scripts/install_and_run_tests_on_image.sh $TEMP_PATH/install_and_run_tests_on_image.sh

printf "STEP 1: CREATING BASE IMAGES\n"

for image_path in $DOCKERFILE_FOLDER/*; do
    image_name=$(basename $image_path)
    if [ $# != 0 ] && [[ "$*" != *$image_name* ]]; then
        continue
    fi
    has_image=$(docker images | grep obspy | grep $image_name)
    if [ "$has_image" ]; then
        echo "Image '$image_name already exists."
    else
        echo "Image '$image_name will be created."
        docker build -t obspy:$image_name $image_path
    fi
done

# Execute Python once and import ObsPy to trigger building the RELEASE-VERSION
# file.
python -c "import obspy"

# 2. Execute the ObsPy
printf "\nSTEP 2: EXECUTING THE TESTS\n"

# Loop over all ObsPy Docker images.
for image_name in $(docker images | grep obspy | awk '{print $2}'); do
    if [ $# != 0 ] && [[ "$*" != *$image_name* ]]; then
        continue
    fi
    printf 'Running Tests for image "'$image_name'"...\n'
    # Copy dockerfile and render template.
    sed 's/{{IMAGE_NAME}}/'$image_name'/g' scripts/Dockerfile_run_tests.tmpl > $TEMP_PATH/Dockerfile

    # Where to save the logs, and a random ID for the containers.
    LOG_DIR=logs/$DATETIME/$image_name
    mkdir -p $LOG_DIR
    ID=$RANDOM-$RANDOM-$RANDOM

    docker build -t temp:temp $TEMP_PATH

    docker run --name=$ID temp:temp

    docker cp $ID:/INSTALL_LOG.txt $LOG_DIR
    docker cp $ID:/TEST_LOG.txt $LOG_DIR

    docker rm $ID
    docker rmi temp:temp

done
rm -rf $TEMP_PATH
