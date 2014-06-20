OBSPY_PATH=$(dirname $(dirname $(pwd)))
IMAGE_DIR=obspy_run_centos_6
NEW_OBSPY_PATH=$IMAGE_DIR/obspy
ID=$RANDOM-$RANDOM-$RANDOM
LOG_DIR=logs/$IMAGE_DIR/$(date -u +"%Y-%m-%dT%H:%M:%SZ")

mkdir -p $LOG_DIR

# Copy ObsPy
mkdir -p $NEW_OBSPY_PATH
cp -r $OBSPY_PATH/obspy $NEW_OBSPY_PATH/obspy/
cp $OBSPY_PATH/setup.py $NEW_OBSPY_PATH/setup.py
cp $OBSPY_PATH/MANIFEST.in $NEW_OBSPY_PATH/MANIFEST.in

# Copy install scripy
cp ./_install_and_run_tests_on_image.sh $IMAGE_DIR/_install_and_run_tests_on_image.sh

docker build -t temp:temp $IMAGE_DIR

rm -rf $NEW_OBSPY_PATH
rm -rf $IMAGE_DIR/_install_and_run_tests_on_image.sh

docker run --name=$ID temp:temp

docker cp $ID:/INSTALL_LOG.txt $LOG_DIR
docker cp $ID:/TEST_LOG.txt $LOG_DIR

docker rm $ID
docker rmi temp:temp
