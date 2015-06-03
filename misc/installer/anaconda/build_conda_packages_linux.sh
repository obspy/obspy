DOCKER=`which docker.io || which docker`
BUILD_DIR=conda_builds

rm -rf $BUILD_DIR
mkdir $BUILD_DIR


################################################################################
### 32 bit

# Dockerfiles only work with files in their directory structure
cp obspy/meta.yaml LinuxCondaBuilder_32bit/meta.yaml

ID=$RANDOM-$RANDOM-$RANDOM
$DOCKER build -t temp:temp LinuxCondaBuilder_32bit
# Ugly way to ensure a container is running to be able to copy something.
$DOCKER run --name=$ID -d temp:temp python -c "import time; time.sleep(600)"
$DOCKER cp $ID:/miniconda/conda-bld/linux-32 $BUILD_DIR
$DOCKER stop -t 0 $ID
$DOCKER rm $ID
$DOCKER rmi temp:temp

rm -f LinuxCondaBuilder_32bit/meta.yaml


################################################################################
### 64 bit

# Dockerfiles only work with files in their directory structure
cp obspy/meta.yaml LinuxCondaBuilder_64bit/meta.yaml

ID=$RANDOM-$RANDOM-$RANDOM
$DOCKER build -t temp:temp LinuxCondaBuilder_64bit
# Ugly way to ensure a container is running to be able to copy something.
$DOCKER run --name=$ID -d temp:temp python -c "import time; time.sleep(600)"
$DOCKER cp $ID:/miniconda/conda-bld/linux-64 $BUILD_DIR
$DOCKER stop -t 0 $ID
$DOCKER rm $ID
$DOCKER rmi temp:temp

rm -f LinuxCondaBuilder_64bit/meta.yaml
