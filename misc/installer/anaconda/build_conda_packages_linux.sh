DOCKER=`which docker.io || which docker`
BUILD_DIR=conda_builds

mkdir $BUILD_DIR


################################################################################
### 32 bit

# Dockerfiles only work with files in their directory structure
cp obspy/meta.yaml LinuxCondaBuilder_32bit/meta.yaml

ID=$RANDOM-$RANDOM-$RANDOM
$DOCKER build -t obspy_conda_builder_32bit LinuxCondaBuilder_32bit

sleep 10
echo "Done building container!"

# Cleanup potentially existing runs.
$DOCKER kill conda_build_container || true
$DOCKER rm conda_build_container || true
# Ugly way to ensure a container is running to be able to copy something.
$DOCKER run --name=conda_build_container -d obspy_conda_builder_32bit sleep 60
$DOCKER cp conda_build_container:/miniconda/conda-bld/linux-32 $BUILD_DIR
$DOCKER kill conda_build_container
$DOCKER rm conda_build_container
$DOCKER rmi obspy_conda_builder_32bit

rm -f LinuxCondaBuilder_32bit/meta.yaml


# ################################################################################
# ### 64 bit
#
# # Dockerfiles only work with files in their directory structure
# cp obspy/meta.yaml LinuxCondaBuilder_64bit/meta.yaml
#
# ID=$RANDOM-$RANDOM-$RANDOM
# $DOCKER build -t obspy_conda_builder_64bit LinuxCondaBuilder_64bit
#
# sleep 10
# echo "Done building container!"
#
# # Cleanup potentially existing runs.
# $DOCKER kill conda_build_container || true
# $DOCKER rm conda_build_container || true
# # Ugly way to ensure a container is running to be able to copy something.
# $DOCKER run --name=conda_build_container -d obspy_conda_builder_64bit sleep 60
# $DOCKER cp conda_build_container:/miniconda/conda-bld/linux-64 $BUILD_DIR
# $DOCKER kill conda_build_container
# $DOCKER rm conda_build_container
# $DOCKER rmi obspy_conda_builder_64bit
#
# rm -f LinuxCondaBuilder_64bit/meta.yaml
