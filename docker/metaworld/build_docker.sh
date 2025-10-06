#!/bin/bash
set -x

TAG=hypervla_metaworld
# PARENT=nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
# PARENT=nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
PARENT=nvidia/cudagl:11.4.2-devel-ubuntu20.04
USER_ID=`id -u`
GROUP_ID=`id -g`

docker build -f docker/metaworld/Dockerfile \
  --build-arg PARENT_IMAGE=${PARENT} \
  --build-arg USER_ID=${USER_ID} \
  --build-arg GROUP_ID=${GROUP_ID} \
  -t ${TAG} .