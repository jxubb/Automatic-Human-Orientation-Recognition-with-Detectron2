# Automatic-Human-Orientation-Recognition-with-Detectron2
This application is based on [Detectron2](https://github.com/facebookresearch/detectron2) which is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms. The main purpose of this application is to detect human orientation and determine the correct orientation (from 0, 90, 180, and 270 degrees) of a human image/photo.

# Requirements
Ubuntu 18.04

CUDA 10.1

Docker CE latest stable release

# Deploy this model with Docker

## Build Docker Image

docker image build -t <Image Name> .

## Build Docker Container

docker container run --gpus all -d -p <external port number>:<internal port number> --name <Container Name> <Image Name>

## Basic Docker Commamds

### Check Container Status
docker logs <container name>

### Stop Docker Container 
docker stop <container name>


