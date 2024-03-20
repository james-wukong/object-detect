# object-detect

This project aims to provide a hands-on practice with object detection and text-to-image model training and evaluation.

It involves two main pre-trained models:

1. Yolov8, which is used as object detection model in this case
2. Stable Diffusion, which is used as text-to-image generative model

There are three main paths for our services:

1. text to image : /text-to-image, which provide a service that user input a text, and our model will generate an image based on the description, and forward the generated image to object detection model to detect objects
2. video detection: /vidio-detection, which provide a demo that our object detection model would track objects in video.
3. live detection: /live-detection, which rovide a webcam object detection.
 
## Developers

after git clone to your local environment, you need to edit `yolo-data-conf.yaml` file, and chane the path to your local absolute path to your data folder

## How to use (NOT READY YET)

1. install docker in your system
2. enter root directory of this project:
3. run following command in your terminal :

```sh
# linux -> linux/amd64, mac m1/m2 -> linux/arm64
docker buildx build --platform linux/arm64 -t object_detect .

docker run -d --name object_detect -p 8080:8080 object_detect
# docker run -d --name object_detect -p 8080:8080 -v ~/Documents/Docker-Volumns/object-detect:/app  object_detect

docker exec -it object_detect /bin/bash

uvincorn main:app --reload
```
