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

1. after git clone to your local environment, you need to edit `yolo-data-conf.yaml` file, and chane the path to your local absolute path to your data folder
1. test functionalities, mainly web services

## How to use

1. install requirements.txt

```sh
pip install -r requirementst.txt
cd project-root-path
# or you can also use export FLASK_APP=app, then flask run
flask --app app run --debug
```

navigate to 127.0.0.1:5000, to visit the web service.
