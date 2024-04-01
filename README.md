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

install requirements.txt

check the config.py and modify accordingly

create a new .ipynb file by copying sample-finetune.ipynb

change parameters on top of the file accordingly to achieve a better training result

To start the webservice:


```sh
pip install -r requirementst.txt
cd project-root-path
# or you can also use export FLASK_APP=app, then flask run
flask --app app run --port 5000 --debug
```

navigate to 127.0.0.1:5000, to visit the web service.

## Samples

original videos: /app/static/videos/{demo.mp4, demo1.mp4}

sample output videos under directory: /app/static/videos/{output, webcam}

sample image under direcctory: /app/static/images/{gen, pred}