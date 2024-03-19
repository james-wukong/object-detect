import os

from flask import render_template, request, flash
from ultralytics import YOLO

from src.model_a import ModelA

import app
from app.main import bp

MODEL_BASE_PATH = os.path.join(os.getcwd(),
                               'data_models',
                               'my_trained_models')
IMAGE_BASE_PATH = os.path.join(os.getcwd(),
                               'app',
                               'images')


@bp.route('/', methods=['GET', 'POST'])
def index():
    models = {'yolov8n': os.path.join(MODEL_BASE_PATH, '1'),
              'yolov8s': os.path.join(MODEL_BASE_PATH, '1'),
              'yolov8m': os.path.join(MODEL_BASE_PATH, '1'),
              'yolov8x': os.path.join(MODEL_BASE_PATH, '1'),
              'yolov8xl': os.path.join(MODEL_BASE_PATH, '1')
              }
    if request.method == 'POST':
        model = request.form.get('model')
        content = request.form.get('content')

        if not model:
            flash('Model is required!')
        elif not content:
            flash('Content is required!')
        else:
            gen_image = ModelA.generate_image(content,
                                              img_path='')
            yolo = YOLO(os.path.join(model), 'detect')
            model = ModelA.predict(yolo, img='', conf=0.66)
            return render_template("index.html",
                                   gen_image=gen_image,
                                   pred_image='',
                                   test=model,
                                   models=models)

    return render_template('index.html', models=models, test=os.getcwd())


# @bp.route('/text', methods=['POST'])
# def text():
#     return render_template('index.html')


@bp.route('/video', methods=['GET'])
def video():
    return render_template('video.html')


@bp.route('/webcam', methods=['GET'])
def webcam():
    return render_template('webcam.html')
