from __future__ import annotations
import os
import sys
from typing import List

from ultralytics import YOLO
from flask import Flask, flash, request, redirect, Response, render_template, send_from_directory
from werkzeug.utils import secure_filename
from enum import Enum, auto

sys.path.append(os.getcwd())
from src.model_a import ModelA


import cv2
IMAGES = {'png', 'jpg', 'jpeg'}
VIDEOES = {"mp4", "ogv", "wmv", "mov"}

UPLOAD_FOLDER = "./upload/"
ALLOWED_EXTENSIONS = IMAGES.union(VIDEOES)


class Formats(Enum):
    VIDEO = auto()
    IMAGE = auto()
    NONE = auto()
# formats = Enum("VIDEO", "IMAGE", "NONE")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

camera = cv2.VideoCapture(0)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_frame():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.png', frame)
            frame = buffer.tobytes()
            yield(b"--frame\r\n"b"Content-Type: image/png\r\n\r\n"+ frame + b"\r\n")


@app.route("/", methods=["GET", "POST"])
def site():
    filename = ""
    format = Formats.NONE
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if filename.rsplit('.', 1)[1].lower() in IMAGES:
                format = Formats.IMAGE
            if filename.rsplit('.', 1)[1].lower() in VIDEOES:
                format = Formats.VIDEO
    return render_template(
        "tmp_img.html",
        dir=app.config["UPLOAD_FOLDER"],
        filename=filename,
        format=format.name,
    )


@app.route("/upload/<name>")
def load(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route("/cam")
def stream():
    return Response(get_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

app.add_url_rule(
    "/uploads/<name>", endpoint="load", build_only=True
)

