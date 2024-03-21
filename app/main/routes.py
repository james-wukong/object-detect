import os

from flask import render_template, request, flash, stream_with_context
from flask import Response
from config import Config
import cv2
from ultralytics import YOLO

import app
from src.model_a import ModelA

from app.main import bp


@bp.route('/', methods=['GET', 'POST'])
def index():
    """
    1. take text input from form
    2. generate image from stable diffusion
    3. send yolo model to do object detection
    4. send both generated and detected images to index.html
    :return:
    """
    if request.method == 'POST':
        model = request.form.get('model')
        content = request.form.get('content')

        if not model:
            flash('Model is required!')
        elif not content:
            flash('Content is required!')
        else:
            # model id: runwayml/stable-diffusion-v1-5
            gen_image = ModelA.generate_image(content,
                                              device=Config.SD_DEVICE,
                                              model_id='runwayml/stable-diffusion-v1-5',
                                              img_path=f'{Config.IMAGE_BASE_PATH}/gen/')
            # gen_image = 'app/static/images/gen/1.jpeg'
            yolo = YOLO(os.path.join(model), 'detect')

            # Specifies the device for inference (e.g., cpu, cuda:0 or 0).
            # Allows users to select between CPU, a specific GPU, or other compute devices
            # for model execution.
            results, dst_img = ModelA.predict(yolo,
                                              img=gen_image,
                                              conf=0.45,
                                              device=Config.DEVICE)

            return render_template("index.html",
                                   gen_image=gen_image.split('/')[-1],
                                   pred_image=dst_img.split('/')[-1],
                                   models=Config.MODELS,
                                   results=results)

    return render_template('index.html', models=Config.MODELS)


@bp.route('/video', methods=['GET'])
def video():
    if request.method == 'POST':
        model = request.form.get('model')
    return render_template('video.html', models=Config.MODELS)


@bp.route('/video_feed', methods=['GET'])
@bp.route('/video_feed/<model_id>', methods=['GET'])
def video_feed(model_id: str = None):
    video_path = Config.VIDEO_DEMO
    cap = cv2.VideoCapture(video_path)
    model_id = '/'.join(model_id.split('-')) if model_id else None

    # Function to generate videos frames
    def generate_frames(mid):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Run YOLOv8 inference on the frame
            mid = Config.MODELS['yolov8n'] if mid is None else mid
            # mid = 'data_models/my_trained_models/j/yolov8m_hat_glass/weights/best.pt'
            model = YOLO(mid, 'detect')
            # Specifies the device for inference (e.g., cpu, cuda:0 or 0).
            # Allows users to select between CPU, a specific GPU, or other compute devices
            # for model execution.
            results = model(frame,
                            conf=0.35,
                            device=Config.DEVICE)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Convert frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Release capture device
        cap.release()

    # Return Response object with the generated frames
    return Response(stream_with_context(generate_frames(model_id)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route('/webcam', methods=['GET'])
def webcam():
    # if request.method == 'POST':
    #     model = request.form.get('model')
    return render_template('webcam.html', models=Config.MODELS, model_id=None)


@bp.route('/webcam_feed', methods=['GET'])
@bp.route('/webcam_feed/<model_id>', methods=['GET'])
def webcam_feed(model_id: str = None):
    model_id = '/'.join(model_id.split('-')) if model_id else None

    # Function to generate videos frames
    def generate_frames(mid):
        camera = cv2.VideoCapture(0)  # Use 0 for default webcam
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # Run YOLOv8 inference on the frame
                mid = Config.MODELS['yolov8n'] if mid is None else mid
                model = YOLO(mid, 'detect')

                # Specifies the device for inference (e.g., cpu, cuda:0 or 0).
                # Allows users to select between CPU, a specific GPU, or other compute devices
                # for model execution.
                results = model(frame,
                                conf=0.45,
                                device=Config.DEVICE)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # Release capture device
        camera.release()

    # Return Response object with the generated frames
    return Response(generate_frames(model_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
