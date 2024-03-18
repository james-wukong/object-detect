from __future__ import annotations

from typing import Any, List

import cv2
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

from src.yolo import YoloModelInterface


class ModelA(YoloModelInterface):
    """
    implement the abstract methods
    """

    def __init__(self, model: str, data_cfg: str, device: Any):
        """
        initialize the model with pretrained model name and data configure file
        :param model:
        :param data_cfg:
        """
        self.model = model
        self.data_cfg = data_cfg
        self.device = device

    def train(self, project_dir: str, conf: dict = None) -> tuple[YOLO, dict | None]:
        """
        train the model
        :param project_dir:
        :param conf: dictionary, may include: optimizer, momentum,
                    freeze, batch, patience, epochs, tensorboard
        :return:
        """
        # data_cfg = 'yolo-data-conf.yaml'
        # m1/m2 training, for gpu: [0]
        train_args = {
            # Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto
            "optimizer": conf.get('optimizer', 'Adam'),
            "lr0": conf.get('learning_rate', 1e-4),
            "momentum": conf.get('momentum', 0.938),
            "freeze": conf.get('freeze', 16),  # the first N layers
            "batch": conf.get('batch', -1),
            "patience": conf.get('patience', 5),
            "epochs": conf.get('epochs', 3),
            # cmd tensorboard --logdir runs/train/ (default)
            # "tensorboard": conf.get('tensorboard', True),
            # "log_dir": 'runs/train/',
            "project": project_dir,
            "name": conf.get('model_name', 'yolov8n_j_hg'),
            "device": self.device if self.device else 'cpu'
        }

        # train model, baseline: v8n | v8s | v8m | v8m | v8x
        model = YOLO(f'data_models/pre_trained_models/{self.model}', task='detect')
        result = model.train(data=self.data_cfg,
                             save=True,
                             amp=True,  # Enables Automatic Mixed Precision (AMP) training
                             exist_ok=True,
                             pretrained=True,  # Determines whether to start training from a pretrained model.
                             verbose=True,
                             **train_args)

        return model, result

    def val(self, model: YOLO, **kwargs) -> dict | None:
        """
        evaluate the model
        :param model: YOLO model
        :param kwargs: split | iou | conf | batch
        :return:
        """
        excluded_dict = {key: value for key, value in kwargs.items() if key not in ('data', 'device')}
        val_args = {
            # Determines the dataset split to use for validation (val, test, or train)
            "split": excluded_dict.get('split', 'val'),
            "iou": excluded_dict.get('iou', 0.6),
            "conf": excluded_dict.get('conf', 0.001),  # the confidence
            "batch": excluded_dict.get('batch', 16),
        }

        # Validate the model
        metrics = model.val(data=self.data_cfg,
                            device=self.device,
                            **val_args)
        # metrics.box.map  # map50-95
        # metrics.box.map50  # map50
        # metrics.box.map75  # map75
        # metrics.box.maps  # a list contains map50-95 of each category

        return metrics

    def predict(self, model: YOLO, img: str, conf: float = 0.65) -> List:
        # Run inference on 'bus.jpg' with arguments
        result = model.predict(source=img,
                               save=True,
                               conf=conf)

        return result

    def export(self, model: YOLO, fmt: str = 'onnx') -> None:
        model.export(format=fmt)

    def track_video(self):
        # Load the YOLOv8 model
        model = YOLO('yolov8n.pt')

        # Open the video file
        video_path = "path/to/video.mp4"
        cap = cv2.VideoCapture(video_path)

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()

    def track_webcam(self):
        pass

    def benchmark(self, **kwargs) -> None:
        excluded_dict = {key: value for key, value in kwargs.items()
                         if key not in ('data', 'device')}
        bm_args = {
            "model": excluded_dict.get('model', self.model),
            "verbose": excluded_dict.get('verbose', True),
        }
        benchmark(data=self.data_cfg,
                  device=self.device,
                  **bm_args)