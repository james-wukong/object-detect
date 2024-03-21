from __future__ import annotations

import os
import time
import random
import string
from typing import Any, List

import cv2
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
import torch
from diffusers import StableDiffusionPipeline

from src.yolo import YoloModelInterface


class ModelA(YoloModelInterface):
    """
    implement the abstract methods
    """

    def __init__(self, model: str, device: Any, data_cfg=None):
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
                            device='cpu',
                            **val_args)
        # metrics.box.map  # map50-95
        # metrics.box.map50  # map50
        # metrics.box.map75  # map75
        # metrics.box.maps  # a list contains map50-95 of each category

        return metrics

    @staticmethod
    def predict(model: YOLO, img: Any, conf: float = 0.65,
                device: str = 'cpu') -> tuple[List, str]:
        # Run inference on 'bus.jpg' with arguments
        results = model(source=img,
                       save=False,
                       device=device,
                       conf=conf)
        if isinstance(img, list):
            dst_img = img[0].split('/')
        else:
            dst_img = img.split('/')
        dst_img[-2] = 'pred'
        dst_img = '/'.join(dst_img)
        for r in results:
            r.save(filename=dst_img)

        return results, dst_img

    def export(self, model: YOLO, fmt: str = 'onnx') -> None:
        model.export(format=fmt)

    def track_video(self, model_name: str = 'yolov8n.pt',
                    video_path: str = ''):
        # Load the YOLOv8 model
        model = YOLO(model_name)

        # Open the videos file
        cap = cv2.VideoCapture(video_path)

        # Loop through the videos frames
        while cap.isOpened():
            # Read a frame from the videos
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Display the annotated frame
                cv2.imshow("YOLOv8 Custom Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        # Release the videos capture object and close the display window
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

    @staticmethod
    def generate_image(prompt: str,
                       model_id: str = 'runwayml/stable-diffusion-v1-5',
                       device: str = 'cuda',
                       img_path: str = '') -> str:
        """
        generate image with stable diffusion model
        :param prompt: prompt text
        :param model_id: model id
        :param device: cpu or cuda
        :param img_path: path to save image
        :return: str, image path
        """
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe = pipe.to(device)

        image = pipe(prompt, num_inference_steps=15).images[0]
        rand_str = ''.join(random.choices(string.ascii_uppercase +
                                          string.digits, k=5))
        timestamp = str(round(time.time() * 1000))
        img_name = os.path.join(img_path, f'{rand_str}_{timestamp}.png')

        image.save(img_name)

        return img_name
