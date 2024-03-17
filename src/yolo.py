from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, List

import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark


class YoloModelInterface(ABC):

    @abstractmethod
    def train(self, project_dir: str, conf: dict = None) -> tuple[YOLO, dict | None]:
        pass

    @abstractmethod
    def val(self, model: YOLO) -> dict:
        pass

    @abstractmethod
    def predict(self, model: YOLO, img: str, conf: float = 0.5) -> List:
        pass

    @abstractmethod
    def export(self, model: YOLO, format: str = '-') -> Any:
        pass

    @abstractmethod
    def track(self):
        pass

    @abstractmethod
    def benchmark(self, **kwargs) -> pd.DataFrame:
        pass
#
# data_cfg = 'yolo-data-conf.yaml'
# # m1/m2 training, for gpu: [0]
# train_args = {
#     # Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto
#     "optimizer": 'Adam',
#     "momentum": 0.938,
#     "freeze": 10, # the first N layers
#     "batch": -1,
#     "patience": 5,
#     "epochs": 5,
#     # cmd tensorboard --logdir runs/train/ (default)
#     "tensorboard": True,
#     # "log_dir": 'runs/train/',
#     "project_dir": os.path.join(os.getcwd(), 'data_models', 'my_trained_model'),
#     "model_name": 'yolov8s_hg',
#     "device": 'mps'
# }
#
# # train model, baseline: v8n | v8s | v8m | v8m | v8x
# model = YOLO('pre_trained_models/yolov8s.pt')
# model.train(data=data_cfg,
#             save=True,
#             amp=True, # Enables Automatic Mixed Precision (AMP) training
#             exist_ok=True,
#             pretrained=True,
#             verbose=2,
#             **train_args)
#
# # Customize validation settings
# validation_results = model.val(data=data_cfg,
#                                imgsz=640,
#                                batch=16,
#                                conf=0.25,
#                                iou=0.6,
#                                device='0')
# print(validation_results)
#
#
# # Resume training
# model = YOLO('path/to/last.pt')
# results = model.train(resume=True)
#
#
# # Benchmark on GPU, such as "cpu", "cuda:0"
# benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)