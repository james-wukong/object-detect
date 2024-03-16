from __future__ import annotations

from typing import Any, List

from ultralytics import YOLO

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

    def train(self, project_dir: str, conf: dict = None) -> tuple:
        """
        train the model
        :param project_dir:
        :param conf: dictionary, may include: optimizer, momentum, freeze, batch, patience, epochs, tensorboard
        :return:
        """
        # data_cfg = 'yolo-data-conf.yaml'
        # m1/m2 training, for gpu: [0]
        train_args = {
            # Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto
            "optimizer": conf.get('optimizer', 'Adam'),
            "momentum": conf.get('momentum', 0.938),
            "freeze": conf.get('freeze', 10),  # the first N layers
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
                             pretrained=True, # Determines whether to start training from a pretrained model.
                             verbose=True,
                             **train_args)

        return model, result

    def val(self, model: YOLO, batch_size=-1) -> dict:
        """
        evaluate model
        :param model: YOLO, trained model
        :return:
        """
        # Validate the model
        metrics = model.val(data=self.data_cfg,
                            batch=batch_size,
                            conf=0.25,
                            iou=0.6,
                            device=self.device)
        # metrics.box.map  # map50-95
        # metrics.box.map50  # map50
        # metrics.box.map75  # map75
        # metrics.box.maps  # a list contains map50-95 of each category

        return metrics

    def predict(self, model: YOLO, conf: float = 0.5) -> List:
        # Run inference on 'bus.jpg' with arguments
        result = model.predict('bus.jpg', save=True, conf=conf)

        return result

    def export(self):
        pass


    def track(self):
        pass

    def benchmark(self):
        pass