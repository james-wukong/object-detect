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

    @staticmethod
    @abstractmethod
    def predict(self, model: YOLO, img: str, conf: float = 0.5) -> tuple[List, str]:
        pass

    @abstractmethod
    def export(self, model: YOLO, format: str = '-') -> None:
        pass

    @abstractmethod
    def track_video(self):
        pass

    @abstractmethod
    def track_webcam(self):
        pass

    @abstractmethod
    def benchmark(self, **kwargs) -> None:
        pass
