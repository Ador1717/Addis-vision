"""
Singleton model manager — loads each YOLO model once and reuses it.
Thread-safe via a simple dict + lock pattern.
"""
import threading
from pathlib import Path
from ultralytics import YOLO
from app.config import MODELS


class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models: dict[str, YOLO] = {}
                    cls._instance._model_lock = threading.Lock()
        return cls._instance

    def get(self, model_key: str) -> YOLO:
        with self._model_lock:
            if model_key not in self._models:
                path = MODELS.get(model_key)
                if not path or not Path(path).exists():
                    raise FileNotFoundError(
                        f"Model '{model_key}' not found at: {path}"
                    )
                self._models[model_key] = YOLO(path)
            return self._models[model_key]

    def available_models(self) -> list[dict]:
        result = []
        for key, path in MODELS.items():
            result.append(
                {"id": key, "path": path, "available": Path(path).exists()}
            )
        return result


model_manager = ModelManager()
