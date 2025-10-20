"""训练模块"""
from .train_pedestrian import main as train_pedestrian
from .train_qc import train_qc_model

__all__ = ["train_pedestrian", "train_qc_model"]
