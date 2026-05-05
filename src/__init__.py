"""
JuaSamaki: Fish Detection and Weight Estimation
A computer vision system for aquaculture farmers to automatically detect, measure, and estimate fish weight.
"""

__version__ = "0.1.0"
__author__ = "KaraniWachira"

from .fish_detector import FishDetector
from .weight_predictor import WeightPredictor
from .calibration import CalibrationManager

__all__ = ["FishDetector", "WeightPredictor", "CalibrationManager"]
