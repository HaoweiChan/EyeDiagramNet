"""
Machine Learning Callbacks Module

This module contains PyTorch Lightning callbacks for training enhancements:
- DynamicThresholdOptimizer: Automatically optimizes classification thresholds during training
- GradientTracker: Tracks gradient flow and identifies unused parameters for debugging
- SParameterVisualizer: Visualizes S-parameter reconstructions during training
- EWPredictionWriter: Writes eye width predictions and metadata to files during testing/prediction
"""

from .dynamic_threshold import DynamicThresholdOptimizer
from .prediction_writer import EWPredictionWriter
from .gradient_tracker import GradientTracker  
from .sparam_visualizer import SParameterVisualizer

__all__ = [
    'DynamicThresholdOptimizer',
    'EWPredictionWriter',
    'GradientTracker', 
    'SParameterVisualizer'
]
