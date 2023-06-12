"""
===========================
Generic Metric class
===========================
"""
from enum import Enum

class Metric(Enum):
    """
    Enumeration class representing different metrics used in evaluation.

    Members:
    - Loss: Represents the loss metric.
    - Accuracy: Represents the accuracy metric.
    - F1Score: Represents the F1 score metric.
    - Recall: Represents the recall metric.
    - Precision: Represents the precision metric.
    """


    Loss = "Loss"
    Accuracy = "Accuracy"
    F1Score = "F1Score"
    Recall = "Recall"
    Precision = "Precision"