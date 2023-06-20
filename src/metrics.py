"""
===========================
Generic Metric class
===========================

This module defines a generic Metric enumeration class.
This class enumerates different metrics that can be used in evaluating the performance of a machine learning model.
These metrics include Loss, Accuracy, F1Score, Recall, Precision, and AUC (Area under the curve of the ROC).
It can be used in the context of machine learning model evaluation where one needs to switch between different metrics
based on the problem at hand.
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
    - AUC: Area under the curve of the ROC.
    """


    Loss = "Loss"
    Accuracy = "Accuracy"
    F1Score = "F1Score"
    Recall = "Recall"
    Precision = "Precision"
    AUC = "AUC"