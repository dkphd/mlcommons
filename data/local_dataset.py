from abc import ABC, abstractmethod
from data.mldata import MLData
from data.local_artifact import LocalArtifact
from typing import Union
from pathlib import Path


class LocalDataset(LocalArtifact, MLData):
    """
    An abstract base class used to represent a local dataset.

    This classd derives from both LocalArtifact and MLData to include the data fields and load/savw methods
    """
