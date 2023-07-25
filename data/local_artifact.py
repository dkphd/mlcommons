from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path


class LocalArtifact(ABC):
    """
    An abstract base class used to represent a local artifact.
    """

    def __init__(self, local_path: Union[str, Path]):
        self.local_path = Path(local_path)
        self.local_path_used = False  # if either save or load was used, this variable should be set to True

    @staticmethod
    @abstractmethod
    def load(path: Union[str, Path]):
        """
        Method for loading the artifact from a local storage

        This method should be overridden by any non-abstract child class of LocalArtifact.

        Parameters
        ----------
        path : Union[str, Path]
            The path of the file to load data from. This can be a string or a Path object.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Abstract method for saving the artifact to a local storage.

        This method should be overridden by any non-abstract child class of LocalArtifact.

        """
        pass

    def _mark_local_path_used(self):
        self.local_path_used = True
