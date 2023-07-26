from abc import ABC, abstractmethod
from typing import Callable, Union
from pandas import DataFrame
from numpy import ndarray
from gems.dask import LocalRunner

class DistributedFeatureExtractor(ABC):

    @abstractmethod
    def __init__(self):
        self.runner = LocalRunner()
        print(f"Dashboard adress: {self.runner.get_dashboard_adress()}")

    @abstractmethod
    def calculate_features(self, features: Union[ndarray, DataFrame], function: Callable):
        
        self.features_list = [x for x in features]
        self.futures_list = self.runner.map_function(function, self.features_list)
        self.calculated_features = self.runner.gather_futures(self.futures_list)
    