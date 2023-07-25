"""
This module defines the MLData data class which is used to store training, validation, and testing data for machine learning models.
"""

from dataclasses import dataclass
from typing import Union, Optional
import numpy
import dask.array
import dask.dataframe
import pandas

# Define a type hint for valid data types
valid_type = Union[
    numpy.ndarray,
    pandas.DataFrame,
    pandas.Series,
    dask.array.Array,
    dask.dataframe.DataFrame,
    dask.dataframe.Series
]


@dataclass
class MLData:
    """
    A class used to store the data for machine learning models.

    Attributes
    ----------
    X_train : valid_type
        feature vectors for the training dataset
    X_val : valid_type, optional
        feature vectors for the validation dataset, by default None
    X_test : valid_type, optional
        feature vectors for the testing dataset, by default None
    y_train : valid_type
        target values for the training dataset
    y_val : valid_type, optional
        target values for the validation dataset, by default None
    y_test : valid_type, optional
        target values for the testing dataset, by default None
    """

    X_train: valid_type
    X_val: Optional[valid_type]
    X_test: Optional[valid_type]

    y_train: valid_type
    y_val: Optional[valid_type]
    y_test: Optional[valid_type]
