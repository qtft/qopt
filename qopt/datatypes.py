"""
A collection of commonly defined data types.
"""
from typing import Iterable, Union

import numpy as np

array_like = Union[np.ndarray, Iterable, int, float]
