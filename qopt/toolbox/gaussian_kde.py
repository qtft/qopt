"""
An implementation of Gaussian Kernel Density Estimator.

It is a building block for other algorithms
such as Horn–Gottlieb's quantum-inspired clustering algorithm.
"""
from typing import Optional, TypeVar

import numpy as np

from qopt.datatypes import array_like

T = TypeVar('T', bound='GaussianKDE')


##############
# Main class #
##############

class GaussianKDE:
    """
    Basic Kernel Density Estimator (KDE) that uses
    the Gaussian distribution function as the kernel function.

    Attribute `data` is the only required attribute for this class.
    It represents a dataset as a numpy array of shape (N, D)
    where N and D are the number of data points and dimensions, respectively.
    """
    data: np.ndarray
    sigma: Optional[float]

    __slots__ = ['data', 'sigma']

    def __init__(self, data: array_like, *, sigma: float = 1, _copy: bool = True):
        # Convert dataset into numpy array, freeze the content,
        # and verify that it is two-dimensional
        if _copy:
            self.data = np.array(data)
            self.data.setflags(write=False)
            assert len(self.data.shape) == 2
        else:
            self.data = data
        # Keep the Gaussian sigma value
        self.sigma = sigma

    @property
    def n_points(self) -> int:
        """
        Number of data points in the dataset.
        """
        return self.data.shape[0]

    @property
    def n_dims(self) -> int:
        """
        Number of dimensions in the dataset.
        """
        return self.data.shape[1]

    def replace(self: T, *, sigma: float) -> T:
        """
        Creates a new KDE estimator using a new Gaussian sigma value
        but retain the original dataset.
        """
        return GaussianKDE(self.data, sigma=sigma, _copy=False)

    def wave(self, x: np.ndarray):
        """
        Wave function (Psi) based on the given point x.
        """
        x = self._sanitize_x(x)
        return np.sum(np.exp(-self.dist(x) / (2 * self.sigma ** 2)))

    def potential(self, x: np.ndarray) -> float:
        """
        Potential energy (V) of a given point x.
        """
        x = self._sanitize_x(x)
        dist = self.dist(x)
        return np.sum(dist * np.exp(-dist / (2 * self.sigma ** 2))) / self.wave(x)

    def dist(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Euclidean distance between the given point x
        and each point in the dataset.
        """
        x = self._sanitize_x(x)
        return np.square(np.linalg.norm(x - self.data, axis=1))

    def _sanitize_x(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x)
        assert len(x.shape) == 1
        assert x.shape[0] == self.data.shape[1]
        return x
