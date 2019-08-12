from typing import Optional

import numpy as np


class HornGottliebQClustering:
    """
    Quantum-inspired clustering algorithm by David Horn and Assaf Gottlieb.

    Default parameter sigma can be provided at the construction of the
    instance, and can be overridden when calling particular methods.
    """
    # TODO: memory optimizations

    data: np.ndarray
    sigma: Optional[float]

    __slots__ = ['data', 'sigma']

    def __init__(self, data: np.ndarray, sigma: Optional[float] = None):
        assert len(self.data.shape) == 2
        self.data = data
        self.sigma = sigma

    def wave(self, x: np.ndarray, *, sigma: Optional[float] = None):
        """
        Parzen-window estimator (wave function, Psi)
        of a given point with Gaussian radius sigma.
        """
        x = self._sanitize_x(x)
        sigma = self._sanitize_sigma(sigma)
        return self._wave(x, sigma=sigma)

    def potential(self, x: np.ndarray, *, sigma: Optional[float] = None):
        """
        Potential energy (V) of a given point with Gaussian radius sigma.
        """
        x = self._sanitize_x(x)
        sigma = self._sanitize_sigma(sigma)
        return self._potential(x, sigma=sigma)

    def _wave(self, x: np.ndarray, *, sigma: float):
        return np.sum(np.exp(-self._dist_square(x) / (2 * sigma ** 2)))

    def _potential(self, x: np.ndarray, *, sigma: float):
        dist_square = self._dist_square(x)
        return (
            np.sum(dist_square * np.exp(-dist_square / (2 * sigma ** 2)))
            / self._wave(x, sigma=sigma)
        )

    def _dist_square(self, x: np.ndarray):
        return np.sum(np.square(x - self.data), axis=1)

    def _sanitize_x(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 1
        assert x.shape[0] == self.data.shape[1]
        return x

    def _sanitize_sigma(self, sigma: Optional[float]) -> float:
        sigma = sigma or self.sigma
        if sigma is None:
            raise ValueError("parameter sigma not provided")
        return sigma
