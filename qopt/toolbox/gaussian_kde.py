"""
An implementation of Gaussian Kernel Density Estimator.

It is a building block for other algorithms
such as Horn–Gottlieb's quantum-inspired clustering algorithm.
"""
import numpy as np

from qopt.datatypes import array_like


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
    __slots__ = ['data']

    class ParametrizedGaussianKDE:
        """
        A parametrized version of GaussianKDE class.
        Each instance of this class will reference to
        an instance of the original GaussianKDE class
        but some parameters will be pre-defined, such as Gaussian sigma.
        """
        parent: 'GaussianKDE'
        sigma: float
        __slots__ = ['parent', 'sigma']

        def __init__(self, parent: 'GaussianKDE', *, sigma: float = 1):
            self.parent = parent
            self.sigma = sigma

        def wave(self, x: np.ndarray):
            """
            Wave function (Psi) based on the given point x.
            """
            return self.parent.wave(x, sigma=self.sigma)

        def potential(self, x: np.ndarray, sigma: float = 1.0) -> float:
            """
            Potential energy (V) of a given point x.
            """
            return self.parent.potential(x, sigma=self.sigma)

    def __init__(self, data: array_like):
        self.data = np.array(data)
        self.data.setflags(write=False)

    def parametrize(self, **options):
        """
        Extends the instance of GaussianKDE class with
        the specified arguments (such as Gaussian sigma value).

        Keyword Args:
            sigma: Floating point value representing Gaussian radius of KDE
        """
        return GaussianKDE.ParametrizedGaussianKDE(self, **options)

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

    def wave(self, x: np.ndarray, *, sigma: float = 1.0):
        """
        Wave function (Psi) based on the given point x
        assuming the given Gaussian sigma value.
        """
        x = self._sanitize_x(x)
        return np.sum(np.exp(-self.dist(x) / (2 * sigma ** 2)))

    def potential(self, x: np.ndarray, sigma: float = 1.0) -> float:
        """
        Potential energy (V) of a given point x
        assuming the given Gaussian sigma value.
        """
        x = self._sanitize_x(x)
        dist = self.dist(x)
        return np.sum(dist * np.exp(-dist / (2 * sigma ** 2))) / self.wave(x)

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
