from kymatio.sklearn import Scattering2D
import numpy as np

class scattering_transform:
    """
    Feature extraction method using scattering wavelet transform.
    It computes locally translation-invariant image descriptors with a cascade of three operations:
    wavelet decomposition, complex modulus, and a local averaging.
    It carries the same locally translation-invariant characterization of SIFT but
    contains more high frequency information.
    Extracting image features using scattering transform includes the following steps:
    1. Compute scattering features for a M*N image with pre-specified number of layers (K), and
    scales (J) and orientations (L) for wavelet decomposition. The kth layer output has a size of L^k*(J choose k).
    2. For each resulting scattering transformed image with size (M/(2^J), N/(2^J)), calculate the mean and variance.
    For a 2-layer scattering transform operator, the resulting feature vector has a size of 2*(1+JL+L^2*J(J-1)/2).

    """
    def __init__(self,J,shape,L=8,max_order=2):
        """
        Constructor of scattering transform feature object.
        :param J(int): log2 of the scattering scale.
        :param shape (tuple of ints): shape of input image.
        :param L(int): number of orientations. Default to 8.
        :param max_order (int): number of layers. Default to 2.
        """
        self.J = J
        self.shape = shape
        self.L = L
        self.max_order = max_order

    def fit(self, X):
        """
        Fit input images into scattering transform feature object. The input needs to be a list of ndarrays.
        :param X: a list of single-channel ndarrays. The Scattering2D class requires ndarray with shape
        (n_samples,n_pixel_x,n_pixel_y) as input, but to keep the input to fit function consistent to other
        feature extraction modulus, a list is asked for input.
        :return: object, instance itself
        """
        self.sctr = Scattering2D(J = self.J, shape = self.shape, L = self.L, max_order=self.max_order)
        # convert list to ndarray
        self.X = np.stack(X, axis=0)
        return self

    def transform(self, X):
        """
        Extract scattering transform features from input images. The input needs to be a ndarray.
        Two steps were included:
        1. Calculate scattering coefficients.
        2. Calculate mean and variance of each transformed image, rendering scattering transform features.
        :param X: a single-channel ndarray with shape (n_samples,n_pixel_x,n_pixel_y)
        :return: Extracted scattering transform features. ndarray with shape (n_samples, n_sctr_features)
        """
        scattering_coefs = self.sctr.scattering(self.X)
        sctr_features = np.hstack((scattering_coefs.mean(axis=(2, 3)), scattering_coefs.var(axis=(2, 3))))
        return sctr_features

    def fit_transform(self, X):
        """
        Combine fit() and transform().
        """
        # convert list to ndarray
        self.X = np.stack(X, axis=0)
        self.sctr = Scattering2D(J=self.J, shape=self.shape, L=self.L, max_order=self.max_order)
        scattering_coefs = self.sctr.scattering(self.X)
        sctr_features = np.hstack((scattering_coefs.mean(axis=(2, 3)), scattering_coefs.var(axis=(2, 3))))
        return sctr_features