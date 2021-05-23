import mahotas as mh
import numpy as np

class haralick:
    """
    Calculate Haralick texture features of input images based on grayscale level co-occurrence matrix (GLCM). GLCM
    contains information about how often a pair of pixel values co-occur adjacency, as defined by different angle
    and distance combination. Extracting Haralick texture features includes two steps:
    1. Construct GLCMs
    - Four different directions of adjacency are considered:
    * top-to-bottom
    * left-to-right
    * top left-to-bottom right
    * top right-to-left bottom
    - Number and value of different distances (scales) can be specified as hyperparameters.
    - One GLCM will be constructed for each angle-distance combination. An image I with p different gray-scale values
    will produce one p*p co-occurrence matrix C. C_{i,j} entry is the probability that the ith pixel value is found
    adjacent to the jth pixel value.
    2. Extract texture features from the constructed GLCMs
    - Haralick (1970) extracted 14 texture features based on GLCMs. Details and formulas can be found at
    http://haralick.org/journals/TexturalFeatures.pdf. The last feature is often omitted due to computational
    instability, thus resulting in 13 features.
    - Description of the 13 features (adapted from Cellprofiler documentation):
    * AngularSecondMoment: Measure of image homogeneity.
    A higher value of this feature indicates that the intensity varies less in an image. Value 1 indicates homogeneity.

    * Contrast: Measure of local variation in an image, with 0 for a uniform image and a high value for a high degree of
    local variation.

    * Correlation: Measure of linear dependency of intensity values in an image. For an image with large areas of
    similar intensities, correlation is much higher than for an image with noisier, uncorrelated intensities.

    * Variance: Measure of the variation of image intensity values.

    * Inverse Difference Moment: Another feature to represent image contrast.
    Has a low value for inhomogeneous images, and a relatively higher value for homogeneous images.

    * Sum Average: The average of the normalized grayscale image in the spatial domain.

    * Sum Variance: The variance of the normalized grayscale image in the spatial domain.

    * SumEntropy: A measure of randomness within an image.

    * Entropy: An indication of the complexity within an image. A complex image produces a high entropy value.

    * Difference Variance: The image variation in a normalized co-occurrence matrix.

    * Difference Entropy: Another indication of the amount of randomness in an image.

    * InfoMeas1: A measure of the total amount of information contained within a region of pixels derived from the
    recurring spatial relationship between specific intensity values.

    * InfoMeas2: An additional measure of the total amount of information contained within a region of pixels derived
    from the recurring spatial relationship between specific intensity values. It is a complementary value to InfoMeas1
    and is on a different scale.
    """
    def __init__(self,distance,ignore_zeros=True):
        """
        Constructor of Haralick feature extraction object.
        :param distance: the distance between a pair of pixels to be considered adjacent. It can be an iterable of any
        size, containing integers. For input image with size M*N, the integer needs to be an integer in the range of
        [1,min(M,N)-1].
        :param ignore_zeros: whether to ignore zero values (background) when constructing GLCM. Default to True.
        """
        self.distance = distance
        self.ignore_zeros = ignore_zeros

    def fit(self,X):
        """
        Fit input images into Haralick feature extraction object.
        :param X: Input image. It needs to be a list of single-channel ndarrays.
        :return: object itself.
        """
        return self

    def transform(self,X):
        """
        Transform input image into Haralick features for each angle-distance combination. With length of distance=s,
        4*13*s features will be extracted for each input image.
        :param X: Input image. It needs to be a list of single-channel ndarrays.
        :return: ndarray with size (n_samples,n_Haralick_features)
        """
        features = [None]*len(X)
        for (i,img) in enumerate(X):
            haralick = [mh.features.haralick(img, ignore_zeros=self.ignore_zeros, distance=s).flatten()
                        for s in self.distance]
            features[i] = np.hstack(haralick) # 1D array with size n_Haralick_features
        features = np.vstack(features)
        return features

