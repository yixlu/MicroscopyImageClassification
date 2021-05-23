import numpy as np
import cv2

class IntensityMeasure:
    """
    Extract intensity features from input images. An optional mask can be applied to define ROI, where intensity
    measurements are made exclusively.
    """
    def __init__(self,mask=None):
        """
        Class constructor method for intensity measurements.
        :param mask: an optional mask. Must be a list of single-channel array with binary values. If not specified,
        no mask will be applied.
        """
        if mask:
            for (i,m) in enumerate(mask):
                assert len(np.unique(m)) == 2, "Mask is not binary."
        self.mask=mask

    def fit(self,X):
        """
        If mask is specified, apply mask to the input images X. Number of masks must equal number of input images.
        :param X: a list of single-channel arrays.
        :return: object, instance itself
        """
        if self.mask:
            assert len(X) == len(self.mask), "Number of source images and number of masks do not equal."
            self.roi = [None]*len(X)
            for (i,(img,mask)) in enumerate(zip(X,self.mask)):
                self.roi[i]=img[mask != 0]
        else:
            self.roi = X
        return self

    def binary_centroid(self,img):
        """
        Helper function: calculate binary_centroid of ROI defined objects.
        :param img: a single-channel arrays.
        :return: a tuple (X,Y): coordinates of binary centroid.
        """
        M_bi = cv2.moments(img, binaryImage=True)  # binary image moments
        X_bi = int(M_bi["m10"] / M_bi["m00"])
        Y_bi = int(M_bi["m01"] / M_bi["m00"])
        return (X_bi,Y_bi)

    def gray_centroid(self,img):
        """
        Helper function: calculate grayscale centroid of ROI defined objects.
        :param img: a single-channel arrays.
        :return: a tuple (X,Y): coordinates of grayscale centroid.
        """
        M_gy = cv2.moments(img, binaryImage=False)  # grayscale image moments
        X_gy = int(M_gy["m10"] / M_gy["m00"])
        Y_gy = int(M_gy["m01"] / M_gy["m00"])
        return (X_gy,Y_gy)

    def transform(self,X):
        """
        Measures several intensity features:
        @ MeanIntensity: the average pixel intensity
        @ StdIntensity: the standard deviation of pixel intensities
        @ MinIntensity: the minimum of intensity
        @ MaxIntensity: the maximum of intensity
        @ MassDisplacement: difference between the centroids of ROI defined objects in grayscale representation and
        that in binary representation. # Only applicable when self.mask is specified.
        :param X: a list of single-channel arrays.
        :return: ndarray with shape (n_samples, n_intensity_features)
        """
        MeanIntensity = list(map(np.mean,self.roi))
        StdIntensity = list(map(np.std,self.roi))
        MinIntensity = list(map(np.min,self.roi))
        MaxIntensity = list(map(np.max,self.roi))
        if self.mask:
            MassDisplacement = [None]*len(X)
            for (i,(img,mask)) in enumerate(zip(X,self.mask)):
                img_ = img.copy()
                img_[mask==0]=0
                X_bi,Y_bi = self.binary_centroid(img_)
                X_gy,Y_gy = self.gray_centroid(img_)
                MassDisplacement[i] = ((X_bi - X_gy) ** 2 + (Y_bi - Y_gy) ** 2) ** 0.5
            intensity_features = np.vstack((MeanIntensity,StdIntensity,MinIntensity,MaxIntensity,MassDisplacement)).T
        else:
            intensity_features = np.vstack((MeanIntensity,StdIntensity,MinIntensity,MaxIntensity)).T
        return intensity_features
