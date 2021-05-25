import cv2
import numpy as np
from sklearn.cluster import KMeans

class SIFT_FeatureExtractor:
    """
    SIFT Feature Extractor class. This feature extraction technique goes through
    the following steps:

    1. Using SIFT to extract keypoints and descriptors of each image.
    2. Using k-means to cluster the descriptors.
    3. Construct "barplots" for each image on the number of descriptors found in each 
    cluster. This step is also known as the Bag of Features (BoF) step.
    4. The results are stored in a matrix and output as transformed data.

    In the initialization step, we need to specify the following parameters:

    - sift_nfeatures => default=0 (all features used)
    - kmeans_nclusters => default=5
    """
    def __init__(self, sift_nfeatures=0, kmeans_nclusters=5):
        """
        Constructor for the SIFT Feature Extractor object.
        """
        self.sift_nfeatures = sift_nfeatures
        self.kmeans_nclusters = kmeans_nclusters

    def fit(self, X):
        """
        Fit images into SIFT_FeatureExtractor. Input images should be in the form of a list 
        of numpy arrays. This function computes the following steps:

        1. Using SIFT to extract keypoints and descriptors of each image.
        2. Using k-means to cluster the descriptors.
        """
        # SIFT feature extraction
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.sift_nfeatures)
        kp = self.sift.detect(X, None)
        kp, des = self.sift.compute(X, kp)
        des_all = np.vstack(tuple(des))
        # K-Means clustering of features
        self.kmeans = KMeans(n_clusters=self.kmeans_nclusters)
        self.kmeans.fit(des_all)

    def transform(self, X):
        """
        Transform images into its Bag of Features (BoF) "barplots". Input images should be
        in the form of a list of numpy arrays. This function computes the following steps:

        1. Using SIFT to extract keypoints and descriptors of each image.
        2. Assign the descriptors of each image to their nearest clusters.
        3. Construct "barplots" for each image on the number of descriptors found in each 
        cluster.
        4. The results are stored in a matrix and output as transformed data.

        Note: For the training data, it is not advised to run fit() and transform()
        separately as SIFT step becomes redundant. Please refer to fit_transform() for
        the training data.
        """
        # SIFT feature extraction
        kp = self.sift.detect(X, None)
        kp, des = self.sift.compute(X, kp)
        n = len(X)
        Xt = np.zeros((n, self.kmeans_nclusters))
        # data transformation for each image
        for (i, de) in enumerate(des):
            labs = self.kmeans.predict(de)
            v, c = np.unique(labs, return_counts=True)
            for j in range(self.kmeans_nclusters):
                if j in v:
                    idx = np.where(v == j)
                    Xt[i,j] = c[idx[0][0]]
        return Xt

    def fit_transform(self, X):
        """
        Fit and transform the images into its Bag of Features (BoF) "barplots". Input images
        should be in the form of a list of numpy arrays. This function computes the
        following steps:

        1. Using SIFT to extract keypoints and descriptors of each image.
        2. Using k-means to cluster the descriptors.
        3. Assign the descriptors of each image to their nearest clusters.
        4. Construct "barplots" for each image on the number of descriptors found in each 
        cluster.
        5. The results are stored in a matrix and output as transformed data.
        """
         # SIFT feature extraction
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.sift_nfeatures)
        kp = self.sift.detect(X, None)
        kp, des = self.sift.compute(X, kp)
        des_all = np.vstack(tuple(des))
        # K-Means clustering of features
        self.kmeans = KMeans(n_clusters=self.kmeans_nclusters)
        self.kmeans.fit(des_all)
        n = len(X)
        Xt = np.zeros((n, self.kmeans_nclusters))
        # data transformation for each image
        for (i, de) in enumerate(des):
            labs = self.kmeans.predict(de)
            v, c = np.unique(labs, return_counts=True)
            for j in range(self.kmeans_nclusters):
                if j in v:
                    idx = np.where(v == j)
                    Xt[i,j] = c[idx[0][0]]
        return Xt