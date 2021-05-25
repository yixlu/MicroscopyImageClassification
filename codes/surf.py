import cv2
import mahotas
import numpy as np
from sklearn.cluster import KMeans

class SURF_FeatureExtractor:
    """
    Speeded-up Robust Features (SURF) Extractor class. SURF is very similar to SIFT but faster and less computationally
    expensive.This feature extraction technique goes through the following steps:

    1. Using SURF to identify keypoints and generate descriptors for each keypoint.
    2. Using k-means to cluster the descriptors.
    3. Construct "barplots" for each image on the number of descriptors found in each
    cluster. This step is also known as the Bag of Features (BoF) step.
    4. The results are stored in a matrix and output as transformed data.
    """
    def __init__(self, hessianThreshold=100,extended=False,kmeans_nclusters=5):
        """
        Constructor for the SURF Feature Extractor object.
        :param hessianThreshold: threshold of Hessian key point selector. Default to 100.
        :param extended: whether to use extended version of descriptors, which has 128 elements. If false, 64-element
        descriptors are calculated. Default to false.
        :param kmeans_nclusters: number of clusters to fit using kmeans. Default to 5.
        """

        self.hessianThreshold = hessianThreshold
        self.extended=extended
        self.kmeans_nclusters = kmeans_nclusters

    def fit(self, X):
        """
        Fit images into SURF_FeatureExtractor. Input images should be in the form of a list of ndarrays.
        :param X: a list of ndarrays as input.
        :return: object itself
        """
        # SURF feature extraction
        self.surf = cv2.xfeatures2d.SURF_create(hessianThreshold=self.hessianThreshold,extended=self.extended)
        des = [None]*len(X)
        for (i,img) in enumerate(X):
            kp_surf, des_surf = self.surf.detectAndCompute(img, None)
            des[i] = np.vstack(des_surf)
        des_all = np.vstack(des)
        # K-Means clustering of features
        self.kmeans = KMeans(n_clusters=self.kmeans_nclusters)
        self.kmeans.fit(des_all)
        return self

    def transform(self, X):
        """
        Transform images into its Bag of Features (BoF) "barplots". Input images should be
        in the form of a list of numpy arrays.
        Note: For the training data, it is not advised to run fit() and transform()
        separately as SURF step becomes redundant. Please refer to fit_transform() for
        the training data.
        """
        # SURF feature extraction
        des = [None] * len(X)
        for (i, img) in enumerate(X):
            kp_surf, des_surf = self.surf.detectAndCompute(img, None)
            des[i] = np.vstack(des_surf)
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
        should be in the form of a list of numpy arrays.
        """
        # SURF feature extraction
        self.surf = cv2.xfeatures2d.SURF_create(hessianThreshold=self.hessianThreshold, extended=self.extended)
        des = [None] * len(X)
        for (i, img) in enumerate(X):
            kp_surf, des_surf = self.surf.detectAndCompute(img, None)
            des[i] = np.vstack(des_surf)
        des_all = np.vstack(des)
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