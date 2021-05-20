import pywt
import numpy as np
from scipy.fftpack import dct

class SWT_FeatureExtractor:
    """
    Stationary Wavelet Transform (SWT) based feature extractor. This method is based on
    Qayyum et al's "Facial Expression Recognition Using Stationary Wavelet Transform
    Features". The procedures are as follows:

    1. n levels of SWT decomposition of each image.
    2. Compute the 8x8 block Discrete Cosine Transform (DCT) on each set of detail
    coefficients.
    3. Reshape and output the results as 1D vectors.
    """
    def __init__(self, wt="haar", n_levels=1):
        """
        Initializer for the feature extractor. Specify the number of wavelet type and 
        decomposition levels, default is set as wt="haar" and n_levels=1.
        """
        self.wt = wt
        self.n_levels = n_levels

    def fit_transform(self, X):
        """
        Perform data transformation
        """
        Xt = np.empty((len(X), 192*self.n_levels))
        for (i, img) in enumerate(X):
            _, *Xw = pywt.swt2(img, self.wt, self.n_levels, start_level=0, trim_approx=True)
            # counter to track column of Xt
            counter = 0
            for detail_coefs in Xw:
                for mat in detail_coefs:
                    dmat = dct(dct(mat, 2, axis=0, norm="ortho"), 2, axis=1, norm="ortho")
                    Xt[i,counter:(counter+64)] = dmat[::8, ::8].reshape(-1)
                    counter += 64
        return Xt
