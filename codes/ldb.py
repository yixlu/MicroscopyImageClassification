# install PyCall from Julia's end
import julia
julia.install()

from julia import Main
import os
import numpy as np

# set current directory as working directory
filedir = os.path.dirname(os.path.realpath(__file__))
os.chdir(filedir)
# activate current environment
Main.eval("""
using Pkg;
Pkg.activate(".");
Pkg.instantiate()
""")
Main.using("Wavelets")
Main.using("WaveletsExt")

class LDB_FeatureExtractor:
    """
    Local Discriminant Basis feature extraction class. This is a wrapper object
    for the functions from WaveletsExt.jl in Julia. For more information on this
    package, please visit the documentation on JuliaHub or UCD4IDS on GitHub.

    Attributes:
    - wt => Wavelet used.
    - max_dec_level => Max decomposition level
    - dm => Type of discriminant measure
    - en => Type of energy map
    - dp => Type of discriminant power
    - top_k => Number of coefficients used in each node to determine the
        discriminant measure.
    - n_features => Number of features to be returned as output.
    - n => Length of signal
    - Gamma => Computed energy map
    - DM => Computed discriminant measure
    - cost => Computed wavelet packet decomposition (WPD) tree cost based on the
        discriminant measure DM
    - tree => Compute best WPD tree based on DM
    - DP => Computed discriminant power
    - order => Ordering of DP by descending order
    """
    def __init__(self, wt=Main.wavelet(Main.WT.haar), max_dec_level=Main.nothing,
                 dm=Main.AsymmetricRelativeEntropy(), en=Main.TimeFrequency(),
                 dp=Main.BasisDiscriminantMeasure(), top_k=Main.nothing, 
                 n_features=Main.nothing):
        """
        Initialize LDB object. 
        
        Arguments:
        - wt => Default is Main.wavelet(Main.WT.haar). Other acceptable wavelets
            are:
            * Main.wavelet(Main.WT.dbN)     [replace N with any number between 1:Inf]
            * Main.wavelet(Main.WT.coifN)   [replace N with any number from 2,4,6,8]
            * Main.wavelet(Main.WT.symN)    [replace N with any number between 4:10 inclusive]
            * Main.wavelet(Main.battN)      [replace N with any number from 2,4,6]
        - max_dec_level => Number of decomposition levels. Default of 
            Main.nothing means signal decomposes to max level.
        - dm => Discriminant measure. Default is 
            Main.AsymmetricRelativeEntropy(). Other acceptable values are:
            * Main.SymmetricRelativeEntropy()
            * Main.LpEntropy()
            * Main.HellingerDistance()
        - en => Energy map. Default is Main.TimeFrequency(). Other acceptable 
            value is:
            * Main.ProbabilityDensity()
        - dp => Discriminant power measure. Default is 
            Main.BasisDiscriminantMeasure(). Other acceptable values are:
            * Main.FishersClassSeparability()
            * Main.RobustFishersClassSeparability()
        - top_k => Number of coefficients used in each node to determine the 
            discriminant measure. Default of Main.nothing means all coefficients
            are used.
        - n_features => Number of features to be returned in output. Default of 
            Main.nothing means all features are returned.
        """
        self.wt = wt
        self.max_dec_level = max_dec_level
        self.dm = dm
        self.en = en
        self.dp = dp
        self.top_k = top_k
        self.n_features = n_features
        self.ldb = Main.LocalDiscriminantBasis(wt=self.wt, max_dec_level=self.max_dec_level,
                                               dm=self.dm, en=self.en, dp=self.dp,
                                               top_k=self.top_k, n_features=self.n_features)

    def fit(self, X, y):
        """
        Fits the Local Discriminant Basis feature selection algorithm onto the
        input images X with labels y.
        """
        # restructure data
        n = len(X)
        Xt = np.stack(X, axis=2)
        Xt = Xt.reshape(-1,n)
        Xt = Xt.astype("float")
        # wrapper function for fit!
        Main.eval("""
            function fit(ldb, X, y)
                fit!(ldb, X, y)
            end
        """)
        # fit LDB
        Main.fit(self.ldb, Xt, y)
        # save attributes
        self.n = self.ldb.n
        self.Gamma = self.ldb.Γ
        self.DM = self.ldb.DM
        self.cost = self.ldb.cost
        self.tree = self.ldb.tree
        self.DP = self.ldb.DP
        self.order = self.ldb.order
        return None

    def transform(self, X, y=None):
        """
        Extract the LDB features on signals X.

        Input y is not used, but is included as a parameter by convention and is
        completely optional.
        """
        # restructure data
        n = len(X)
        Xt = np.stack(X, axis=2)
        Xt = Xt.reshape(-1,n)
        Xt = Xt.astype("float")
        # transform data based on LDB
        Xf = Main.transform(self.ldb, Xt)
        # transpose results to follow sklearn convention
        Xf = Xf.T
        return Xf

    def fit_transform(self, X, y):
        """
        Fit and transform the images X with labels y using Local Discriminant
        Basis.
        """
        # restructure data
        n = len(X)
        Xt = np.stack(X, axis=2)
        Xt = Xt.reshape(-1,n)
        Xt = Xt.astype("float")
        # fit and transform the data
        Xf = Main.fit_transform(self.ldb, Xt, y)
        Xf = Xf.T
        return Xf

    def inverse_transform(self, X, y=None):
        """
        Compute the inverse transform on the feature matrix X to form the
        original images based on the LDB class.

        Input y is not used, but is included as a parameter by convention and is
        completely optional.
        """
        # restructure data
        N = X.shape[0]
        x = X.T
        # inverse transform the data
        Xi = Main.inverse_transform(self.ldb, x)
        # restructure data
        Xi = Xi.T
        Xi = Xi.reshape((64, 64, N))
        # return as list of images
        Xm = [None]*N
        for i in range(N):
            Xm[i] = Xi[:,:,i]
        return Xm

    def change_nfeatures(self, X, n_features):
        """
        Change the number of features from self.n_features to n_features.

        Note: if the input n_features is larger than self.n_features, it results
        in the regeneration of signals based on the current self.n_features 
        before reselecting the features. This will cause additional features to
        be less accurate and effective.
        """
        self.n_features = n_features
        # transpose data to fit Julia convention
        Xt = X.T
        # change number of features
        Xt = Main.change_nfeatures(self.ldb, Xt, n_features)
        # transpose data to fit Python convention
        Xt = Xt.T
        return Xt