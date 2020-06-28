import numpy as np
from scipy import io
from scipy import sparse
import warnings
from sklearn.utils import *
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (check_is_fitted, check_random_state,
                                FLOAT_DTYPES)
from sklearn.utils.sparsefuncs import (inplace_column_scale,
                                mean_variance_axis, incr_mean_variance_axis,
                                min_max_axis)

def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale
    
class SStandardScaler(TransformerMixin, BaseEstimator):
    def __init__(self, axis=0, with_mean=True, with_std=True, copy=True):
        # self.features = features
        self.axis=axis
        self.with_mean=with_mean
        self.with_std=with_std
        self.copy=copy
    
    def scale(self, X, ddof=True):
        
        """Standardize a dataset along any axis

        Center to the mean and component wise scale to unit variance.

        Read more in the :ref:`User Guide <preprocessing_scaler>`.

        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data to center and scale.

        axis : int (0 by default)
            axis used to compute the means and standard deviations along. If 0,
            independently standardize each feature, otherwise (if 1) standardize
            each sample.

        with_mean : boolean, True by default
            If True, center the data before scaling.

        with_std : boolean, True by default
            If True, scale the data to unit variance (or equivalently,
            unit standard deviation).

        copy : boolean, optional, default True
            set to False to perform inplace row normalization and avoid a
            copy (if the input is already a numpy array or a scipy.sparse
            CSC matrix and if axis is 1).

        Notes
        -----
        This implementation will refuse to center scipy.sparse matrices
        since it would make them non-sparse and would potentially crash the
        program with memory exhaustion problems.

        Instead the caller is expected to either set explicitly
        `with_mean=False` (in that case, only variance scaling will be
        performed on the features of the CSC matrix) or to call `X.toarray()`
        if he/she expects the materialized dense array to fit in memory.

        To avoid memory copy the caller should pass a CSC matrix.

        NaNs are treated as missing values: disregarded to compute the statistics,
        and maintained during the data transformation.

        We use a biased estimator for the standard deviation, equivalent to
        `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
        affect model performance.

        For a comparison of the different scalers, transformers, and normalizers,
        see :ref:`examples/preprocessing/plot_all_scaling.py
        <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

        See also
        --------
        StandardScaler: Performs scaling to unit variance using the``Transformer`` API
            (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).

        """  # noqa
        X = check_array(X, accept_sparse='csc', copy=self.copy, ensure_2d=False,
                        estimator='the scale function', dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')
        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` instead"
                    " See docstring for motivation and alternatives.")
            if self.axis != 0:
                raise ValueError("Can only scale sparse matrix on axis=0, "
                                " got axis=%d" % self.axis)
            if self.with_std:
                _, var = mean_variance_axis(X, axis=0)
                var = _handle_zeros_in_scale(var, copy=False)
                inplace_column_scale(X, 1 / np.sqrt(var))
        else:
            X = np.asarray(X)
            if self.with_mean:
                mean_ = np.nanmean(X, self.axis)
            if self.with_std:
                if ddof:
                    scale_ = np.std(X, axis=self.axis, ddof=1)
                else:
                    scale_ = np.nanstd(X, axis)
                
            # Xr is a view on the original array that enables easy use of
            # broadcasting on the axis in which we are interested in
            Xr = np.rollaxis(X, self.axis)
            if self.with_mean:
                Xr -= mean_
                mean_1 = np.nanmean(Xr, axis=0)
                # Verify that mean_1 is 'close to zero'. If X contains very
                # large values, mean_1 can also be very large, due to a lack of
                # precision of mean_. In this case, a pre-scaling of the
                # concerned feature is efficient, for instance by its mean or
                # maximum.
                if not np.allclose(mean_1, 0):
                    Xr -= mean_1
            if self.with_std:
                scale_ = _handle_zeros_in_scale(scale_, copy=False)
                Xr /= scale_
                if self.with_mean:
                    mean_2 = np.nanmean(Xr, axis=0)
                    # If mean_2 is not 'close to zero', it comes from the fact that
                    # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
                    # if mean_1 was close to zero. The problem is thus essentially
                    # due to the lack of precision of mean_. A solution is then to
                    # subtract the mean again:
                    if not np.allclose(mean_2, 0):
                        warnings.warn("Numerical issues were encountered "
                                    "when scaling the data "
                                    "and might not be solved. The standard "
                                    "deviation of the data is probably "
                                    "very close to 0. ")
                        Xr -= mean_2
        return X

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None, ddof=True):
        return self.scale(X, ddof=ddof)

if __name__ == '__main__':
    mat_file = io.loadmat('./Data Base/DB3.mat')
    features_name = mat_file['features_name']
    classes = mat_file['classes']
    features = mat_file['features']
    n = 0 #np.array([[0]], dtype=np.uint8)
    N = 0 # np.array([[0]], dtype=np.uint8)
    R = 1 #np.array([[1]])
    
    Per = 0.95
    
    # features_n = normalized_features(features)
    
    
    
    
    
    
    scale = SStandardScaler().fit_transform(features)
    
    
    
    
    
    
    