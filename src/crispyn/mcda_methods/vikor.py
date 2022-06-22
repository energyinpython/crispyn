import numpy as np

from .mcda_method import MCDA_method


class VIKOR(MCDA_method):
    def __init__(self, normalization_method = None, v = 0.5):
        """Create the VIKOR method object.

        Parameters
        -----------

            normalization_method : function
                VIKOR does not use normalization by default, thus `normalization_method` is set to None by default.
                However, you can choose method for normalization of decision matrix chosen `normalization_method` from `normalizations`.
                It is used in a way `normalization_method(X, types)` where `X` is a decision matrix
                and `types` is a vector with criteria types where 1 means profit and -1 means cost.
            v : float
                parameter that is the weight of strategy of the majority of criteria (the maximum group utility)
        """
        self.v = v
        self.normalization_method = normalization_method

    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.
        
        Parameters
        -----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Matrix containing vectors with criteria weights in subsequent rows. 
                Sum of weights in each vector must be equal to 1.
            types: ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.
        
        Returns
        --------
            ndrarray
                Matrix with vectors containing preference values of each alternative. 
                The best alternative has the lowest preference value. Vectors are placed
                in subsequent columns of matrix.
        
        Examples
        ---------
        >>> vikor = VIKOR(normalization_method = minmax_normalization)
        >>> pref = vikor(matrix, weights, types)
        >>> rank = np.zeros((pref.shape))
        >>> for i in range(pref.shape[1]):
        >>>     rank[:, i] = rank_preferences(pref[:, i], reverse = False)
        """
        VIKOR._verify_input_data(matrix, weights, types)
        return VIKOR._vikor(matrix, weights, types, self.normalization_method, self.v)

    @staticmethod
    def _vikor(matrix, weights, types, normalization_method, v):
        # if weights are vector with one dimension
        if len(weights.shape) == 1:
            weights = weights.reshape(1, -1)
        # array for collecting vectors with preference values in subsequent columns
        pref = np.zeros((matrix.shape[0], weights.shape[0]))
        for el, w in enumerate(weights):
            # Without applying a special normalization method
            if normalization_method == None:

                # Determine the best `fstar` and the worst `fmin` values of all criterion function
                maximums_matrix = np.amax(matrix, axis = 0)
                minimums_matrix = np.amin(matrix, axis = 0)

                fstar = np.zeros(matrix.shape[1])
                fmin = np.zeros(matrix.shape[1])

                # for profit criteria (`types` == 1) and for cost criteria (`types` == -1)
                fstar[types == 1] = maximums_matrix[types == 1]
                fstar[types == -1] = minimums_matrix[types == -1]
                fmin[types == 1] = minimums_matrix[types == 1]
                fmin[types == -1] = maximums_matrix[types == -1]

                weighted_matrix = w * ((fstar - matrix) / (fstar - fmin))
            else:
                # With applying the special normalization method
                norm_matrix = normalization_method(matrix, types)
                fstar = np.amax(norm_matrix, axis = 0)
                fmin = np.amin(norm_matrix, axis = 0)
                weighted_matrix = w * ((fstar - norm_matrix) / (fstar - fmin))

            # Calculate the `S` and `R` values
            S = np.sum(weighted_matrix, axis = 1)
            R = np.amax(weighted_matrix, axis = 1)
            # Calculate the Q values
            Sstar = np.min(S)
            Smin = np.max(S)
            Rstar = np.min(R)
            Rmin = np.max(R)
            Q = v * (S - Sstar) / (Smin - Sstar) + (1 - v) * (R - Rstar) / (Rmin - Rstar)
            # save vector with preference values in array `pref`
            pref[:, el] = Q
        return pref