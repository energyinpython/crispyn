import numpy as np

from .vikor import VIKOR
from ..additions import rank_preferences


class VIKOR_SMAA():
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
            weights : ndarray
                Matrix with i vectors in rows of n weights in columns. i means number of
                iterations of SMAA
            types : ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.
        
        Returns
        --------
            ndrarray, ndarray, ndarray
                Matrix with acceptability indexes values for each alternative in rows in relation to each rank in columns,
                Matrix with central weight vectors for each alternative in rows
                Matrix with final ranking of alternatives
        
        Examples
        ---------
        >>> vikor_smaa = VIKOR_SMAA(normalization_method = minmax_normalization)
        >>> rank_acceptability_index, central_weight_vector, rank_scores = vikor_smaa(matrix, weights, types)
        """

        return VIKOR_SMAA._vikor_smaa(self, matrix, weights, types, self.normalization_method, self.v)


    # function to generate multiple weight vectors
    # returns matrix with n weights in columns and number of vectors in rows equal to iterations number
    def _generate_weights(self, n, iterations):
        """
        Function to generate multiple weight vectors

        Parameters
        -----------
            n : int
                Number of criteria
            iterations : int
                Number of weight vector to generate

        Returns
        ----------
            ndarray
                Matrix containing in rows vectors with weights for n criteria

        """
        weight_vectors = np.zeros((iterations, n))
        # n weight generation - when no preference information available
        # generate n - 1 uniform distributed weights within the range [0, 1]
        for i in range(iterations):
            w = np.random.uniform(0, 1, n)

            # sort weights into ascending order (q[1], ..., q[n-1])
            ind = np.argsort(w)
            w = w[ind]

            # insert 0 as the first q[0] and 1 as the last (q[n]) numbers
            w = np.insert(w, 0, 0)
            w = np.insert(w, len(w), 1)

            # the weights are obtained as intervals between consecutive numbers (w[j] = q[j] - q[j-1])
            weights = [w[i] - w[i - 1] for i in range(1, n + 1)]
            weights = np.array(weights)

            # scale the generated weights so that their sum is 1
            new_weights = weights / np.sum(weights)
            weight_vectors[i, :] = new_weights
        return weight_vectors


    @staticmethod
    def _vikor_smaa(self, matrix, weights, types, normalization_method, v):
        m, n = matrix.shape

        # Central weight vector for each alternative
        central_weight_vector = np.zeros((m, n))
        
        # Rank acceptability index of each place for each alternative
        rank_acceptability_index = np.zeros((m, m))

        # Ranks
        rank_score = np.zeros(m)

        vikor = VIKOR()
    
        # Calculate alternatives preference function values with VIKOR method
        pref = vikor(matrix, weights, types)

        # Calculate rankings based on preference values
        rank = np.zeros((pref.shape))
        for i in range(pref.shape[1]):
            rank[:, i] = rank_preferences(pref[:, i], reverse = False)

            # add value for the rank acceptability index for each alternative considering rank and rank score
            # iteration by each alternative
            rr = rank[:, i]
            for k, r in enumerate(rr):
                rank_acceptability_index[k, int(r - 1)] += 1
                # rank score
                # calculate how many alternatives have worst preference values than k-th alternative
                # Note: in VIKOR better alternatives have lower preference values
                better_ranks = rr[rr > rr[k]]
                # add to k-th index value 1 for each alternative that is worse than k-th alternative
                rank_score[k] += len(better_ranks)
            
            # add central weights for the best scored alternative
            ind_min = np.argmin(rr)
            central_weight_vector[ind_min, :] += weights[i, :]

        #
        # end of loop for i iterations
        # Calculate the rank acceptability index
        rank_acceptability_index = rank_acceptability_index / pref.shape[1]

        # Calculate central the weights vectors
        central_weight_vector = central_weight_vector / pref.shape[1]
        for i in range(m):
            if np.sum(central_weight_vector[i, :]):
                central_weight_vector[i, :] = central_weight_vector[i, :] / np.sum(central_weight_vector[i, :])

        # Calculate rank scores
        rank_score = rank_score / pref.shape[1]
        rank_scores = rank_preferences(rank_score, reverse = True)

        return rank_acceptability_index, central_weight_vector, rank_scores