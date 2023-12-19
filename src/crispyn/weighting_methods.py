import copy
import itertools
from scipy import linalg
from scipy.sparse.linalg import eigs

import numpy as np
from .correlations import pearson_coeff
from .normalizations import sum_normalization, minmax_normalization



# Equal weighting
def equal_weighting(matrix):
    """
    Calculate criteria weights using objective Equal weighting method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.

    Returns
    --------
        ndarray
            Vector of criteria weights.

    Examples
    ----------
    >>> weights = equal_weighting(matrix)
    """
    N = np.shape(matrix)[1]
    w = np.ones(N) / N
    return w


# Entropy weighting
def entropy_weighting(matrix):
    """
    Calculate criteria weights using objective Entropy weighting method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.

    Returns
    --------
        ndarray
            Vector of criteria weights.

    Examples
    ----------
    >>> weights = entropy_weighting(matrix)
    """
    # normalize the decision matrix with `sum_normalization` method from `normalizations` as for profit criteria
    types = np.ones(np.shape(matrix)[1])
    pij = sum_normalization(matrix, types)
    # Transform negative values in decision matrix `matrix` to positive values
    pij = np.abs(pij)
    m, n = np.shape(pij)
    H = np.zeros((m, n))

    # Calculate entropy
    for j, i in itertools.product(range(n), range(m)):
        if pij[i, j]:
            H[i, j] = pij[i, j] * np.log(pij[i, j])

    h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))

    # Calculate degree of diversification
    d = 1 - h

    # Set w as the degree of importance of each criterion
    w = d / (np.sum(d))
    return w


# Standard Deviation weighting
def std_weighting(matrix):
    """
    Calculate criteria weights using objective Standard deviation weighting method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.
            
    Returns
    --------
        ndarray
            Vector of criteria weights.

    Examples
    ----------
    >>> weights = std_weighting(matrix)
    """
    
    # Calculate the standard deviation of each criterion in decision matrix
    stdv = np.sqrt((np.sum(np.square(matrix - np.mean(matrix, axis = 0)), axis = 0)) / (matrix.shape[0]))
    # Calculate criteria weights by dividing the standard deviations by their sum
    w = stdv / np.sum(stdv)
    return w


# CRITIC weighting
def critic_weighting(matrix):
    """
    Calculate criteria weights using objective CRITIC weighting method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.
            
    Returns
    --------
        ndarray
            Vector of criteria weights.

    Examples
    ----------
    >>> weights = critic_weighting(matrix)
    """
    # Normalize the decision matrix using Minimum-Maximum normalization `minmax_normalization` from `normalizations` as for profit criteria
    types = np.ones(np.shape(matrix)[1])
    x_norm = minmax_normalization(matrix, types)
    # Calculate the standard deviation
    std = np.std(x_norm, axis = 0)
    n = np.shape(x_norm)[1]
    # Calculate correlation coefficients of all pairs of columns of normalized decision matrix
    correlations = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        correlations[i, j] = pearson_coeff(x_norm[:, i], x_norm[:, j])

    # Calculate the difference between 1 and calculated correlations
    difference = 1 - correlations
    # Multiply the difference by the standard deviation
    C = std * np.sum(difference, axis = 0)
    # Calculate the weights by dividing vector with `C` by their sum
    w = C / np.sum(C)
    return w


# Gini coefficient-based weighting
def gini_weighting(matrix):
    """
    Calculate criteria weights using objective Gini coefficient-based weighting method.

    Parameters
    ----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.
            
    Returns
    --------
        ndarray
            Vector of criteria weights.

    Examples
    ---------
    >>> weights = gini_weighting(matrix)
    """
    m, n = np.shape(matrix)
    G = np.zeros(n)
    # Calculate the Gini coefficient for decision matrix `matrix`
    # iteration over criteria j = 1, 2, ..., n
    for j in range(n):
        Yi = 0
        # iteration over alternatives i = 1, 2, ..., m
        if np.mean(matrix[:, j]):
            for i in range(m):
                Yi += np.sum(np.abs(matrix[i, j] - matrix[:, j]) / (2 * m**2 * (np.sum(matrix[:, j]) / m)))
        else:
            for i in range(m):
                Yi += np.sum(np.abs(matrix[i, j] - matrix[:, j]) / (m**2 - m))

        G[j] = Yi
    # calculate and return the criteria weights by dividing the vector of Gini coefficients by their sum
    w = G / np.sum(G)
    return w


# MEREC weighting
def merec_weighting(matrix, types):
    """
    Calculate criteria weights using objective MEREC weighting method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.
        types : ndarray
            Vector with criteria types.
            
    Returns
    --------
        ndarray
            Vector of criteria weights.

    Examples
    ---------
    >>> weights = merec_weighting(matrix, types)
    """
    X = copy.deepcopy(matrix)
    m, n = X.shape
    # Transform negative values in decision matrix X to positive values
    X = np.abs(X)
    # Normalize the decision matrix X with linear normalization method
    norm_matrix = np.zeros(X.shape)
    norm_matrix[:, types == 1] = np.min(X[:, types == 1], axis = 0) / X[:, types == 1]
    norm_matrix[:, types == -1] = X[:, types == -1] / np.max(X[:, types == -1], axis = 0)
    
    # Calculate the overall performance of the values in normalized matrix using a logarithmic measure with equal criteria weights
    S = np.log(1 + ((1 / n) * np.sum(np.abs(np.log(norm_matrix)), axis = 1)))

    # Calculate the performance of the alternatives by removing each criterion using the logarithmic measure
    Sp = np.zeros(X.shape)

    for j in range(n):
        norm_mat = np.delete(norm_matrix, j, axis = 1)
        Sp[:, j] = np.log(1 + ((1 / n) * np.sum(np.abs(np.log(norm_mat)), axis = 1)))

    # Calculate the summation of absolute deviations
    E = np.sum(np.abs(Sp - S.reshape(-1, 1)), axis = 0)

    # Calculate the final weights of the criteria
    w = E / np.sum(E)
    return w


# Statistical Variance weighting
def stat_var_weighting(matrix):
    """
    Calculate criteria weights using objective Statistical variance weighting method.

    Parameters
    ----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.
            
    Returns
    -------
        ndarray
            Vector of criteria weights.

    Examples
    ---------
    >>> weights = stat_var_weighting(matrix)
    """

    # Normalize the decision matrix `matrix` with `minmax_normalization` method from normalizations
    types = np.ones(np.shape(matrix)[1])
    xn = minmax_normalization(matrix, types)
    # Calculate the statistical variance for each criterion
    v = np.mean(np.square(xn - np.mean(xn, axis = 0)), axis = 0)
    # Calculate the final weights of the criteria
    w = v / np.sum(v)
    return w


# CILOS weighting
def cilos_weighting(matrix, types):
    """
    Calculate criteria weights using objective CILOS weighting method.

    Parameters
    ----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.
        types : ndarray
            Vector with criteria types.
            
    Returns
    -------
        ndarray
            Vector of criteria weights.

    Examples
    >>> weights = cilos_weighting(matrix, types)
    """

    xr = copy.deepcopy(matrix)
    # Convert negative criteria to positive criteria
    xr[:, types == -1] = np.min(matrix[:, types == -1], axis = 0) / matrix[:, types == -1]
    # Normalize the decision matrix `xr` using the sum normalization method
    xn = xr / np.sum(xr, axis = 0)
    
    # Calculate the square matrix
    A = xn[np.argmax(xn, axis = 0), :]
    
    # Calculate relative impact loss matrix
    pij = np.zeros((matrix.shape[1], matrix.shape[1]))
    for j, i in itertools.product(range(matrix.shape[1]), range(matrix.shape[1])):
        pij[i, j] = (A[j, j] - A[i, j]) / A[j, j]

    # Determine the weight system matrix
    F = np.diag(-np.sum(pij - np.diag(np.diag(pij)), axis = 0)) + pij
    
    # Calculate the criterion impact loss weight
    # The criteria weights q are determined from the formulated homogeneous linear system of equations
    # Solve the system equation
    q = linalg.null_space(F)
    
    # Calculate and return the final weights of the criteria
    weights = q / np.sum(q)
    return np.ravel(weights)


# IDOCRIW weighting
def idocriw_weighting(matrix, types):
    """
    Calculate criteria weights using objective IDOCRIW weighting method.

    Parameters
    ----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.
        types : ndarray
            Vector with criteria types.
            
    Returns
    -------
        ndarray
            Vector of criteria weights.

    Examples
    ---------
    >>> weights = idocriw_weighting(matrix, types)
    """
    # Calculate the Entropy weights
    q = entropy_weighting(matrix)
    # Calculate the CILOS weights
    w = cilos_weighting(matrix, types)
    # Aggregate the weight value of the attributes considering Entropy and CILOS weights
    weights = (q * w) / np.sum(q * w)
    return weights


# Angle weighting
def angle_weighting(matrix, types):
    """
    Calculate criteria weights using objective Angle weighting method.

    Parameters
    ----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.
        types : ndarray
            Vector with criteria types.
            
    Returns
    -------
        ndarray
            Vector of criteria weights.

    Examples
    ---------
    >>> weights = angle_weighting(matrix, types)
    """
    m, n = matrix.shape
    # Normalize the decision matrix X using sum_normalization method from normalizations
    X = sum_normalization(matrix, types)
    # Calculate elements of additional column (the reference attribute) which are equal to 1 / m
    B = np.ones(m) * (1 / m)
    # Calculate the angle between attraibutes in decision matrix X and the reference attribute
    u = np.arccos(np.sum(X / m, axis = 0) / (np.sqrt(np.sum(X ** 2, axis = 0)) * np.sqrt(np.sum(B ** 2))))
    # Calculate the final angle weights for each criterion
    w = u / np.sum(u)
    return w


# Coeffcient of variation weighting
def coeff_var_weighting(matrix):
    """
    Calculate criteria weights using objective Coefficient of variation weighting method.
    
    Parameters
    ----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.
            
    Returns
    -------
        ndarray
            Vector of criteria weights.

    Examples
    ---------
    >>> weights = coeff_var_weighting(matrix)
    """
    m, n = matrix.shape
    # Normalize the decision matrix `matrix` with `sum_normalization` method from `normalizations`
    types = np.ones(n)
    B = sum_normalization(matrix, types)

    # Calculate the standard deviation of each column
    Bm = np.sum(B, axis = 0) / m
    std = np.sqrt(np.sum(((B - Bm)**2), axis = 0) / (m - 1))

    # Calculate the Coefficient of Variation for each criterion
    ej = std / Bm
    # Calculate the weights for each criterion
    w = ej / np.sum(ej)
    return w


# AHP weighting
class AHP_WEIGHTING():

    def __init__(self):
        """Create object of the AHP weighting method"""
        
        pass

    def __call__(self, X, compute_priority_vector_method = None):

        if compute_priority_vector_method is None:
            compute_priority_vector_method = self._eigenvector

        return AHP_WEIGHTING._ahp_weighting(self, X, compute_priority_vector_method)


    def _check_consistency(self, X):
        """
        Consistency Check on the Pairwise Comparison Matrix of the Criteria or alternatives

        Parameters
        -----------
            X : ndarray
                matrix of pairwise comparisons

        Examples
        ----------
        >>> PCcriteria = np.array([[1, 1, 5, 3], [1, 1, 5, 3], [1/5, 1/5, 1, 1/3], [1/3, 1/3, 3, 1]])
        >>> ahp_weighting = AHP_WEIGHTING()
        >>> ahp_weighting._check_consistency(PCcriteria)
        """

        n = X.shape[1]
        RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
        lambdamax = np.amax(np.linalg.eigvals(X).real)
        CI = (lambdamax - n) / (n - 1)
        CR = CI / RI[n - 1]
        print("Inconsistency index: ", CR)
        if CR > 0.1:
            print("The pairwise comparison matrix is inconsistent")


    def _eigenvector(self, X):
        """
        Compute the Priority Vector of Criteria (weights) or alternatives using Eigenvector method

        Parameters
        -----------
            X : ndarray
                matrix of pairwise comparisons

        Returns
        ---------
            ndarray
                Eigenvector

        Examples
        ----------
        >>> PCM1 = np.array([[1, 5, 1, 1, 1/3, 3],
        [1/5, 1, 1/3, 1/5, 1/7, 1],
        [1, 3, 1, 1/3, 1/5, 1],
        [1, 5, 3, 1, 1/3, 3],
        [3, 7, 5, 3, 1, 7],
        [1/3, 1, 1, 1/3, 1/7, 1]])
        >>> ahp = AHP()
        >>> S = ahp._eigenvector(PCM1)
        """

        val, vec = eigs(X, k=1)
        eig_vec = np.real(vec)
        S = eig_vec / np.sum(eig_vec)
        S = S.ravel()
        return S


    def _normalized_column_sum(self, X):
        """
        Compute the Priority Vector of Criteria (weights) or alternatives using The normalized column sum method

        Parameters
        -----------
            X : ndarray
                matrix of pairwise comparisons

        Returns
        ---------
            ndarray
                Vector with weights calculated with The normalized column sum method

        Examples
        ----------
        >>> PCM1 = np.array([[1, 5, 1, 1, 1/3, 3],
        [1/5, 1, 1/3, 1/5, 1/7, 1],
        [1, 3, 1, 1/3, 1/5, 1],
        [1, 5, 3, 1, 1/3, 3],
        [3, 7, 5, 3, 1, 7],
        [1/3, 1, 1, 1/3, 1/7, 1]])
        >>> ahp = AHP()
        >>> S = ahp._normalized_column_sum(PCM1)
        """

        return np.sum(X, axis = 1) / np.sum(X)


    def _geometric_mean(self, X):
        """
        Compute the Priority Vector of Criteria (weights) or alternatives using The geometric mean method

        Parameters
        -----------
            X : ndarray
                matrix of pairwise comparisons

        Returns
        ---------
            ndarray
                Vector with weights calculated with The geometric mean method

        Examples
        ----------
        >>> PCM1 = np.array([[1, 5, 1, 1, 1/3, 3],
        [1/5, 1, 1/3, 1/5, 1/7, 1],
        [1, 3, 1, 1/3, 1/5, 1],
        [1, 5, 3, 1, 1/3, 3],
        [3, 7, 5, 3, 1, 7],
        [1/3, 1, 1, 1/3, 1/7, 1]])
        >>> ahp = AHP()
        >>> S = ahp._geometric_mean(PCM1)
        """

        n = X.shape[1]
        numerator = (np.prod(X, axis = 1))**(1 / n)
        denominator = np.sum(numerator)
        return numerator / denominator
    
    @staticmethod
    def _ahp_weighting(self, X, compute_priority_vector_method):
        """
        Calculate criteria weights using subjective AHP weighting method based on
        provided pairwise criteria comparison matrix

        Parameters
        ------------
            X : ndarray
                pairwise criteria comparison matrix

            compute_priority_vector_method : function
                selected function for calculation priority vector
                eigenvector, _normalized_column_sum, _geometric_mean

        Returns
        -------------
            ndarray
                Vector of criteria weights.

        Examples
        -------------
        >>> PCcriteria = np.array([[1, 1, 5, 3], [1, 1, 5, 3], 
            [1/5, 1/5, 1, 1/3], [1/3, 1/3, 3, 1]])
        >>> ahp_weighting = AHP_WEIGHTING()
        >>> weights = ahp_weighting(X = PCcriteria, compute_priority_vector_method=ahp_weighting._normalized_column_sum)
        """
    
        self._check_consistency(X)
        weights = compute_priority_vector_method(X)
        return weights


# SWARA weighting
def swara_weighting(criteria_indexes, s):
    """
    Calculation of criteria weights using SWARA subjective weighting method

    Parameters
    -------------
        criteria_indexes : ndarray
            Vector with indexes of n criteria in accordance with given decision problem from C1 to Cn ordered
            in descending order beginning from the most important criterion
            (Vector with sorted evaluation criteria in descending order, based on their expected significances)

        s : ndarray
            The s vector containing n-1 values of criteria comparison generated in following way: 
            Make the respondent express how much criterion j-1 is more significant than 
            criterion j in percentage in range [0, 1]

    Returns
    ------------
        ndarray
            Vector with criteria weights

    Examples
    -----------
    >>> criteria_indexes = np.array([0, 1, 2, 3, 4, 5, 6])
    >>> s = np.array([0, 0.35, 0.2, 0.3, 0, 0.4])
    >>> swara_weights = swara_weighting(criteria_indexes, s)
    """

    # Calculation of SWARA weights for ordered criteria
    # First criterion is considered as most important

    # Adding 0 at first index
    s = np.insert(s, 0, 0)

    # Determination of the k coefficient
    k = np.ones(len(s))

    # Determination of the recalculated weight q
    q = np.ones(len(s))
    for j in range(1, len(s)):
        k[j] = s[j] + 1
        q[j] = q[j - 1] / k[j]

    # Determination of the relative weights of the evaluation criteria
    weights = q / np.sum(q)
    # Assigning criteria weights according to their original order of criteria in given decision problem
    indexes = np.argsort(criteria_indexes)
    weights = weights[indexes]
    return weights


# LBWA weighting
def lbwa_weighting(criteria_indexes, criteria_values_I):
    """
    Calculation of criteria weights using subjective LBWA weighting method.

    Parameters
    -------------
        criteria_indexes : list including sublists
            A list including sublists containing grouped and ordered indexes of criteria in a given decision problem from C1 to Cn according to their significance, beginning from the most significant

        criteria_values_I : list including sublists
            A list including sublists containing influence values of criteria within each subset provided in order beginning from the most significant

    Returns
    --------------
        ndarray
            Vector of criteria weights

    Examples
    --------------
    >>> criteria_indexes = [
            [1, 4, 6, 5, 0, 2],
            [7, 3]
        ]
    >>>  criteria_values_I = [
            [0, 2, 3, 4, 4, 5],
            [1, 2]
        ]
    >>> weights = lbwa_weighting(criteria_indexes, criteria_values_I)

    >>> criteria_indexes = [
            [4, 7, 8, 0],
            [2, 3],
            [],
            [5],
            [],
            [],
            [1, 6]
        ]

    >>> criteria_values_I = [
            [0, 1, 2, 4],
            [1, 2],
            [],
            [2],
            [],
            [],
            [1, 3]
        ]

    >>> weights = lbwa_weighting(criteria_indexes, criteria_values_I)
    """
    # Determination of r coefficient
    r = 0
    for el in criteria_values_I:
        if len(el) > r:
            r = len(el)

    # Determination of r0 elasticity coefficient
    r0 = r + 1

    lbwa_influence_function = copy.deepcopy(criteria_values_I)

    best_weight = 0
    lenght = 0

    # Calculation of the influence function of the criteria
    for ind1, el1 in enumerate(criteria_values_I):
        for ind2, el2 in enumerate(el1):
            lbwa_influence_function[ind1][ind2] = r0 / ((ind1 + 1) * r0 + el2)
            best_weight += r0 / ((ind1 + 1) * r0 + el2)
            lenght += 1

    # Calculation of the optimum values of the weight coefficients of criteria
    # it is calculated the weight coefficient of the most significant criterion
    best_weight = 1 / best_weight

    # Calculation of the weight coefficients of the remaining criteria
    weights = np.zeros(lenght)

    for ind1, el1 in enumerate(lbwa_influence_function):
        for ind2, el2 in enumerate(el1):
            weights[criteria_indexes[ind1][ind2]] = lbwa_influence_function[ind1][ind2] * best_weight
        
    # Criterion considered as the most important criterion has index [0][0] in list with influence values
    weights[criteria_indexes[0][0]] = best_weight
    return weights


# SAPEVO weighting
def sapevo_weighting(criteria_matrix):
    """
    Calculate criteria weights using SAPEVO subjective weighting method

    Parameters
    ------------
        criteria_matrix : ndarray
            Matrix with degrees of pairwise criteria comparison in scale from -3 to 3

    Returns
    ----------
        ndarray:
            Vector of criteria weights

    Examples
    -----------
    >>> criteria_matrix = np.array([
        [0, 0, 3, 3, 1, 3, 2, 1, 2],
        [0, 0, 3, 3, 1, 3, 2, 1, 2],
        [-3, -3, 0, 0, -1, -2, -2, -1, -2],
        [-3, -3, 0, 0, -2, 2, -2, -2, -2],
        [-1, -1, 1, 2, 0, 2, 0, -1, 1],
        [-3, -3, 2, -2, -2, 0, -2, -1, -2],
        [-3, -2, 2, 2, 0, 2, 0, 3, 0],
        [-1, -1, 1, 2, 1, 1, -3, 0, -1],
        [-2, -2, 2, 2, -1, 2, 0, 1, 0],
    ])

    >>> weights = sapevo_weighting(criteria_matrix)
    """
    # Calculation of the sum of degrees of preference in criteria comparison matrix in column vector
    sum_vector = np.sum(criteria_matrix, axis = 1)

    # Normalization of the column vector using Minimum-Maximum normalizations
    norm_vector = (sum_vector - np.min(sum_vector)) / (np.max(sum_vector) - np.min(sum_vector))
    return norm_vector / np.sum(norm_vector)