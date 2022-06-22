import numpy as np

# Linear normalization
def linear_normalization(matrix, types):
    """
    Normalize decision matrix using linear normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = linear_normalization(matrix, types)
    """
    x_norm = np.zeros(np.shape(matrix))
    x_norm[:, types == 1] = matrix[:, types == 1] / (np.amax(matrix[:, types == 1], axis = 0))
    x_norm[:, types == -1] = np.amin(matrix[:, types == -1], axis = 0) / matrix[:, types == -1]
    return x_norm


# Mininum-Maximum normalization
def minmax_normalization(matrix, types):
    """
    Normalize decision matrix using minimum-maximum normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = minmax_normalization(matrix, types)
    """
    x_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    x_norm[:, types == 1] = (matrix[:, types == 1] - np.amin(matrix[:, types == 1], axis = 0)
                             ) / (np.amax(matrix[:, types == 1], axis = 0) - np.amin(matrix[:, types == 1], axis = 0))

    x_norm[:, types == -1] = (np.amax(matrix[:, types == -1], axis = 0) - matrix[:, types == -1]
                           ) / (np.amax(matrix[:, types == -1], axis = 0) - np.amin(matrix[:, types == -1], axis = 0))

    return x_norm


# Maximum normalization
def max_normalization(matrix, types):
    """
    Normalize decision matrix using maximum normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = max_normalization(matrix, types)
    """
    maximes = np.amax(matrix, axis = 0)
    matrix = matrix / maximes
    matrix[:, types == -1] = 1 - matrix[:, types == -1]
    return matrix


# Sum normalization
def sum_normalization(matrix, types):
    """
    Normalize decision matrix using sum normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = sum_normalization(matrix, types)
    """
    x_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    x_norm[:, types == 1] = matrix[:, types == 1] / np.sum(matrix[:, types == 1], axis = 0)
    x_norm[:, types == -1] = (1 / matrix[:, types == -1]) / np.sum((1 / matrix[:, types == -1]), axis = 0)

    return x_norm


# Vector normalization
def vector_normalization(matrix, types):
    """
    Normalize decision matrix using vector normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    -----------
    >>> nmatrix = vector_normalization(matrix, types)
    """
    x_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    x_norm[:, types == 1] = matrix[:, types == 1] / (np.sum(matrix[:, types == 1] ** 2, axis = 0))**(0.5)
    x_norm[:, types == -1] = 1 - (matrix[:, types == -1] / (np.sum(matrix[:, types == -1] ** 2, axis = 0))**(0.5))

    return x_norm
