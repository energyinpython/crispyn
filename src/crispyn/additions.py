import numpy as np

# reverse = True: descending order ( for example TOPSIS, CODAS), False: ascending order (for example VIKOR, SPOTIS)
def rank_preferences(pref, reverse = True):
    """
    Rank alternatives according to MCDA preference function values. If more than one alternative
    have the same preference function value, they will be given the same rank value (tie).

    Parameters
    ------------
        pref : ndarray
            Vector with MCDA preference function values for alternatives
        reverse : bool
            The boolean variable is True for MCDA methods that rank alternatives in descending
            order (for example, TOPSIS, CODAS) and False for MCDA methods that rank alternatives in ascending
            order (for example, VIKOR, SPOTIS)
    
    Returns
    ---------
        ndarray
            Vector with alternatives ranking. Alternative with 1 value is the best and has the first position in the ranking.
    
    Examples
    ----------
    >>> rank = rank_preferences(pref, reverse = True)
    """

    # Create an array ndarray for the ranking values of alternatives
    rank = np.zeros(len(pref))
    # Generate sorted vector with MCDA preference function values
    # sorting order is determined by variable `reverse` and depends on MCDA method
    sorted_pref = sorted(pref, reverse = reverse)
    # position of the best alternative in ranking is denoted by 1, so assign 1 to `pos`
    pos = 1
    for i in range(len(sorted_pref) - 1):
        # find index in vector with preference values `pref` equal to preference value in sorted vector `sorted_pref`
        ind = np.where(sorted_pref[i] == pref)[0]
        # assign rank denoted by `pos` to index `ind` in rank vector
        rank[ind] = pos
        # if the next preference value in sorted vector is higher than actual, increase `pos` which denotes rank
        # if the next preference value is equal to actual, `pos` will be unchanged
        if sorted_pref[i] != sorted_pref[i + 1]:
            pos += 1
    # find index with the last preference value
    ind = np.where(sorted_pref[i + 1] == pref)[0]
    # assign the last place `pos` to alternative with the last preference value
    rank[ind] = pos
    return rank.astype(int)