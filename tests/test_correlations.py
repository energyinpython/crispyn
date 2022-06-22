import unittest
import numpy as np
from scipy.stats import pearsonr

from crispyn import correlations as corrs

# Test for Spearman rank correlation coefficient
class Test_Spearman(unittest.TestCase):

    def test_spearman(self):
        """Test based on paper Sałabun, W., & Urbaniak, K. (2020, June). A new coefficient of rankings similarity 
        in decision-making problems. In International Conference on Computational Science 
        (pp. 632-645). Springer, Cham. DOI: https://doi.org/10.1007/978-3-030-50417-5_47"""

        R = np.array([1, 2, 3, 4, 5])
        Q = np.array([1, 3, 2, 4, 5])
        test_result = corrs.spearman(R, Q)
        real_result = 0.9
        self.assertEqual(test_result, real_result)


# Test for Weighted Spearman rank correlation coefficient
class Test_Weighted_Spearman(unittest.TestCase):

    def test_weighted_spearman(self):
        """Test based on paper Sałabun, W., & Urbaniak, K. (2020, June). A new coefficient of rankings similarity 
        in decision-making problems. In International Conference on Computational Science 
        (pp. 632-645). Springer, Cham. DOI: https://doi.org/10.1007/978-3-030-50417-5_47"""

        R = np.array([1, 2, 3, 4, 5])
        Q = np.array([1, 3, 2, 4, 5])
        test_result = corrs.weighted_spearman(R, Q)
        real_result = 0.8833
        self.assertEqual(np.round(test_result, 4), real_result)


# Test for Pearson correlation coefficient
class Test_Pearson(unittest.TestCase):

    def test_pearson(self):
        """Test based on paper Sałabun, W., & Urbaniak, K. (2020, June). A new coefficient of rankings similarity 
        in decision-making problems. In International Conference on Computational Science 
        (pp. 632-645). Springer, Cham. DOI: https://doi.org/10.1007/978-3-030-50417-5_47"""

        R = np.array([1, 2, 3, 4, 5])
        Q = np.array([1, 3, 2, 4, 5])
        test_result = corrs.pearson_coeff(R, Q)
        real_result, _ = pearsonr(R, Q)
        self.assertEqual(test_result, real_result)


def main():
    test_spearman_coeff = Test_Spearman()
    test_spearman_coeff.test_spearman()

    test_weighted_spearman_coeff = Test_Weighted_Spearman()
    test_weighted_spearman_coeff.test_weighted_spearman()

    test_pearson_coeff = Test_Pearson()
    test_pearson_coeff.test_pearson()


if __name__ == '__main__':
    main()