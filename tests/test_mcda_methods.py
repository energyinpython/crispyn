import unittest
import numpy as np

from crispyn.mcda_methods import VIKOR
from crispyn.additions import rank_preferences


# Test for VIKOR method
class Test_VIKOR(unittest.TestCase):

    def test_vikor(self):
        """Test based on paper Papathanasiou, J., & Ploskas, N. (2018). Vikor. In Multiple Criteria Decision Aid 
        (pp. 31-55). Springer, Cham. DOI: https://doi.org/10.1007/978-3-319-91648-4_2"""

        matrix = np.array([[8, 7, 2, 1],
        [5, 3, 7, 5],
        [7, 5, 6, 4],
        [9, 9, 7, 3],
        [11, 10, 3, 7],
        [6, 9, 5, 4]])

        weights = np.array([0.4, 0.3, 0.1, 0.2])

        types = np.array([1, 1, 1, 1])

        method = VIKOR(v = 0.625)
        test_result = method(matrix, weights, types)
        real_result = np.array([0.640, 1.000, 0.693, 0.271, 0.000, 0.694])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for rank preferences
class Test_Rank_preferences(unittest.TestCase):

    def test_rank_preferences(self):
        """Test based on paper Papathanasiou, J., & Ploskas, N. (2018). Vikor. In Multiple Criteria Decision Aid 
        (pp. 31-55). Springer, Cham. DOI: https://doi.org/10.1007/978-3-319-91648-4_2"""

        pref = np.array([0.640, 1.000, 0.693, 0.271, 0.000, 0.694])
        test_result = rank_preferences(pref , reverse = False)
        real_result = np.array([3, 6, 4, 2, 1, 5])
        self.assertEqual(list(test_result), list(real_result))


def main():
    test_vikor = Test_VIKOR()
    test_vikor.test_vikor()

    test_rank_preferences = Test_Rank_preferences()
    test_rank_preferences.test_rank_preferences()


if __name__ == '__main__':
    main()