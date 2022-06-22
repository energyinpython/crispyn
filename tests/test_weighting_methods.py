import unittest
import numpy as np
from crispyn import weighting_methods as mcda_weights
from crispyn import normalizations as norms


# Test for CRITIC weighting
class Test_CRITIC(unittest.TestCase):

    def test_critic(self):
        """Test based on paper Tuş, A., & Aytaç Adalı, E. (2019). The new combination with CRITIC and WASPAS methods 
        for the time and attendance software selection problem. Opsearch, 56(2), 528-538. 
        DOI: https://doi.org/10.1007/s12597-019-00371-6"""

        matrix = np.array([[5000, 3, 3, 4, 3, 2],
        [680, 5, 3, 2, 2, 1],
        [2000, 3, 2, 3, 4, 3],
        [600, 4, 3, 1, 2, 2],
        [800, 2, 4, 3, 3, 4]])

        types = np.array([-1, 1, 1, 1, 1, 1])

        test_result = mcda_weights.critic_weighting(matrix)
        real_result = np.array([0.157, 0.249, 0.168, 0.121, 0.154, 0.151])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for MEREC weighting
class Test_MEREC(unittest.TestCase):

    def test_merec(self):
        """Test based on paper Keshavarz-Ghorabaee, M., Amiri, M., Zavadskas, E. K., Turskis, Z., & Antucheviciene, 
        J. (2021). Determination of objective weights using a new method based on the removal 
        effects of criteria (MEREC). Symmetry, 13(4), 525. DOI: https://doi.org/10.3390/sym13040525"""

        matrix = np.array([[450, 8000, 54, 145],
        [10, 9100, 2, 160],
        [100, 8200, 31, 153],
        [220, 9300, 1, 162],
        [5, 8400, 23, 158]])

        types = np.array([1, 1, -1, -1])

        test_result = mcda_weights.merec_weighting(matrix, types)
        real_result = np.array([0.5752, 0.0141, 0.4016, 0.0091])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))


# Test for Entropy weighting
class Test_Entropy(unittest.TestCase):

    def test_Entropy(self):
        """Test based on paper Xu, X. (2004). A note on the subjective and objective integrated approach to 
        determine attribute weights. European Journal of Operational Research, 156(2), 
        530-532. DOI: https://doi.org/10.1016/S0377-2217(03)00146-2"""

        matrix = np.array([[30, 30, 38, 29],
        [19, 54, 86, 29],
        [19, 15, 85, 28.9],
        [68, 70, 60, 29]])

        types = np.array([1, 1, 1, 1])

        test_result = mcda_weights.entropy_weighting(matrix)
        real_result = np.array([0.4630, 0.3992, 0.1378, 0.0000])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))

    def test_Entropy2(self):
        """Test based on paper Zavadskas, E. K., & Podvezko, V. (2016). Integrated determination of objective 
        criteria weights in MCDM. International Journal of Information Technology & Decision 
        Making, 15(02), 267-283. DOI: https://EconPapers.repec.org/RePEc:wsi:ijitdm:v:15:y:2016:i:02:n:s0219622016500036"""

        matrix = np.array([[3.0, 100, 10, 7],
        [2.5, 80, 8, 5],
        [1.8, 50, 20, 11],
        [2.2, 70, 12, 9]])

        types = np.array([-1, 1, -1, 1])

        test_result = mcda_weights.entropy_weighting(matrix)
        real_result = np.array([0.1146, 0.1981, 0.4185, 0.2689])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))

    def test_Entropy3(self):
        """Test based on paper Ersoy, Y. (2021). Equipment selection for an e-commerce company using entropy-based 
        topsis, edas and codas methods during the COVID-19. LogForum, 17(3). DOI: http://doi.org/10.17270/J.LOG.2021.603"""

        matrix = np.array([[256, 8, 41, 1.6, 1.77, 7347.16],
        [256, 8, 32, 1.0, 1.8, 6919.99],
        [256, 8, 53, 1.6, 1.9, 8400],
        [256, 8, 41, 1.0, 1.75, 6808.9],
        [512, 8, 35, 1.6, 1.7, 8479.99],
        [256, 4, 35, 1.6, 1.7, 7499.99]])

        types = np.array([-1, 1, -1, 1])
        test_result = mcda_weights.entropy_weighting(matrix)

        real_result = np.array([0.405, 0.221, 0.134, 0.199, 0.007, 0.034])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))

    def test_Entropy4(self):
        """Test based on paper Lee, H. C., & Chang, C. T. (2018). Comparative analysis of MCDM 
        methods for ranking renewable energy sources in Taiwan. Renewable and Sustainable Energy 
        Reviews, 92, 883-896. DOI: https://doi.org/10.1016/j.rser.2018.05.007"""

        matrix = np.array([[4550, 30, 6.74, 20, 15, 5, 85, 150, 0.87, 4.76],
        [3005, 60.86, 2.4, 35, 27, 4, 26, 200, 0.17, 4.51],
        [2040, 14.85, 1.7, 90, 25, 5, 26, 500, 0.27, 4.19],
        [3370, 99.4, 3.25, 25.3, 54, 3, 45, 222, 0.21, 3.78],
        [3920, 112.6, 4.93, 11.4, 71.7, 2, 50, 100, 0.25, 4.11]])

        types = np.array([-1, -1, -1, 1, 1, 1, -1, -1, 1, 1])

        test_result = mcda_weights.entropy_weighting(matrix)
        real_result = np.array([0.026, 0.154, 0.089, 0.199, 0.115, 0.04, 0.08, 0.123, 0.172, 0.002])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))
        

# Test for CILOS weighting
class Test_CILOS(unittest.TestCase):

    def test_cilos(self):
        """Test based on paper Alinezhad, A., & Khalili, J. (2019). New methods and applications in multiple 
        attribute decision making (MADM) (Vol. 277). Cham: Springer. DOI: https://doi.org/10.1007/978-3-030-15009-9"""

        matrix = np.array([[3, 100, 10, 7],
        [2.500, 80, 8, 5],
        [1.800, 50, 20, 11],
        [2.200, 70, 12, 9]])

        types = np.array([-1, 1, -1, 1])

        test_result = mcda_weights.cilos_weighting(matrix, types)
        real_result = np.array([0.334, 0.220, 0.196, 0.250])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


    def test_cilos2(self):
        """Test based on paper Zavadskas, E. K., & Podvezko, V. (2016). Integrated determination of objective 
        criteria weights in MCDM. International Journal of Information Technology & Decision 
        Making, 15(02), 267-283. DOI: https://EconPapers.repec.org/RePEc:wsi:ijitdm:v:15:y:2016:i:02:n:s0219622016500036"""

        matrix = np.array([[0.6, 100, 0.8, 7],
        [0.72, 80, 1, 5],
        [1, 50, 0.4, 11],
        [0.818, 70, 0.667, 9]])

        types = np.array([1, 1, 1, 1])

        test_result = mcda_weights.cilos_weighting(matrix, types)
        real_result = np.array([0.3343, 0.2199, 0.1957, 0.2501])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))


# Test for IDOCRIW weighting
class Test_IDOCRIW(unittest.TestCase):

    def test_idocriw(self):
        """Test based on paper Zavadskas, E. K., & Podvezko, V. (2016). Integrated determination of objective 
        criteria weights in MCDM. International Journal of Information Technology & Decision 
        Making, 15(02), 267-283. DOI: https://EconPapers.repec.org/RePEc:wsi:ijitdm:v:15:y:2016:i:02:n:s0219622016500036"""

        matrix = np.array([[3.0, 100, 10, 7],
        [2.5, 80, 8, 5],
        [1.8, 50, 20, 11],
        [2.2, 70, 12, 9]])

        types = np.array([-1, 1, -1, 1])

        test_result = mcda_weights.idocriw_weighting(matrix, types)
        real_result = np.array([0.1658, 0.1886, 0.35455, 0.2911])
        self.assertEqual(list(np.round(test_result, 3)), list(np.round(real_result, 3)))


# Test for Angle weighting
class Test_Angle(unittest.TestCase):

    def test_angle(self):
        """Test based on paper Shuai, D., Zongzhun, Z., Yongji, W., & Lei, L. (2012, May). A new angular method to 
        determine the objective weights. In 2012 24th Chinese Control and Decision Conference 
        (CCDC) (pp. 3889-3892). IEEE. DOI: https://doi.org/10.1109/CCDC.2012.6244621"""

        matrix = np.array([[30, 30, 38, 29],
        [19, 54, 86, 29],
        [19, 15, 85, 28.9],
        [68, 70, 60, 29]])

        types = np.array([1, 1, 1, 1])

        test_result = mcda_weights.angle_weighting(matrix, types)
        real_result = np.array([0.4150, 0.3612, 0.2227, 0.0012])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))


# Test for Coefficient of Variation weighting
class Test_Coeff_var(unittest.TestCase):

    def test_coeff_var(self):
        """Test based on paper Shuai, D., Zongzhun, Z., Yongji, W., & Lei, L. (2012, May). A new angular method to 
        determine the objective weights. In 2012 24th Chinese Control and Decision Conference 
        (CCDC) (pp. 3889-3892). IEEE. DOI: https://doi.org/10.1109/CCDC.2012.6244621"""

        matrix = np.array([[30, 30, 38, 29],
        [19, 54, 86, 29],
        [19, 15, 85, 28.9],
        [68, 70, 60, 29]])
        
        types = np.array([1, 1, 1, 1])

        test_result = mcda_weights.coeff_var_weighting(matrix)
        real_result = np.array([0.4258, 0.3610, 0.2121, 0.0011])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))


# Test for Standard Deviation weighting
class Test_STD(unittest.TestCase):

    def test_std(self):
        """Test based on paper Sałabun, W., Wątróbski, J., & Shekhovtsov, A. (2020). Are mcda methods benchmarkable? 
        a comparative study of topsis, vikor, copras, and promethee ii methods. Symmetry, 12(9), 
        1549. DOI: https://doi.org/10.3390/sym12091549"""

        matrix = np.array([[0.619, 0.449, 0.447],
        [0.862, 0.466, 0.006],
        [0.458, 0.698, 0.771],
        [0.777, 0.631, 0.491],
        [0.567, 0.992, 0.968]])
        
        types = np.array([1, 1, 1])

        test_result = mcda_weights.std_weighting(matrix)
        real_result = np.array([0.217, 0.294, 0.488])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for Equal weighting
class Test_equal(unittest.TestCase):

    def test_equal(self):
        """Test based on paper Sałabun, W., Wątróbski, J., & Shekhovtsov, A. (2020). Are mcda methods benchmarkable? 
        a comparative study of topsis, vikor, copras, and promethee ii methods. Symmetry, 12(9), 
        1549. DOI: https://doi.org/10.3390/sym12091549"""

        matrix = np.array([[0.619, 0.449, 0.447],
        [0.862, 0.466, 0.006],
        [0.458, 0.698, 0.771],
        [0.777, 0.631, 0.491],
        [0.567, 0.992, 0.968]])
        
        types = np.array([1, 1, 1])

        test_result = mcda_weights.equal_weighting(matrix)
        real_result = np.array([0.333, 0.333, 0.333])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for Statistical variance weighting
class Test_stat_var(unittest.TestCase):

    def test_stat_var(self):
        """Test based on paper Sałabun, W., Wątróbski, J., & Shekhovtsov, A. (2020). Are mcda methods benchmarkable? 
        a comparative study of topsis, vikor, copras, and promethee ii methods. Symmetry, 12(9), 
        1549. DOI: https://doi.org/10.3390/sym12091549"""

        matrix = np.array([[0.619, 0.449, 0.447],
        [0.862, 0.466, 0.006],
        [0.458, 0.698, 0.771],
        [0.777, 0.631, 0.491],
        [0.567, 0.992, 0.968]])
        
        types = np.array([1, 1, 1])

        test_result = mcda_weights.stat_var_weighting(matrix)
        xn = norms.minmax_normalization(matrix, np.ones(matrix.shape[1]))
        real_result = np.var(xn, axis = 0) / np.sum(np.var(xn, axis = 0))
        self.assertEqual(list(np.round(test_result, 4)), list(np.round(real_result, 4)))


# Test for Gini coefficient-based weighting
class Test_gini(unittest.TestCase):

    def test_gini(self):
        """Test based on paper Bączkiewicz, A., Wątróbski, J., Kizielewicz, B., & Sałabun, W. (2021). Towards 
        Reliable Results-A Comparative Analysis of Selected MCDA Techniques in the Camera 
        Selection Problem. In Information Technology for Management: Business and Social Issues 
        (pp. 143-165). Springer, Cham. DOI: 10.1007/978-3-030-98997-2_7"""

        matrix = np.array([[29.4, 83, 47, 114, 12, 30, 120, 240, 170, 90, 1717.75],
        [30, 38.1, 124.7, 117, 16, 60, 60, 60, 93, 70, 2389],
        [29.28, 59.27, 41.13, 58, 16, 30, 60, 120, 170, 78, 239.99],
        [33.6, 71, 55, 159, 23.6, 60, 240, 240, 132, 140, 2099],
        [21, 59, 41, 66, 16, 24, 60, 120, 170, 70, 439],
        [35, 65, 42, 134, 12, 60, 240, 240, 145, 60, 1087],
        [47, 79, 54, 158, 19, 60, 120, 120, 360, 72, 2499],
        [28.3, 62.3, 44.9, 116, 12, 30, 60, 60, 130, 90, 999.99],
        [36.9, 28.6, 121.6, 130, 12, 60, 120, 120, 80, 80, 1099],
        [32, 59, 41, 60, 16, 30, 120, 120, 170, 60, 302.96],
        [28.4, 66.3, 48.6, 126, 12, 60, 240, 240, 132, 135, 1629],
        [29.8, 46, 113, 47, 18, 50, 50, 50, 360, 72, 2099],
        [20.2, 64, 80, 70, 8, 24, 60, 120, 166, 480, 699.99],
        [33, 60, 44, 59, 12, 30, 60, 120, 170, 90, 388],
        [29, 59, 41, 55, 16, 30, 60, 120, 170, 120, 299],
        [29, 59, 41, 182, 12, 30, 30, 60, 94, 140, 249],
        [29.8, 59.2, 41, 65, 16, 30, 60, 120, 160, 90, 219.99],
        [28.8, 62.5, 41, 70, 12, 60, 120, 120, 170, 138, 1399.99],
        [24, 40, 59, 60, 12, 10, 30, 30, 140, 78, 269.99],
        [30, 60, 45, 201, 16, 30, 30, 30, 170, 90, 199.99]])
        
        types = np.array([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1])

        test_result = mcda_weights.gini_weighting(matrix)
        
        real_result = np.array([0.0362, 0.0437, 0.0848, 0.0984, 0.0480, 0.0842, 0.1379, 0.1125, 0.0745, 0.1107, 0.1690])
        self.assertEqual(list(np.round(test_result, 4)), list(np.round(real_result, 4)))
        
        
        
def main():
    # Test of the CRITIC weighting method
    test_critic = Test_CRITIC()
    test_critic.test_critic()

    # Test of the MEREC weighting method
    test_merec = Test_MEREC()
    test_merec.test_merec()

    # Test of the Entropy weighting method
    test_entropy = Test_Entropy()
    test_entropy.test_Entropy()
    test_entropy.test_Entropy2()
    test_entropy.test_Entropy3()
    test_entropy.test_Entropy4()

    # Test of the CILOS weighting method
    test_cilos = Test_CILOS()
    test_cilos.test_cilos()
    test_cilos.test_cilos2()

    # Test of the IDOCRIW weighting method
    test_idocriw = Test_IDOCRIW()
    test_idocriw.test_idocriw()

    # Test of the Angle weighting method
    test_angle = Test_Angle()
    test_angle.test_angle()

    # Test of the Coefficient of variation weighting method
    test_coeff_var = Test_Coeff_var()
    test_coeff_var.test_coeff_var()

    # Test of the Standard deviation weighting method
    test_std = Test_STD()
    test_std.test_std()

    # Test of the Equal weighting method
    test_equal = Test_equal()
    test_equal.test_equal()

    # Test of the Statistical variance weighting method
    test_stat_var = Test_stat_var()
    test_stat_var.test_stat_var()
    
    # Test of the Gini coefficient-based weighting method
    test_gini = Test_gini()
    test_gini.test_gini()

if __name__ == '__main__':
    main()

