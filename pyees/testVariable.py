import unittest
import numpy as np
from random import uniform
try:
    from .variable import variable
except ImportError:
    from variable import variable

class test(unittest.TestCase): 

    def testSingleNumber(self):
        A = variable(1.3, 'm')
        B = variable(2.0, 'm', 0.01)
        C = variable([1.0, 1.3], 'L/min', np.array([20, 30]))
        D = variable(np.array([11, 1111]), 'L/min', [2.1, 3.9])
        self.assertEqual(A.value, 1.3)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 0)
        self.assertEqual(B.value, 2.0)
        self.assertEqual(B.unit, 'm')
        self.assertEqual(B.uncert, 0.01)
        np.testing.assert_equal(C.value, [1.0, 1.3])
        self.assertEqual(C.unit, 'L/min')
        np.testing.assert_equal(C.uncert, [20, 30])
        np.testing.assert_equal(D.value, [11.0, 1111.0])
        self.assertEqual(D.unit, 'L/min')
        np.testing.assert_equal(D.uncert, [2.1, 3.9])

        with self.assertRaises(Exception) as context:
            variable(1.3, 'm', 'hej')
        self.assertTrue("could not convert string to float: 'hej'" in str(context.exception))

        with self.assertRaises(Exception) as context:
            variable('med', 'm', 1.0)
        self.assertTrue("could not convert string to float: 'med'" in str(context.exception))

        with self.assertRaises(Exception) as context:
            variable(1.3, 'm', [1.0, 2.3])
        self.assertTrue("The lenght of the value has to be equal to the lenght of the uncertanty" in str(context.exception))

        with self.assertRaises(Exception) as context:
            variable(1.3, 'm', np.array([1.0, 2.3]))
        self.assertTrue("The lenght of the value has to be equal to the lenght of the uncertanty" in str(context.exception))

        with self.assertRaises(Exception) as context:
            variable(np.array([1.0, 2.3]), 'm', 1.5)
        self.assertTrue("The lenght of the value has to be equal to the lenght of the uncertanty" in str(context.exception))

        with self.assertRaises(Exception) as context:
            variable([1.0, 2.3], 'm', 1.5)
        self.assertTrue("The lenght of the value has to be equal to the lenght of the uncertanty" in str(context.exception))

    def test_add(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'L/min', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'L/min', uncert=[53.9, 24.75, 6.4])

        C = A + B
        self.assertAlmostEqual(C.value, 12.3 + 745.1)
        self.assertEqual(C.unit, 'L/min')
        self.assertAlmostEqual(C.uncert, np.sqrt((1 * 2.6)**2 + (1 * 53.9)**2))

        C.convert('m3/s')
        self.assertAlmostEqual(C.value, (12.3 + 745.1) / 1000 / 60)
        self.assertEqual(C.unit, 'm3/s')
        self.assertAlmostEqual(C.uncert, np.sqrt((1 * 2.6 / 1000 / 60)**2 + (1 * 53.9 / 1000 / 60)**2))

        C_vec = A_vec + B_vec
        np.testing.assert_almost_equal(C_vec.value, np.array([12.3 + 745.1, 54.3 + 496.13, 91.3 + 120.54]))
        self.assertEqual(C_vec.unit, 'L/min')
        np.testing.assert_almost_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((1 * 2.6)**2 + (1 * 53.9)**2),
                np.sqrt((1 * 5.4)**2 + (1 * 24.75)**2),
                np.sqrt((1 * 10.56)**2 + (1 * 6.4)**2),
            ]))

        C_vec.convert('mL/h')
        np.testing.assert_almost_equal(C_vec.value, np.array([(12.3 + 745.1) * 1000 * 60, (54.3 + 496.13) * 1000 * 60, (91.3 + 120.54) * 1000 * 60]))
        self.assertEqual(C_vec.unit, 'mL/h')
        np.testing.assert_almost_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((1 * 2.6 * 1000 * 60)**2 + (1 * 53.9 * 1000 * 60)**2),
                np.sqrt((1 * 5.4 * 1000 * 60)**2 + (1 * 24.75 * 1000 * 60)**2),
                np.sqrt((1 * 10.56 * 1000 * 60)**2 + (1 * 6.4 * 1000 * 60)**2),
            ]))

    def test_sub(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'L/min', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'L/min', uncert=[53.9, 24.75, 6.4])

        C = A - B
        self.assertAlmostEqual(C.value, 12.3 - 745.1)
        self.assertEqual(C.unit, 'L/min')
        self.assertAlmostEqual(C.uncert, np.sqrt((1 * 2.6)**2 + (1 * 53.9)**2))

        C.convert('kL/s')
        self.assertAlmostEqual(C.value, (12.3 - 745.1) / 1000 / 60)
        self.assertEqual(C.unit, 'kL/s')
        self.assertAlmostEqual(C.uncert, np.sqrt((1 * 2.6 / 1000 / 60)**2 + (1 * 53.9 / 1000 / 60)**2))

        C_vec = A_vec - B_vec
        np.testing.assert_almost_equal(C_vec.value, np.array([12.3 - 745.1, 54.3 - 496.13, 91.3 - 120.54]))
        self.assertEqual(C_vec.unit, 'L/min')
        np.testing.assert_almost_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((1 * 2.6)**2 + (1 * 53.9)**2),
                np.sqrt((1 * 5.4)**2 + (1 * 24.75)**2),
                np.sqrt((1 * 10.56)**2 + (1 * 6.4)**2),
            ]))

    def test_add_with_different_units(self):
        A = variable(12.3, 'm3/s', uncert=2.6)
        B = variable(745.1, 'L/min', uncert=53.9)
        C = A + B
        self.assertAlmostEqual(C.value, 12.3 + 745.1 / 1000 / 60)
        self.assertEqual(C.unit, 'm3/s')
        self.assertAlmostEqual(C.uncert, np.sqrt(2.6**2 + (53.9 / 1000 / 60)**2))

        A = variable(12.3, 'C', uncert=2.6)
        B = variable(745.1, 'K', uncert=53.9)
        C = A + B
        self.assertAlmostEqual(C.value, 12.3 + 273.15 + 745.1)
        self.assertEqual(C.unit, 'K')
        self.assertAlmostEqual(C.uncert, np.sqrt(2.6**2 + 53.9**2))

        A = variable(12.3, 'C', uncert=2.6)
        B = variable(745.1, 'DELTAK', uncert=53.9)
        C = A + B
        self.assertAlmostEqual(C.value, 12.3 + 745.1)
        self.assertEqual(C.unit, 'C')
        self.assertAlmostEqual(C.uncert, np.sqrt(2.6**2 + 53.9**2))

        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm', uncert=[53.9, 24.75, 6.4])

        with self.assertRaises(Exception) as context:
            A + B
        self.assertTrue('You tried to add a variable in [L/min] to a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec + B_vec
        self.assertTrue('You tried to add a variable in [L/min] to a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

    def test_sub_with_different_units(self):
        A = variable(12.3, 'm3/s', uncert=2.6)
        B = variable(745.1, 'L/min', uncert=53.9)
        C = A - B
        self.assertAlmostEqual(C.value, 12.3 - 745.1 / 1000 / 60)
        self.assertEqual(C.unit, 'm3/s')
        self.assertAlmostEqual(C.uncert, np.sqrt(2.6**2 + (53.9 / 1000 / 60)**2))

        A = variable(12.3, 'C', uncert=2.6)
        B = variable(745.1, 'K', uncert=53.9)
        C = A - B
        self.assertAlmostEqual(C.value, 12.3 + 273.15 - 745.1)
        self.assertEqual(C.unit, 'K')
        self.assertAlmostEqual(C.uncert, np.sqrt(2.6**2 + 53.9**2))

        A = variable(12.3, 'C', uncert=2.6)
        B = variable(745.1, 'DELTAK', uncert=53.9)
        C = A - B
        self.assertAlmostEqual(C.value, 12.3 - 745.1)
        self.assertEqual(C.unit, 'C')
        self.assertAlmostEqual(C.uncert, np.sqrt(2.6**2 + 53.9**2))
        
        A = variable(12.3, 'C', uncert=2.6)
        B = variable(745.1, 'C', uncert=53.9)
        C = A - B
        self.assertAlmostEqual(C.value, 12.3 - 745.1)
        self.assertEqual(C.unit, 'DELTAC')
        self.assertAlmostEqual(C.uncert, np.sqrt(2.6**2 + 53.9**2))


        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm', uncert=[53.9, 24.75, 6.4])

        with self.assertRaises(Exception) as context:
            A - B
        self.assertTrue('You tried to subtract a variable in [m] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec - B_vec
        self.assertTrue('You tried to subtract a variable in [m] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

    def test_multiply(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm', uncert=[53.9, 24.75, 6.4])

        C = A * B

        self.assertAlmostEqual(C.value, 12.3 * 745.1)
        self.assertTrue(C._unitObject._assertEqual('L-m/min'))
        self.assertAlmostEqual(C.uncert, np.sqrt((745.1 * 2.6)**2 + (12.3 * 53.9)**2))

        C_vec = A_vec * B_vec
        np.testing.assert_array_equal(C_vec.value, np.array([12.3 * 745.1, 54.3 * 496.13, 91.3 * 120.54]))
        self.assertTrue(C._unitObject._assertEqual('L-m/min'))
        np.testing.assert_array_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((745.1 * 2.6)**2 + (12.3 * 53.9)**2),
                np.sqrt((496.13 * 5.4)**2 + (54.3 * 24.75)**2),
                np.sqrt((120.54 * 10.56)**2 + (91.3 * 6.4)**2),
            ]))

        C_vec.convert('m3-km / s')
        np.testing.assert_array_equal(C_vec.value, np.array([12.3 * 745.1, 54.3 * 496.13, 91.3 * 120.54]) / 1000 / 1000 / 60)
        self.assertEqual(C_vec.unit, 'm3-km/s')
        np.testing.assert_almost_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((745.1 / 1000 * 2.6 / 1000 / 60)**2 + (12.3 / 1000 / 60 * 53.9 / 1000)**2),
                np.sqrt((496.13 / 1000 * 5.4 / 1000 / 60)**2 + (54.3 / 1000 / 60 * 24.75 / 1000)**2),
                np.sqrt((120.54 / 1000 * 10.56 / 1000 / 60)**2 + (91.3 / 1000 / 60 * 6.4 / 1000)**2),
            ]), decimal=7)

        a = variable(1.2, 'm/N', 0.15)
        b = variable(7.43, 'N/cm', 2.5)
        c = a * b
        self.assertAlmostEqual(c.value, 891.6)
        self.assertEqual(c.unit, '1')
        self.assertAlmostEqual(c.uncert, 320.032970958)

    def test_divide(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm', uncert=[53.9, 24.75, 6.4])

        C = A / B
        self.assertAlmostEqual(C.value, 12.3 / 745.1)
        self.assertTrue(C._unitObject._assertEqual('L/min-m'))
        self.assertAlmostEqual(C.uncert, np.sqrt((1 / 745.1 * 2.6)**2 + (12.3 / (745.1**2) * 53.9)**2))

        C.convert('m3/h-mm')
        self.assertAlmostEqual(C.value, 12.3 / 745.1 / 1000 * 60 / 1000)
        self.assertEqual(C.unit, 'm3/h-mm')
        self.assertAlmostEqual(C.uncert, np.sqrt((1 / (745.1 * 1000) * 2.6 / 1000 * 60)**2 + (12.3 / ((745.1)**2) * 53.9 / 1000 * 60 / 1000)**2))

        C_vec = A_vec / B_vec
        np.testing.assert_array_equal(C_vec.value, np.array([12.3 / 745.1, 54.3 / 496.13, 91.3 / 120.54]))
        self.assertTrue(C_vec._unitObject._assertEqual('L/min-m'))
        np.testing.assert_array_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((1 / 745.1 * 2.6)**2 + (12.3 / (745.1)**2 * 53.9)**2),
                np.sqrt((1 / 496.13 * 5.4)**2 + (54.3 / (496.13)**2 * 24.75)**2),
                np.sqrt((1 / 120.54 * 10.56)**2 + (91.3 / (120.54)**2 * 6.4)**2),
            ]))

        C_vec.convert('m3 / h -mm')
        np.testing.assert_almost_equal(C_vec.value, np.array([12.3 / 745.1, 54.3 / 496.13, 91.3 / 120.54]) / 1000 * 60 / 1000)
        self.assertEqual(C_vec.unit, 'm3/h-mm')
        np.testing.assert_almost_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((1 / 745.1 * 2.6 / 1000 * 60 / 1000)**2 + (12.3 / (745.1)**2 * 53.9 / 1000 * 60 / 1000)**2),
                np.sqrt((1 / 496.13 * 5.4 / 1000 * 60 / 1000)**2 + (54.3 / (496.13)**2 * 24.75 / 1000 * 60 / 1000)**2),
                np.sqrt((1 / 120.54 * 10.56 / 1000 * 60 / 1000)**2 + (91.3 / (120.54)**2 * 6.4 / 1000 * 60 / 1000)**2),
            ]))

        a = variable(1.2, 'm/N', 0.15)
        b = variable(7.43, 'cm/N', 2.5)
        c = a / b
        self.assertAlmostEqual(c.value, 16.1507402423)
        self.assertEqual(c.unit, '1')
        self.assertAlmostEqual(c.uncert, 5.79718414412)

    def test_add_unit_order(self):
        A = variable(10, 'm-K')
        B = variable(3, 'K-m')
        A_vec = variable([12.3, 54.3, 91.3], 'K-m', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm-K', uncert=[53.9, 24.75, 6.4])
        C = A + B
        C_vec = A_vec + B_vec

    def test_sub_unit_order(self):
        A = variable(10, 'm-K')
        B = variable(3, 'K-m')
        A_vec = variable([12.3, 54.3, 91.3], 'K-m', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm-K', uncert=[53.9, 24.75, 6.4])
        C = A - B
        C_vec = A_vec - B_vec

    def test_pow(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)
        C = variable(745.1, '1', uncert=53.9)
        D = variable(0.34, '1', uncert=0.01)

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm', uncert=[53.9, 24.75, 6.4])
        C_vec = variable([745.1, 496.13, 120.54], '1', uncert=[53.9, 24.75, 6.4])
        D_vec = variable([0.34, 0.64, 0.87], '1', uncert=[0.01, 0.084, 0.12])

        with self.assertRaises(Exception) as context:
            A ** B
        self.assertTrue('The exponent can not have a unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec ** B_vec
        self.assertTrue('The exponent can not have a unit' in str(context.exception))

        E = C**D
        self.assertAlmostEqual(E.value, 745.1**0.34)
        self.assertEqual(E.unit, '1')
        self.assertAlmostEqual(E.uncert, np.sqrt((0.34 * 745.1**(0.34 - 1) * 53.9)**2 + (745.1**0.34 * np.log(745.1) * 0.01)**2))

        E_vec = C_vec**D_vec
        np.testing.assert_equal(E_vec.value, [745.1 ** 0.34, 496.13**0.64, 120.54**0.87])
        self.assertEqual(E_vec.unit, '1')
        self.assertAlmostEqual(E_vec.uncert[0], np.sqrt((0.34 * 745.1**(0.34 - 1) * 53.9)**2 + (745.1**0.34 * np.log(745.1) * 0.01)**2))

        
        F = A**2
        self.assertAlmostEqual(F.value, (12.3)**2)
        self.assertEqual(F.unit, 'L2/min2')
        self.assertAlmostEqual(F.uncert, np.sqrt((2 * 12.3**(2 - 1) * 2.6)**2))

        F.convert('m6/s2')
        self.assertAlmostEqual(F.value, (12.3 / 1000 / 60)**2)
        self.assertEqual(F.unit, 'm6/s2')
        self.assertAlmostEqual(F.uncert, np.sqrt((2 * (12.3 / 1000 / 60)**(2 - 1) * 2.6 / 1000 / 60)**2))

        F_vec = A_vec**2
        np.testing.assert_array_almost_equal(F_vec.value, np.array([(12.3)**2, 54.3**2, 91.3**2]))
        self.assertEqual(F_vec.unit, 'L2/min2')
        np.testing.assert_array_almost_equal(
            F_vec.uncert,
            np.array([
                np.sqrt((2 * 12.3**(2 - 1) * 2.6)**2),
                np.sqrt((2 * 54.3**(2 - 1) * 5.4)**2),
                np.sqrt((2 * 91.3**(2 - 1) * 10.56)**2)
            ]))

        F_vec.convert('m6 / s2')
        np.testing.assert_array_almost_equal(F_vec.value, np.array([(12.3 / 1000 / 60)**2, (54.3 / 1000 / 60)**2, (91.3 / 1000 / 60)**2]))
        self.assertEqual(F_vec.unit, 'm6/s2')
        np.testing.assert_array_almost_equal(
            F_vec.uncert,
            np.array([
                np.sqrt((2 * 12.3 / 1000 / 60**(2 - 1) * 2.6 / 1000 / 60)**2),
                np.sqrt((2 * 54.3 / 1000 / 60**(2 - 1) * 5.4 / 1000 / 60)**2),
                np.sqrt((2 * 91.3 / 1000 / 60**(2 - 1) * 10.56 / 1000 / 60)**2)
            ]))

        G = 2.54**D
        self.assertAlmostEqual(G.value, 2.54**0.34)
        self.assertEqual(G.unit, '1')
        self.assertAlmostEqual(G.uncert, np.sqrt((2.54**0.34 * np.log(2.54) * 0.01)**2))

        G_vec = 2.54**D_vec
        np.testing.assert_equal(G_vec.value, [2.54**0.34, 2.54 **0.64 , 2.54**0.87])
        self.assertEqual(G_vec.unit, '1')
        self.assertAlmostEqual(G_vec.uncert[0], np.sqrt((2.54**0.34 * np.log(2.54) * 0.01)**2))

        
        
    def test_log(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        C = variable(745.1, '1', uncert=53.9)

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        C_vec = variable([745.1, 496.13, 120.54], '1', uncert=[53.9, 24.75, 6.4])

        with self.assertRaises(Exception) as context:
            np.log(A)
        self.assertTrue('You can only take the natural log of a variable if it has no unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            np.log10(A)
        self.assertTrue('You can only take the base 10 log of a variable if it has no unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            np.log(A_vec)
        self.assertTrue('You can only take the natural log of a variable if it has no unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            np.log10(A_vec)
        self.assertTrue('You can only take the base 10 log of a variable if it has no unit' in str(context.exception))

        D = np.log(C)
        self.assertAlmostEqual(D.value, np.log(745.1))
        self.assertEqual(D.unit, '1')
        self.assertAlmostEqual(D.uncert, np.sqrt((1 / 745.1) * 53.9)**2)

        D_vec = np.log(C_vec)
        np.testing.assert_array_equal(D_vec.value, np.array([np.log(745.1), np.log(496.13), np.log(120.54)]))
        self.assertEqual(D_vec.unit, '1')
        np.testing.assert_array_equal(
            D_vec.uncert,
            np.array([
                np.sqrt(((1 / 745.1) * 53.9)**2),
                np.sqrt(((1 / 496.13) * 24.75)**2),
                np.sqrt(((1 / 120.54) * 6.4)**2)
            ]))

        E = np.log10(C)
        self.assertAlmostEqual(E.value, np.log10(745.1))
        self.assertEqual(E.unit, '1')
        self.assertAlmostEqual(E.uncert, np.sqrt((1 / (745.1 * np.log10(745.1))) * 53.9)**2)

        E_vec = np.log10(C_vec)
        np.testing.assert_array_equal(E_vec.value, np.array([np.log10(745.1), np.log10(496.13), np.log10(120.54)]))
        self.assertEqual(E_vec.unit, '1')
        np.testing.assert_array_equal(
            E_vec.uncert,
            np.array([
                np.sqrt(((1 / (745.1 * np.log10(745.1))) * 53.9)**2),
                np.sqrt(((1 / (496.13 * np.log10(496.13))) * 24.75)**2),
                np.sqrt(((1 / (120.54 * np.log10(120.54))) * 6.4)**2)
            ]))

    def test_exp(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        C = variable(12.3, '1', uncert=5.39)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        C_vec = variable([12.3, 54.3, 91.3], '1', uncert=[2.6, 5.4, 10.56])

        with self.assertRaises(Exception) as context:
            np.exp(A)
        self.assertTrue('The exponent can not have a unit' in str(context.exception))

        c_vec = np.exp(C_vec)
        np.testing.assert_equal(c_vec.value, [np.e**12.3, np.e**54.3, np.e**91.3])
        self.assertEqual(c_vec.unit, '1')
        self.assertEqual(c_vec.uncert[0], np.sqrt((np.e**12.3 * np.log(np.e) * 2.6)**2))
        
        D = np.exp(C)
        self.assertAlmostEqual(D.value, np.e**12.3)
        self.assertEqual(D.unit, '1')
        self.assertAlmostEqual(D.uncert, np.sqrt((np.e**12.3 * np.log(np.e) * 5.39)**2))

        with self.assertRaises(Exception) as context:
            np.exp(A_vec)
        self.assertTrue('The exponent can not have a unit' in str(context.exception))

    def testIndex(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])

        with self.assertRaises(Exception) as context:
            a = A[0]
        self.assertTrue("'scalarVariable' object is not subscriptable" in str(context.exception))
        
        a_vec = A_vec[0, 1]
        np.testing.assert_equal(a_vec.value, [12.3, 54.3])
        self.assertEqual(a_vec.unit, 'L/min')
        np.testing.assert_equal(a_vec.uncert, [2.6, 5.4])

        a_vec = A_vec[0, 2]
        np.testing.assert_equal(a_vec.value, [12.3, 91.3])
        self.assertEqual(a_vec.unit, 'L/min')
        np.testing.assert_equal(a_vec.uncert, [2.6, 10.56])

        a_vec = A_vec[2, 0]
        np.testing.assert_equal(a_vec.value, [91.3, 12.3])
        self.assertEqual(a_vec.unit, 'L/min')
        np.testing.assert_equal(a_vec.uncert, [10.56, 2.6])

        with self.assertRaises(Exception) as context:
            a = A[1]
        self.assertTrue("'scalarVariable' object is not subscriptable" in str(context.exception))

        with self.assertRaises(Exception) as context:
            a = A_vec[23]
        self.assertTrue('Index out of bounds' in str(context.exception))

    def testAddEqual(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'L/min', uncert=53.9)

        A += B
        self.assertAlmostEqual(A.value, 12.3 + 745.1)
        self.assertEqual(A.unit, 'L/min')
        self.assertAlmostEqual(A.uncert, np.sqrt((1 * 2.6)**2 + (1 * 53.9)**2))
        A = variable(12.3, 'L/min', uncert=2.6)

        A += 2
        self.assertAlmostEqual(A.value, 12.3 + 2)
        self.assertEqual(A.unit, 'L/min')
        self.assertAlmostEqual(A.uncert, np.sqrt((1 * 2.6)**2))

        A = variable(12.3, 'L/min', uncert=2.6)
        B = 2
        B += A
        self.assertAlmostEqual(B.value, 2 + 12.3)
        self.assertEqual(B.unit, 'L/min')
        self.assertAlmostEqual(B.uncert, np.sqrt((1 * 2.6)**2))

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'L/min', uncert=[53.9, 24.75, 6.4])

        A_vec += B_vec
        np.testing.assert_almost_equal(A_vec.value, np.array([12.3 + 745.1, 54.3 + 496.13, 91.3 + 120.54]))
        self.assertEqual(A_vec.unit, 'L/min')
        np.testing.assert_almost_equal(
            A_vec.uncert,
            np.array([
                np.sqrt((1 * 2.6)**2 + (1 * 53.9)**2),
                np.sqrt((1 * 5.4)**2 + (1 * 24.75)**2),
                np.sqrt((1 * 10.56)**2 + (1 * 6.4)**2),
            ]))

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        A = variable(12.3, 'L/min', uncert=2.6)
        A_vec += A
        np.testing.assert_almost_equal(A_vec.value, np.array([12.3 + 12.3, 54.3 + 12.3, 91.3 + 12.3]))
        self.assertEqual(A_vec.unit, 'L/min')
        np.testing.assert_almost_equal(
            A_vec.uncert,
            np.array([
                np.sqrt((1 * 2.6)**2 + (1 * 2.6)**2),
                np.sqrt((1 * 5.4)**2 + (1 * 2.6)**2),
                np.sqrt((1 * 10.56)**2 + (1 * 2.6)**2),
            ]))

        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm', uncert=[53.9, 24.75, 6.4])

        with self.assertRaises(Exception) as context:
            A += B
        self.assertTrue('You tried to add a variable in [L/min] to a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B += A
        self.assertTrue('You tried to add a variable in [m] to a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec += B_vec
        self.assertTrue('You tried to add a variable in [L/min] to a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B_vec += A_vec
        self.assertTrue('You tried to add a variable in [m] to a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec += B
        self.assertTrue('You tried to add a variable in [L/min] to a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B_vec += A
        self.assertTrue('You tried to add a variable in [m] to a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A += B_vec
        self.assertTrue('You tried to add a variable in [L/min] to a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B += A_vec
        self.assertTrue('You tried to add a variable in [m] to a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

    def testSubEqual(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'L/min', uncert=53.9)

        A -= B
        self.assertAlmostEqual(A.value, 12.3 - 745.1)
        self.assertEqual(A.unit, 'L/min')
        self.assertAlmostEqual(A.uncert, np.sqrt((1 * 2.6)**2 + (1 * 53.9)**2))
        A = variable(12.3, 'L/min', uncert=2.6)

        A -= 2
        self.assertAlmostEqual(A.value, 12.3 - 2)
        self.assertEqual(A.unit, 'L/min')
        self.assertAlmostEqual(A.uncert, np.sqrt((1 * 2.6)**2))

        A = variable(12.3, 'L/min', uncert=2.6)
        B = 2
        B -= A
        self.assertAlmostEqual(B.value, 2 - 12.3)
        self.assertEqual(B.unit, 'L/min')
        self.assertAlmostEqual(B.uncert, np.sqrt((1 * 2.6)**2))

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'L/min', uncert=[53.9, 24.75, 6.4])

        A_vec -= B_vec
        np.testing.assert_almost_equal(A_vec.value, np.array([12.3 - 745.1, 54.3 - 496.13, 91.3 - 120.54]))
        self.assertEqual(A_vec.unit, 'L/min')
        np.testing.assert_almost_equal(
            A_vec.uncert,
            np.array([
                np.sqrt((1 * 2.6)**2 + (1 * 53.9)**2),
                np.sqrt((1 * 5.4)**2 + (1 * 24.75)**2),
                np.sqrt((1 * 10.56)**2 + (1 * 6.4)**2),
            ]))

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        A = variable(12.3, 'L/min', uncert=2.6)
        A_vec -= A
        np.testing.assert_almost_equal(A_vec.value, np.array([12.3 - 12.3, 54.3 - 12.3, 91.3 - 12.3]))
        self.assertEqual(A_vec.unit, 'L/min')
        np.testing.assert_almost_equal(
            A_vec.uncert,
            np.array([
                np.sqrt((1 * 2.6)**2 + (1 * 2.6)**2),
                np.sqrt((1 * 5.4)**2 + (1 * 2.6)**2),
                np.sqrt((1 * 10.56)**2 + (1 * 2.6)**2),
            ]))

        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm', uncert=[53.9, 24.75, 6.4])

        with self.assertRaises(Exception) as context:
            A -= B
        self.assertTrue('You tried to subtract a variable in [m] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B -= A
        self.assertTrue('You tried to subtract a variable in [L/min] from a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec -= B_vec
        self.assertTrue('You tried to subtract a variable in [m] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B_vec -= A_vec
        self.assertTrue('You tried to subtract a variable in [L/min] from a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec -= B
        self.assertTrue('You tried to subtract a variable in [m] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B_vec -= A
        self.assertTrue('You tried to subtract a variable in [L/min] from a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A -= B_vec
        self.assertTrue('You tried to subtract a variable in [m] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B -= A_vec
        self.assertTrue('You tried to subtract a variable in [L/min] from a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

    def testMultiEqual(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)

        A *= B
        self.assertAlmostEqual(A.value, 12.3 * 745.1)
        self.assertTrue(A._unitObject._assertEqual('L-m/min'))
        self.assertAlmostEqual(A.uncert, np.sqrt((745.1 * 2.6)**2 + (12.3 * 53.9)**2))

        A = variable(12.3, 'L/min', uncert=2.6)
        A *= 2
        self.assertAlmostEqual(A.value, 12.3 * 2)
        self.assertEqual(A.unit, 'L/min')
        self.assertAlmostEqual(A.uncert, np.sqrt((2 * 2.6)**2))

        A = variable(12.3, 'L/min', uncert=2.6)
        B = 2
        B *= A
        self.assertAlmostEqual(B.value, 12.3 * 2)
        self.assertEqual(B.unit, 'L/min')
        self.assertAlmostEqual(B.uncert, np.sqrt((2 * 2.6)**2))

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm', uncert=[53.9, 24.75, 6.4])

        A_vec *= B_vec
        np.testing.assert_array_almost_equal(A_vec.value, np.array([12.3 * 745.1, 54.3 * 496.13, 91.3 * 120.54]))
        self.assertTrue(A_vec._unitObject._assertEqual('L-m/min'))
        np.testing.assert_array_almost_equal(
            A_vec.uncert,
            np.array([
                np.sqrt((745.1 * 2.6)**2 + (12.3 * 53.9)**2),
                np.sqrt((496.13 * 5.4)**2 + (54.3 * 24.75)**2),
                np.sqrt((120.54 * 10.56)**2 + (91.3 * 6.4)**2),
            ]))

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        A = variable(12.3, 'L/min', uncert=2.6)
        A_vec *= A
        np.testing.assert_array_almost_equal(A_vec.value, np.array([12.3 * 12.3, 54.3 * 12.3, 91.3 * 12.3]))
        self.assertEqual(A_vec.unit, 'L2/min2')
        np.testing.assert_array_almost_equal(
            A_vec.uncert,
            np.array([
                np.sqrt((12.3 * 2.6)**2 + (12.3 * 2.6)**2),
                np.sqrt((12.3 * 5.4)**2 + (54.3 * 2.6)**2),
                np.sqrt((12.3 * 10.56)**2 + (91.3 * 2.6)**2),
            ]))

    def testDivEqual(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)

        A /= B
        self.assertAlmostEqual(A.value, 12.3 / 745.1)
        self.assertTrue(A._unitObject._assertEqual('L/min-m'))
        self.assertAlmostEqual(A.uncert, np.sqrt((1 / 745.1 * 2.6)**2 + (12.3 / (745.1**2) * 53.9)**2))

        A = variable(12.3, 'L/min', uncert=2.6)
        A /= 2
        self.assertAlmostEqual(A.value, 12.3 / 2)
        self.assertEqual(A.unit, 'L/min')
        self.assertAlmostEqual(A.uncert, np.sqrt((1 / 2 * 2.6)**2))

        A = variable(12.3, 'L/min', uncert=2.6)
        B = 2
        B /= A
        self.assertAlmostEqual(B.value, 2 / 12.3)
        self.assertEqual(B.unit, 'min/L')
        self.assertAlmostEqual(B.uncert, np.sqrt((2 / (12.3**2) * 2.6)**2))

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm', uncert=[53.9, 24.75, 6.4])

        A_vec /= B_vec
        np.testing.assert_array_almost_equal(A_vec.value, np.array([12.3 / 745.1, 54.3 / 496.13, 91.3 / 120.54]))
        self.assertTrue(A_vec._unitObject._assertEqual('L/min-m'))
        np.testing.assert_array_almost_equal(
            A_vec.uncert,
            np.array([
                np.sqrt((1 / 745.1 * 2.6)**2 + (12.3 / (745.1**2) * 53.9)**2),
                np.sqrt((1 / 496.13 * 5.4)**2 + (54.3 / (496.13**2) * 24.75)**2),
                np.sqrt((1 / 120.54 * 10.56)**2 + (91.3 / (120.54**2) * 6.4)**2),
            ]))

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        A = variable(12.3, 'L/min', uncert=2.6)
        A_vec /= A
        np.testing.assert_array_almost_equal(A_vec.value, np.array([12.3 / 12.3, 54.3 / 12.3, 91.3 / 12.3]))
        self.assertEqual(A_vec.unit, '1')
        np.testing.assert_array_almost_equal(
            A_vec.uncert,
            np.array([
                np.sqrt((1 / 12.3 * 2.6)**2 + (12.3 / (12.3**2) * 2.6)**2),
                np.sqrt((1 / 12.3 * 5.4)**2 + (54.3 / (12.3**2) * 2.6)**2),
                np.sqrt((1 / 12.3 * 10.56)**2 + (91.3 / (12.3**2) * 2.6)**2),
            ]))

    def testPrintValueAndUncertScalar(self):
        A = variable(123456789 * 10**(0), 'm', uncert=123456789 * 10**(-2), nDigits=3)
        self.assertEqual(str(A), '123000000 +/- 1000000 [m]')

        A = variable(123456789 * 10**(-2), 'm', uncert=123456789 * 10**(-4), nDigits=3)
        self.assertEqual(str(A), '1230000 +/- 10000 [m]')

        A = variable(123456789 * 10**(-4), 'm', uncert=123456789 * 10**(-6), nDigits=3)
        self.assertEqual(str(A), '12300 +/- 100 [m]')

        A = variable(123456789 * 10**(-6), 'm', uncert=123456789 * 10**(-8), nDigits=3)
        self.assertEqual(str(A), '123 +/- 1 [m]')

        A = variable(123456789 * 10**(-7), 'm', uncert=123456789 * 10**(-9), nDigits=3)
        self.assertEqual(str(A), '12.3 +/- 0.1 [m]')

        A = variable(123456789 * 10**(-8), 'm', uncert=123456789 * 10**(-10), nDigits=3)
        self.assertEqual(str(A), '1.23 +/- 0.01 [m]')

        A = variable(123456789 * 10**(-9), 'm', uncert=123456789 * 10**(-11), nDigits=3)
        self.assertEqual(str(A), '0.123 +/- 0.001 [m]')

        A = variable(123456789 * 10**(-10), 'm', uncert=123456789 * 10**(-12), nDigits=3)
        self.assertEqual(str(A), '0.0123 +/- 0.0001 [m]')

        A = variable(123456789 * 10**(-12), 'm', uncert=123456789 * 10**(-14), nDigits=3)
        self.assertEqual(str(A), '0.000123 +/- 1e-06 [m]')

        A = variable(123456789 * 10**(-14), 'm', uncert=123456789 * 10**(-16), nDigits=3)
        self.assertEqual(str(A), '0.00000123 +/- 1e-08 [m]')

        A = variable(123456789 * 10**(-16), 'm', uncert=123456789 * 10**(-18), nDigits=3)
        self.assertEqual(str(A), '0.0000000123 +/- 1e-10 [m]')

        A = variable(10.0, 'm', uncert=0.1)
        self.assertEqual(str(A), '10.0 +/- 0.1 [m]')

    def testPrintValueScalar(self):
        A = variable(123456789 * 10**(0), 'm', nDigits=6)
        self.assertEqual(str(A), '1.23457e+08 [m]')

        A = variable(123456789 * 10**(-2), 'm', nDigits=7)
        self.assertEqual(str(A), '1234568 [m]')

        A = variable(123456789 * 10**(-4), 'm', nDigits=3)
        self.assertEqual(str(A), '1.23e+04 [m]')

        A = variable(123456789 * 10**(-6), 'm', nDigits=3)
        self.assertEqual(str(A), '123 [m]')

        A = variable(123456789 * 10**(-7), 'm', nDigits=3)
        self.assertEqual(str(A), '12.3 [m]')

        A = variable(123456789 * 10**(-8), 'm', nDigits=3)
        self.assertEqual(str(A), '1.23 [m]')

        A = variable(123456789 * 10**(-9), 'm', nDigits=2)
        self.assertEqual(str(A), '0.12 [m]')

        A = variable(123456789 * 10**(-10), 'm', nDigits=3)
        self.assertEqual(str(A), '0.0123 [m]')

        A = variable(123456789 * 10**(-12), 'm', nDigits=3)
        self.assertEqual(str(A), '0.000123 [m]')

        A = variable(123456789 * 10**(-14), 'm', nDigits=5)
        self.assertEqual(str(A), '1.2346e-06 [m]')

        A = variable(123456789 * 10**(-16), 'm', nDigits=3)
        self.assertEqual(str(A), '1.23e-08 [m]')

    def testRoot(self):
        A = variable(10, 'L2/min2')
        a = np.sqrt(A)
        self.assertEqual(a.value, np.sqrt(10))
        self.assertEqual(a.unit, 'L/min')

        a = A**(1 / 2)
        self.assertEqual(a.value, 10**(1 / 2))
        self.assertEqual(a.unit, 'L/min')

        for i in range(1, 20):
            u = f'L{i+1}/min{i+1}'
            A = variable(10, u)
            power = 1 / (i + 1)
            a = A**power
            self.assertAlmostEqual(a.value, 10**(1 / (i + 1)))
            self.assertEqual(a.unit, 'L/min')

            scale = uniform(0.5, 0.99)
            with self.assertRaises(Exception) as context:
                A ** (power * scale)
            self.assertTrue(f'You can not raise a variable with the unit {u} to the power of {power * scale}' in str(context.exception))

            scale = uniform(1.01, 1.5)
            with self.assertRaises(Exception) as context:
                A ** (power * scale)
            self.assertTrue(f'You can not raise a variable with the unit {u} to the power of {power * scale}' in str(context.exception))

        A = variable(10, 'L2/m')
        with self.assertRaises(Exception) as context:
            np.sqrt(A)
        self.assertTrue('You can not raise a variable with the unit L2/m to the power of 0.5' in str(context.exception))

    def testLargerUncertThenValue(self):

        A = variable(0.003, 'L/min', 0.2)
        self.assertEqual(str(A), '0.0 +/- 0.2 [L/min]')

        A = variable(1, 'L/min', 10)
        self.assertEqual(str(A), '0 +/- 10 [L/min]')

        A = variable(1, 'L/min', 2.3)
        self.assertEqual(str(A), '1 +/- 2 [L/min]')

        A = variable(105, 'L/min', 135.653)
        self.assertEqual(str(A), '100 +/- 100 [L/min]')

        A = variable(10.5, 'L/min', 135.653)
        self.assertEqual(str(A), '0 +/- 100 [L/min]')

        A = variable(0.0543, 'L/min', 0.07)
        self.assertEqual(str(A), '0.05 +/- 0.07 [L/min]')

        A = variable(0.0543, 'L/min', 0.7)
        self.assertEqual(str(A), '0.0 +/- 0.7 [L/min]')

    def testUnitless(self):
        with self.assertRaises(Exception) as context:
            A = variable(10, 'P', 1)
        self.assertTrue('''The unit (P) was not found. Therefore it was interpreted as a prefix and a unit. However a combination of prefix and unit which matches P was not found''' in str(context.exception))

        A = variable(10, '1', 1)
        self.assertEqual(A.value, 10)
        self.assertEqual(A.unit, '1')
        self.assertEqual(A.uncert, 1)

        A = variable(10, '', 1)
        self.assertEqual(A.value, 10)
        self.assertEqual(A.unit, '1')
        self.assertEqual(A.uncert, 1)

    def test_r_pow(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)
        C = variable(74.51, '1', uncert=5.39)
        D = variable(0.34, '1', uncert=0.01)

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm', uncert=[53.9, 24.75, 6.4])
        C_vec = variable([745.1, 496.13, 120.54], '1', uncert=[53.9, 24.75, 6.4])
        D_vec = variable([0.34, 0.64, 0.87], '1', uncert=[0.01, 0.084, 0.12])

        with self.assertRaises(Exception) as context:
            2**A
        self.assertTrue('The exponent can not have a unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            2**B
        self.assertTrue('The exponent can not have a unit' in str(context.exception))

        c = 2**C
        self.assertEqual(c.value, 2**74.51)
        self.assertEqual(c.unit, '1')
        self.assertEqual(c.uncert, np.sqrt((2**74.51 * np.log(2) * 5.39)**2 + (74.51 * 2**(74.51 - 1) * 0)**2))

        d = 2**D
        self.assertEqual(d.value, 2**0.34)
        self.assertEqual(d.unit, '1')
        self.assertEqual(d.uncert, np.sqrt((2**0.34 * np.log(2) * 0.01)**2 + (0.34 * 2**(0.34 - 1) * 0)**2))

        with self.assertRaises(Exception) as context:
            2**A_vec
        self.assertTrue('The exponent can not have a unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            2**B_vec
        self.assertTrue('The exponent can not have a unit' in str(context.exception))
     
        d_vec = 2**D_vec
        np.testing.assert_equal(d_vec.value, [2**0.34, 2**0.64, 2**0.87])
        self.assertEqual(d_vec.unit, '1')
        self.assertEqual(d_vec.uncert[0], np.sqrt((2**0.34 * np.log(2) * 0.01)**2 + (0.34 * 2**(0.34 - 1) * 0)**2))

        
    def testPrettyPrint(self):
        a = variable(12.3, 'm')
        b = variable(12.3, 'm', 2.5)
        c = variable([12.3, 56.2], 'm')
        d = variable([12.3, 56.2], 'm', [2.5, 7.3])

        self.assertEqual(a.__str__(pretty=False), '12.3 [m]')
        self.assertEqual(b.__str__(pretty=False), '12 +/- 2 [m]')
        self.assertEqual(c.__str__(pretty=False), '[12.3, 56.2] [m]')
        self.assertEqual(d.__str__(pretty=False), '[12, 56] +/- [2, 7] [m]')

        self.assertEqual(a.__str__(pretty=True), '12.3\\ \\left [m\\right ]')
        self.assertEqual(b.__str__(pretty=True), '12 \pm 2\\ \\left [m\\right ]')
        self.assertEqual(c.__str__(pretty=True), '[12.3, 56.2]\\ \\left [m\\right ]')
        self.assertEqual(d.__str__(pretty=True), '[12, 56] \pm [2, 7]\\ \\left [m\\right ]')

    def testMax(self):

        A = variable(10, 'm', 2.3)
        A = np.max(A)
        self.assertEqual(A.value, 10)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 2.3)

        A = variable(10, 'm')
        A = np.max(A)
        self.assertEqual(A.value, 10)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 0)

        A = variable([10, 15.7], 'm', [2.3, 5.6])
        A = np.max(A)
        self.assertEqual(A.value, 15.7)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 5.6)

        A = variable([10, 15.7], 'm')
        A = np.max(A)
        self.assertEqual(A.value, 15.7)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 0)

    def testMin(self):

        A = variable(10, 'm', 2.3)
        A = np.min(A)
        self.assertEqual(A.value, 10)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 2.3)

        A = variable(10, 'm')
        A = np.min(A)
        self.assertEqual(A.value, 10)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 0)

        A = variable([10, 15.7], 'm', [2.3, 5.6])
        A = np.min(A)
        self.assertEqual(A.value, 10)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 2.3)

        A = variable([10, 15.7], 'm')
        A = np.min(A)
        self.assertEqual(A.value, 10)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 0)

    def testMean(self):

        A = variable(10, 'm', 2.3)
        A = np.mean(A)
        self.assertEqual(A.value, 10)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 2.3)

        A = variable(10, 'm')
        A = np.mean(A)
        self.assertEqual(A.value, 10)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 0)

        A = variable([10, 15.7], 'm', [2.3, 5.6])
        A = np.mean(A)
        self.assertEqual(A.value, (10 + 15.7) / 2)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, np.sqrt((1 / 2 * 2.3) ** 2 + (1 / 2 * 5.6) ** 2))

        A = variable([10, 15.7], 'm')
        A = np.mean(A)
        self.assertEqual(A.value, (10 + 15.7) / 2)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 0)

    def testSum(self):
        A = variable(10, 'm', 2.3)
        B = variable(8, 'm', 1.7)
        C = sum([A, B])
        self.assertEqual(C.value, 10 + 8)
        self.assertEqual(C.unit, 'm')
        self.assertEqual(C.uncert, np.sqrt(2.3**2 + 1.7**2))

    def testTrig(self):
        a = variable(75, 'deg', 1)
        b = np.sin(a)
        self.assertAlmostEqual(b.value, 0.96592582628)
        self.assertEqual(b.unit, '1')
        self.assertAlmostEqual(b.uncert, np.sqrt(((1 * np.pi / 180) * (np.cos(75 * np.pi / 180)))**2))

        a = variable(75, 'deg', 1)
        b = np.cos(a)
        self.assertAlmostEqual(b.value, 0.2588190451)
        self.assertEqual(b.unit, '1')
        self.assertAlmostEqual(b.uncert, np.sqrt(((1 * np.pi / 180) * (-np.sin(75 * np.pi / 180)))**2))

        a = variable(75, 'deg', 1)
        b = np.tan(a)
        self.assertAlmostEqual(b.value, 3.73205080757)
        self.assertEqual(b.unit, '1')
        self.assertAlmostEqual(b.uncert, np.sqrt(((1 * np.pi / 180) * (2 / (np.cos(2 * 75 * np.pi / 180) + 1)))**2))

        a = variable(0.367, 'rad', 0.0796)
        b = np.sin(a)
        self.assertAlmostEqual(b.value, 0.35881682685)
        self.assertEqual(b.unit, '1')
        self.assertAlmostEqual(b.uncert, np.sqrt(((0.0796) * (np.cos(0.367)))**2))

        a = variable(0.367, 'rad', 0.0796)
        b = np.cos(a)
        self.assertAlmostEqual(b.value, 0.9334079948)
        self.assertEqual(b.unit, '1')
        self.assertAlmostEqual(b.uncert, np.sqrt(((0.0796) * (-np.sin(0.367)))**2))

        a = variable(0.367, 'rad', 0.0796)
        b = np.tan(a)
        self.assertAlmostEqual(b.value, 0.38441584907)
        self.assertEqual(b.unit, '1')
        self.assertAlmostEqual(b.uncert, np.sqrt(((0.0796) * (2 / (np.cos(2 * 0.367) + 1)))**2))

    def testProductRule(self):

        a = variable(23, 'deg', 2)
        b = np.sin(a)
        c = a * b
        val = 23
        unc = 2
        self.assertEqual(c.value, val * np.sin(np.pi / 180 * val))
        self.assertEqual(c.unit, 'deg')
        self.assertEqual(c.uncert, np.sqrt((unc * (np.sin(np.pi / 180 * val) + (np.pi / 180 * val) * np.cos(np.pi / 180 * val)))**2))

        a = variable(23, 'deg', 2)
        a.convert('rad')
        b = np.sin(a)
        c = a * b
        val = np.pi / 180 * 23
        unc = np.pi / 180 * 2
        self.assertEqual(c.value, val * np.sin(val))
        self.assertEqual(c.unit, 'rad')
        self.assertEqual(c.uncert, np.sqrt((unc * (np.sin(val) + val * np.cos(val)))**2))

        a = variable(23, 'deg', 2)
        b = np.sin(a)
        a.convert('rad')
        c = a * b
        val = np.pi / 180 * 23
        unc = np.pi / 180 * 2
        self.assertAlmostEqual(c.value, val * np.sin(val))
        self.assertEqual(c.unit, 'rad')
        self.assertAlmostEqual(c.uncert, np.sqrt((unc * (np.sin(val) + val * np.cos(val)))**2))

        a = variable(np.pi / 180 * 23, 'rad', np.pi / 180 * 2)
        b = np.sin(a)
        a.convert('deg')
        c = a * b
        val = 23
        unc = 2
        self.assertEqual(c.value, val * np.sin(np.pi / 180 * val))
        self.assertEqual(c.unit, 'deg')
        self.assertEqual(c.uncert, np.sqrt((unc * (np.sin(np.pi / 180 * val) + (np.pi / 180 * val) * np.cos(np.pi / 180 * val)))**2))

        a = variable(200, 'L/min', 1.5)
        b = a**2
        self.assertAlmostEqual(b.value, 200 ** 2)
        self.assertEqual(b.unit, 'L2/min2')
        self.assertAlmostEqual(b.uncert, np.sqrt((1.5 * 2 * 200)**2))

        b /= a
        self.assertAlmostEqual(b.value, 200)
        self.assertEqual(b.unit, 'L/min')
        self.assertAlmostEqual(b.uncert, np.sqrt((1.5 * 1)**2))

        b += a
        self.assertAlmostEqual(b.value, 400)
        self.assertEqual(b.unit, 'L/min')
        self.assertAlmostEqual(b.uncert, np.sqrt((1.5 * 2)**2))

        b *= a
        self.assertAlmostEqual(b.value, 80000)
        self.assertEqual(b.unit, 'L2/min2')
        self.assertAlmostEqual(b.uncert, np.sqrt((1.5 * 4 * 200)**2))

        b /= 23.8
        self.assertAlmostEqual(b.value, 3361.3445378151260504201680672269)
        self.assertEqual(b.unit, 'L2/min2')
        self.assertAlmostEqual(b.uncert, np.sqrt((1.5 * 0.16806722689075630252100840336134 * 200)**2))

        b *= np.sin(a * variable(1, 'rad-min/L'))
        self.assertAlmostEqual(b.value, -2935.453099878973383976532508069948132551783965504369163751)
        self.assertEqual(b.unit, 'L2/min2')
        self.assertAlmostEqual(b.uncert, np.sqrt((1.5 * 2 * 200 * (2 * np.sin(200) + 200 * np.cos(200)) / 23.8)**2))

        a /= variable(100, 'L/min')
        a.convert('')
        b /= np.exp(a)
        self.assertAlmostEqual(b.value, -397.2703766999135885608809478456258749790006977070134430245)
        self.assertEqual(b.unit, 'L2/min2')
        self.assertAlmostEqual(b.uncert, np.sqrt((1.5 * 2 * 200 * np.exp(-200 / 100) * ((2 * 100 - 200) * np.sin(200) + 100 * 200 * np.cos(200)) / (23.8 * 100))**2))

        a = variable(37, 'deg', 2.3)
        b = a**2 * np.cos(3 * a * np.sin(a))
        self.assertAlmostEqual(b.value, 539.274244145)
        self.assertEqual(b.unit, 'deg2')
        self.assertAlmostEqual(b.uncert, np.sqrt((2.3 * (-44.47986837334018052364896281900061654705050149285482386191581710))**2))

        a = variable(37, 'deg', 2.3)
        b = np.cos(3 * a * np.sin(a))
        a.convert('rad')
        b *= a**2
        self.assertAlmostEqual(b.value, 0.1642723288)
        self.assertEqual(b.unit, 'rad2')
        self.assertAlmostEqual(b.uncert, np.sqrt((np.pi / 180 * 2.3 * (-0.776320153968480543428298272676994570718859842395105372525387644))**2))

        a = variable(2.3, '', 0.11)
        b = variable(1.5, 'rad', 0.89)
        d = np.exp(a**2) * np.cos(b * np.tan(b / 9) + 17.5)
        self.assertAlmostEqual(d.value, 90.459733187853914019107237427714051178688422865118118182659)
        self.assertEqual(d.unit, '1')
        dd_da = 416.11477266412800448789329216748463542196674517954334364023
        dd_db = 59.945989031664355557893562375983977678359283356335782472980
        self.assertAlmostEqual(d.uncert, np.sqrt((dd_da * 0.11)**2 + (dd_db * 0.89)**2))

        """
        e = \sum^{\infty}_{n=0} 1/(n!)
        b = \sum^{\infty}_{n=0} 1/(n!) * a = e*a
        \frac{\partial b}{\partial a} = e
        """
        a = variable(2.3, 'L/min', 0.0237)
        b = variable(0, 'L/min')
        for i in range(15):
            b += 1 / variable(np.math.factorial(i)) * a
        self.assertAlmostEqual(b.value, np.e * 2.3)
        self.assertEqual(b.unit, 'L/min')
        self.assertAlmostEqual(b.uncert, np.sqrt((np.e * 0.0237)**2))

    def testCovariance(self):
        a = variable(123, 'L/min', 9.7)
        b = variable(93, 'Pa', 1.2)
        a._addCovariance(b, [23])
        b._addCovariance(a, [23])
        c = a * b
        self.assertEqual(c.value, 123 * 93)
        self.assertTrue(c._unitObject._assertEqual('L-Pa/min'))
        self.assertEqual(c.uncert, np.sqrt((123 * 1.2)**2 + (93 * 9.7)**2 + 2 * 93 * 123 * 23))

        a = variable(123, 'L/min', 9.7)
        b = variable(93, 'Pa', 1.2)
        a._addCovariance(b, [23])
        b._addCovariance(a, [23])
        a.convert('m3/s')
        c = a * b
        self.assertEqual(c.value, 123 * 93 / 1000 / 60)
        self.assertTrue(c._unitObject._assertEqual('m3-Pa/s'))
        self.assertEqual(c.uncert, np.sqrt((123 / 1000 / 60 * 1.2)**2 + (93 * 9.7 / 1000 / 60)**2 + 2 * 93 * 123 / 1000 / 60 * 23))

    def testConvert(self):
        a = variable(1, 'km')
        b = variable(1, 'm')
        c = a * b
        c.convert('mm2')

    def testCompare(self):
        a = variable(1, 'm')
        b = variable([2, 3, 4], 'm')
        np.testing.assert_equal(a < b, [True, True, True])
        np.testing.assert_equal(a <= b, [True, True, True])
        np.testing.assert_equal(a > b, [False, False, False])
        np.testing.assert_equal(a >= b, [False, False, False])
        np.testing.assert_equal(a == b, [False, False, False])
        np.testing.assert_equal(a != b, [True, True, True])

        a = variable([2, 3, 4], 'm')
        b = variable(1, 'm')
        np.testing.assert_equal(a < b, [False, False, False])
        np.testing.assert_equal(a <= b, [False, False, False])
        np.testing.assert_equal(a > b, [True, True, True])
        np.testing.assert_equal(a >= b, [True, True, True])
        np.testing.assert_equal(a == b, [False, False, False])
        np.testing.assert_equal(a != b, [True, True, True])

        a = variable([1, 2], 'm')
        b = variable([2, 3, 4], 'm')
        with self.assertRaises(Exception) as context:
            c = a < b
        self.assertTrue("operands could not be broadcast together with shapes (2,) (3,)" in str(context.exception))

        with self.assertRaises(Exception) as context:
            c = a <= b
        self.assertTrue("operands could not be broadcast together with shapes (2,) (3,)" in str(context.exception))

        with self.assertRaises(Exception) as context:
            c = a > b
        self.assertTrue("operands could not be broadcast together with shapes (2,) (3,)" in str(context.exception))

        with self.assertRaises(Exception) as context:
            c = a >= b
        self.assertTrue("operands could not be broadcast together with shapes (2,) (3,)" in str(context.exception))

        with self.assertRaises(Exception) as context:
            c = a == b
        self.assertTrue("operands could not be broadcast together with shapes (2,) (3,)" in str(context.exception))

        with self.assertRaises(Exception) as context:
            c = a != b
        self.assertTrue("operands could not be broadcast together with shapes (2,) (3,)" in str(context.exception))
        
        
        a = variable(1, 'm')
        b = variable(2, 'C')
        with self.assertRaises(Exception) as context:
            c = a < b
        self.assertTrue("You cannot compare 1 [m] and 2 [C] as they do not have the same SI base unit" in str(context.exception))

        with self.assertRaises(Exception) as context:
            c = a <= b
        self.assertTrue("You cannot compare 1 [m] and 2 [C] as they do not have the same SI base unit" in str(context.exception))

        with self.assertRaises(Exception) as context:
            c = a > b
        self.assertTrue("You cannot compare 1 [m] and 2 [C] as they do not have the same SI base unit" in str(context.exception))

        with self.assertRaises(Exception) as context:
            c = a >= b
        self.assertTrue("You cannot compare 1 [m] and 2 [C] as they do not have the same SI base unit" in str(context.exception))

        with self.assertRaises(Exception) as context:
            c = a == b
        self.assertTrue("You cannot compare 1 [m] and 2 [C] as they do not have the same SI base unit" in str(context.exception))

        with self.assertRaises(Exception) as context:
            c = a != b
        self.assertTrue("You cannot compare 1 [m] and 2 [C] as they do not have the same SI base unit" in str(context.exception))

        a = variable([1, 2, 3], 'm')
        b = variable([2, 3, 4], 'm')
        np.testing.assert_equal(a < b, [True, True, True])
        np.testing.assert_equal(a <= b, [True, True, True])
        np.testing.assert_equal(a > b, [False, False, False])
        np.testing.assert_equal(a >= b, [False, False, False])
        np.testing.assert_equal(a == b, [False, False, False])
        np.testing.assert_equal(a != b, [True, True, True])
        
        
        a = variable(10,'L/min')
        b = variable(1, 'm3/h')
        np.testing.assert_equal(a>b, False)
        np.testing.assert_equal(a<b, True)
        np.testing.assert_equal(a>=b, False)
        np.testing.assert_equal(a<=b, True)
        np.testing.assert_equal(a==b, False)
        np.testing.assert_equal(a!=b, True)

    def testSetitem(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(45, 'Pa', 1.2)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])

        A_vec[1] = A
        np.testing.assert_equal(A_vec.value, [12.3, 12.3, 91.3])
        self.assertEqual(A_vec.unit, 'L/min')
        np.testing.assert_equal(A_vec.uncert, [2.6, 2.6, 10.56])
        
        with self.assertRaises(Exception) as context:
            A_vec[0] = B
        self.assertTrue("You can not set an element of [12, 12, 90] +/- [3, 3, 10] [L/min] with 45 +/- 1 [Pa] as they do not have the same unit" in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec[0] = A_vec
        self.assertTrue("You can only set an element with a scalar variable" in str(context.exception))

        with self.assertRaises(Exception) as context:
            A[0] = B
        self.assertTrue("'scalarVariable' object does not support item assignment" in str(context.exception))
        
        
    def testAppend(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(45, 'Pa', 1.2)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])

        A_vec.append(A)
        np.testing.assert_equal(A_vec.value, [12.3, 54.3, 91.3, 12.3])
        self.assertEqual(A_vec.unit, 'L/min')
        np.testing.assert_equal(A_vec.uncert, [2.6, 5.4, 10.56, 2.6])
        
        with self.assertRaises(Exception) as context:
            A_vec.append(B)
        self.assertTrue("You can not set an element of [12, 54, 90, 12] +/- [3, 5, 10, 3] [L/min] with 45 +/- 1 [Pa] as they do not have the same unit" in str(context.exception))

        A_vec.append(A_vec)
        np.testing.assert_equal(A_vec.value, [12.3, 54.3, 91.3, 12.3] * 2)
        self.assertEqual(A_vec.unit, 'L/min')
        np.testing.assert_equal(A_vec.uncert, [2.6, 5.4, 10.56, 2.6] * 2)
        
        with self.assertRaises(Exception) as context:
            A.append(B)
        self.assertTrue("'scalarVariable' object has no attribute 'append'" in str(context.exception))
        
        

if __name__ == '__main__':
    unittest.main()
    # test().testCompare()
    