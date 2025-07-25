import unittest
import math
import numpy as np
from variable import variable, logarithmic
from unit import unit



class test(unittest.TestCase):

    def testLinspace(self):
        
        a = variable(1, 'm', 0.1)
        b = variable(2, 'm', 0.2)
        c = np.linspace(a,b, 10, endpoint = False)
        np.testing.assert_array_almost_equal(c.value, np.linspace(1,2,10, endpoint = False))
        self.assertEqual(c.unit, 'm')
        np.testing.assert_array_almost_equal(c.uncert, np.linspace(0.1,0.2,10, endpoint = False))
        
        
        a = variable(1, 'm', 0.1)
        b = variable(2, 'm', 0.2)
        c = np.linspace(a,b, 10, endpoint = True)
        np.testing.assert_array_almost_equal(c.value, np.linspace(1,2,10, endpoint = True))
        self.assertEqual(c.unit, 'm')
        np.testing.assert_array_almost_equal(c.uncert, np.linspace(0.1,0.2,10, endpoint = True))
        
        
        a = variable(1, 'm', 0.1)
        b = variable(2, 'm')
        c = np.linspace(a,b, 10, endpoint = True)
        np.testing.assert_array_almost_equal(c.value, np.linspace(1,2,10, endpoint = True))
        self.assertEqual(c.unit, 'm')
        np.testing.assert_array_almost_equal(c.uncert, np.linspace(0.1,0,10, endpoint = True))
        
        
        a = variable(1, 'm', 0.1)
        b = 2
        c = np.linspace(a,b, 10, endpoint = True)
        np.testing.assert_array_almost_equal(c.value, np.linspace(1,2,10, endpoint = True))
        self.assertEqual(c.unit, 'm')
        np.testing.assert_array_almost_equal(c.uncert, np.linspace(0.1,0,10, endpoint = True))
        
        a = variable(1, 'm')
        b = variable(2, 'm', 0.2)
        c = np.linspace(a,b, 10, endpoint = True)
        np.testing.assert_array_almost_equal(c.value, np.linspace(1,2,10, endpoint = True))
        self.assertEqual(c.unit, 'm')
        np.testing.assert_array_almost_equal(c.uncert, np.linspace(0,0.2,10, endpoint = True))
        
        a = 1
        b = variable(2, 'm', 0.2)
        c = np.linspace(a,b, 10, endpoint = True)
        np.testing.assert_array_almost_equal(c.value, np.linspace(1,2,10, endpoint = True))
        self.assertEqual(c.unit, 'm')
        np.testing.assert_array_almost_equal(c.uncert, np.linspace(0,0.2,10, endpoint = True))
        
        a = variable(1, 'm', 0.1)
        b = variable(2, 'm3', 0.2)
        with self.assertRaises(Exception) as context:
            c = np.linspace(a,b, 10, endpoint = True)
        self.assertTrue('The arguments "start" and "stop" has to have the same unit' in str(context.exception))

    def testArguments(self):
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
        self.assertTrue(
            "could not convert string to float: 'hej'" in str(context.exception))

        with self.assertRaises(Exception) as context:
            variable('med', 'm', 1.0)
        self.assertTrue(
            "could not convert string to float: 'med'" in str(context.exception))

        with self.assertRaises(Exception) as context:
            variable(1.3, 'm', [1.0, 2.3])
        self.assertTrue("The lenght of the value has to be equal to the lenght of the uncertanty" in str(
            context.exception))

        with self.assertRaises(Exception) as context:
            variable(1.3, 'm', np.array([1.0, 2.3]))
        self.assertTrue("The lenght of the value has to be equal to the lenght of the uncertanty" in str(
            context.exception))

        with self.assertRaises(Exception) as context:
            variable(np.array([1.0, 2.3]), 'm', 1.5)
        self.assertTrue("The lenght of the value has to be equal to the lenght of the uncertanty" in str(
            context.exception))

        with self.assertRaises(Exception) as context:
            variable([1.0, 2.3], 'm', 1.5)
        self.assertTrue("The lenght of the value has to be equal to the lenght of the uncertanty" in str(
            context.exception))

        a = variable(1, 'm')
        b = variable(2, 'm')
        c = variable([a,b])
        d = variable([1,2], 'm')
        np.testing.assert_equal(c.value, d.value)
        self.assertEqual(c.unit, d.unit)
        np.testing.assert_equal(c.uncert, d.uncert)
        
        
        a = variable(1, 'm')
        b = variable(2, 'm3')
        with self.assertRaises(Exception) as context:
            c = variable([a,b])
        self.assertTrue("You can only create an array variable from a list of scalar variables if all the scalar variables have the same unit" in str(context.exception))

    def test_add(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'L/min', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'L/min',
                         uncert=[53.9, 24.75, 6.4])

        C = A + B
        self.assertAlmostEqual(C.value, 12.3 + 745.1)
        self.assertEqual(C.unit, 'L/min')
        self.assertAlmostEqual(C.uncert, np.sqrt((1 * 2.6)**2 + (1 * 53.9)**2))

        C.convert('m3/s')
        self.assertAlmostEqual(C.value, (12.3 + 745.1) / 1000 / 60)
        self.assertEqual(C.unit, 'm3/s')
        self.assertAlmostEqual(C.uncert, np.sqrt(
            (1 * 2.6 / 1000 / 60)**2 + (1 * 53.9 / 1000 / 60)**2))

        C_vec = A_vec + B_vec
        np.testing.assert_almost_equal(C_vec.value, np.array(
            [12.3 + 745.1, 54.3 + 496.13, 91.3 + 120.54]))
        self.assertEqual(C_vec.unit, 'L/min')
        np.testing.assert_almost_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((1 * 2.6)**2 + (1 * 53.9)**2),
                np.sqrt((1 * 5.4)**2 + (1 * 24.75)**2),
                np.sqrt((1 * 10.56)**2 + (1 * 6.4)**2),
            ]))

        C_vec.convert('mL/h')
        np.testing.assert_almost_equal(C_vec.value, np.array(
            [(12.3 + 745.1) * 1000 * 60, (54.3 + 496.13) * 1000 * 60, (91.3 + 120.54) * 1000 * 60]))
        self.assertEqual(C_vec.unit, 'mL/h')
        np.testing.assert_almost_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((1 * 2.6 * 1000 * 60)**2 + (1 * 53.9 * 1000 * 60)**2),
                np.sqrt((1 * 5.4 * 1000 * 60)**2 + (1 * 24.75 * 1000 * 60)**2),
                np.sqrt((1 * 10.56 * 1000 * 60)**2 + (1 * 6.4 * 1000 * 60)**2),
            ]))

        A = variable(7, '%', 0.1)
        B = 1 + A
        self.assertAlmostEqual(B.value, 8)
        self.assertEqual(B.unit, '%')
        self.assertEqual(B.uncert, 0.1)

    def test_sub(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'L/min', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'L/min',
                         uncert=[53.9, 24.75, 6.4])

        C = A - B
        self.assertAlmostEqual(C.value, 12.3 - 745.1)
        self.assertEqual(C.unit, 'L/min')
        self.assertAlmostEqual(C.uncert, np.sqrt((1 * 2.6)**2 + (1 * 53.9)**2))

        C.convert('kL/s')
        self.assertAlmostEqual(C.value, (12.3 - 745.1) / 1000 / 60)
        self.assertEqual(C.unit, 'kL/s')
        self.assertAlmostEqual(C.uncert, np.sqrt(
            (1 * 2.6 / 1000 / 60)**2 + (1 * 53.9 / 1000 / 60)**2))

        C_vec = A_vec - B_vec
        np.testing.assert_almost_equal(C_vec.value, np.array(
            [12.3 - 745.1, 54.3 - 496.13, 91.3 - 120.54]))
        self.assertEqual(C_vec.unit, 'L/min')
        np.testing.assert_almost_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((1 * 2.6)**2 + (1 * 53.9)**2),
                np.sqrt((1 * 5.4)**2 + (1 * 24.75)**2),
                np.sqrt((1 * 10.56)**2 + (1 * 6.4)**2),
            ]))

        A = variable(7, '%', 0.1)
        B = 1 - A
        self.assertAlmostEqual(B.value, -6)
        self.assertEqual(B.unit, '%')
        self.assertEqual(B.uncert, 0.1)

    def test_add_with_different_units(self):
        A = variable(12.3, 'm3/s', uncert=2.6)
        B = variable(745.1, 'L/min', uncert=53.9)
        C = A + B
        self.assertAlmostEqual(C.value, 12.3 + 745.1 / 1000 / 60)
        self.assertEqual(C.unit, 'm3/s')
        self.assertAlmostEqual(C.uncert, np.sqrt(
            2.6**2 + (53.9 / 1000 / 60)**2))

        A = variable(12.3, 'C', uncert=2.6)
        B = variable(745.1, 'K', uncert=53.9)
        C = A + B
        self.assertAlmostEqual(C.value, 12.3 + 273.15 + 745.1)
        self.assertEqual(C.unit, 'K')
        self.assertAlmostEqual(C.uncert, np.sqrt(2.6**2 + 53.9**2))

        A = variable(12.3, 'C', uncert=2.6)
        B = variable(745.1, 'K', uncert=53.9)
        C = A + B
        self.assertAlmostEqual(C.value, 12.3 + 273.15 + 745.1)
        self.assertEqual(C.unit, 'K')
        self.assertAlmostEqual(C.uncert, np.sqrt(2.6**2 + 53.9**2))

        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm',
                         uncert=[53.9, 24.75, 6.4])

        with self.assertRaises(Exception) as context:
            A + B
        self.assertTrue(
            'You tried to add a variable in [L/min] to a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec + B_vec
        self.assertTrue(
            'You tried to add a variable in [L/min] to a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        a = variable(10, 'L', 1.2)
        b = variable(100, 'mL', 3.9)
        c = a + b
        self.assertEqual(c.value, 10.1)
        self.assertEqual(c.unit, 'L')
        self.assertEqual(c.uncert, np.sqrt((1 * 1.2)**2 + (1 * 3.9/1000)**2))

        a = variable(10, 'C', 1.2)
        b = variable(100, 'muC', 3.9)
        c = a + b
        self.assertEqual(c.value, 10 + 100 * 1e-6)
        self.assertEqual(c.unit, 'C')
        self.assertEqual(c.uncert, np.sqrt((1 * 1.2)**2 + (1 * 3.9*1e-6)**2))

    def test_sub_with_different_units(self):
        A = variable(12.3, 'm3/s', uncert=2.6)
        B = variable(745.1, 'L/min', uncert=53.9)
        C = A - B
        self.assertAlmostEqual(C.value, 12.3 - 745.1 / 1000 / 60)
        self.assertEqual(C.unit, 'm3/s')
        self.assertAlmostEqual(C.uncert, np.sqrt(
            2.6**2 + (53.9 / 1000 / 60)**2))

        A = variable(12.3, 'C', uncert=2.6)
        B = variable(745.1, 'K', uncert=53.9)
        C = A - B
        self.assertAlmostEqual(C.value, 12.3 + 273.15 - 745.1)
        self.assertEqual(C.unit, 'DELTAK')
        self.assertAlmostEqual(C.uncert, np.sqrt(2.6**2 + 53.9**2))

        A = variable(12.3, 'C', uncert=2.6)
        B = variable(745.1, 'K', uncert=53.9)
        C = A - B
        self.assertAlmostEqual(C.value, 12.3 + 273.15 - 745.1)
        self.assertEqual(C.unit, 'DELTAK')
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
        B_vec = variable([745.1, 496.13, 120.54], 'm',
                         uncert=[53.9, 24.75, 6.4])

        with self.assertRaises(Exception) as context:
            A - B
        self.assertTrue(
            'You tried to subtract a variable in [m] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec - B_vec
        self.assertTrue(
            'You tried to subtract a variable in [m] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        a = variable(10, 'L', 1.2)
        b = variable(100, 'mL', 3.9)
        c = a - b
        self.assertEqual(c.value, 10 - 100 / 1000)
        self.assertEqual(c.unit, 'L')
        self.assertEqual(c.uncert, np.sqrt((1 * 1.2)**2 + (1 * 3.9/1000)**2))

        a = variable(10, 'C', 1.2)
        b = variable(100, 'muC', 3.9)
        c = a - b
        self.assertAlmostEqual(c.value, 10 - 100 * 1e-6)
        self.assertEqual(c.unit, 'DELTAK')
        self.assertEqual(c.uncert, np.sqrt((1 * 1.2)**2 + (1 * 3.9*1e-6)**2))

        a = variable(100, 'muC', 3.9)
        b = variable(10, 'C', 1.2)
        c = a - b
        self.assertAlmostEqual(c.value, -10 + 100 * 1e-6)
        self.assertEqual(c.unit, 'DELTAK')
        self.assertEqual(c.uncert, np.sqrt((1 * 1.2)**2 + (1 * 3.9*1e-6)**2))

    def test_multiply(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm',
                         uncert=[53.9, 24.75, 6.4])

        C = A * B

        self.assertAlmostEqual(C.value, 12.3 * 745.1)
        self.assertTrue(C._unitObject == unit('L-m/min'))
        self.assertAlmostEqual(C.uncert, np.sqrt(
            (745.1 * 2.6)**2 + (12.3 * 53.9)**2))

        C_vec = A_vec * B_vec
        np.testing.assert_array_equal(C_vec.value, np.array(
            [12.3 * 745.1, 54.3 * 496.13, 91.3 * 120.54]))
        self.assertTrue(C._unitObject == unit('L-m/min'))
        np.testing.assert_array_almost_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((745.1 * 2.6)**2 + (12.3 * 53.9)**2),
                np.sqrt((496.13 * 5.4)**2 + (54.3 * 24.75)**2),
                np.sqrt((120.54 * 10.56)**2 + (91.3 * 6.4)**2),
            ]))

        C_vec.convert('m3-km / s')
        np.testing.assert_array_equal(C_vec.value, np.array(
            [12.3 * 745.1, 54.3 * 496.13, 91.3 * 120.54]) / 1000 / 1000 / 60)
        self.assertTrue(C_vec._unitObject == unit('m3-km/s'))
        np.testing.assert_almost_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((745.1 / 1000 * 2.6 / 1000 / 60)**2 +
                        (12.3 / 1000 / 60 * 53.9 / 1000)**2),
                np.sqrt((496.13 / 1000 * 5.4 / 1000 / 60)**2 +
                        (54.3 / 1000 / 60 * 24.75 / 1000)**2),
                np.sqrt((120.54 / 1000 * 10.56 / 1000 / 60) **
                        2 + (91.3 / 1000 / 60 * 6.4 / 1000)**2),
            ]), decimal=7)

        a = variable(1.2, 'm/N', 0.15)
        b = variable(7.43, 'N/cm', 2.5)
        c = a * b

        self.assertAlmostEqual(c.value, 891.6)
        self.assertTrue(c._unitObject == unit('1'))
        self.assertAlmostEqual(c.uncert, 320.032970958)

    def test_divide(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm',
                         uncert=[53.9, 24.75, 6.4])

        C = A / B
        self.assertAlmostEqual(C.value, 12.3 / 745.1)
        self.assertTrue(C._unitObject == unit('L/min-m'))
        self.assertAlmostEqual(C.uncert, np.sqrt(
            (1 / 745.1 * 2.6)**2 + (12.3 / (745.1**2) * 53.9)**2))

        C.convert('m3/h-mm')
        self.assertAlmostEqual(C.value, 12.3 / 745.1 / 1000 * 60 / 1000)
        self.assertTrue(C._unitObject == unit('m3/h-mm'))
        self.assertAlmostEqual(C.uncert, np.sqrt(
            (1 / (745.1 * 1000) * 2.6 / 1000 * 60)**2 + (12.3 / ((745.1)**2) * 53.9 / 1000 * 60 / 1000)**2))

        C_vec = A_vec / B_vec
        np.testing.assert_array_equal(C_vec.value, np.array(
            [12.3 / 745.1, 54.3 / 496.13, 91.3 / 120.54]))
        self.assertTrue(C_vec._unitObject == unit('L/min-m'))
        np.testing.assert_array_almost_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((1 / 745.1 * 2.6)**2 + (12.3 / (745.1)**2 * 53.9)**2),
                np.sqrt((1 / 496.13 * 5.4)**2 +
                        (54.3 / (496.13)**2 * 24.75)**2),
                np.sqrt((1 / 120.54 * 10.56)**2 +
                        (91.3 / (120.54)**2 * 6.4)**2),
            ]))

        C_vec.convert('m3 / h -mm')
        np.testing.assert_almost_equal(C_vec.value, np.array(
            [12.3 / 745.1, 54.3 / 496.13, 91.3 / 120.54]) / 1000 * 60 / 1000)
        self.assertTrue(C_vec._unitObject == unit('m3/h-mm'))
        np.testing.assert_almost_equal(
            C_vec.uncert,
            np.array([
                np.sqrt((1 / 745.1 * 2.6 / 1000 * 60 / 1000)**2 +
                        (12.3 / (745.1)**2 * 53.9 / 1000 * 60 / 1000)**2),
                np.sqrt((1 / 496.13 * 5.4 / 1000 * 60 / 1000)**2 +
                        (54.3 / (496.13)**2 * 24.75 / 1000 * 60 / 1000)**2),
                np.sqrt((1 / 120.54 * 10.56 / 1000 * 60 / 1000)**2 +
                        (91.3 / (120.54)**2 * 6.4 / 1000 * 60 / 1000)**2),
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
        B_vec = variable([745.1, 496.13, 120.54], 'm-K',
                         uncert=[53.9, 24.75, 6.4])
        C = A + B
        C_vec = A_vec + B_vec

    def test_sub_unit_order(self):
        A = variable(10, 'm-K')
        B = variable(3, 'K-m')
        A_vec = variable([12.3, 54.3, 91.3], 'K-m', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm-K',
                         uncert=[53.9, 24.75, 6.4])
        C = A - B
        C_vec = A_vec - B_vec

    def test_pow(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)
        C = variable(745.1, '1', uncert=53.9)
        D = variable(0.34, '1', uncert=0.01)

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm',
                         uncert=[53.9, 24.75, 6.4])
        C_vec = variable([745.1, 496.13, 120.54], '1',
                         uncert=[53.9, 24.75, 6.4])
        D_vec = variable([0.34, 0.64, 0.87], '1', uncert=[0.01, 0.084, 0.12])

        with self.assertRaises(Exception) as context:
            A ** B
        self.assertTrue(
            'The exponent can not have a unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec ** B_vec
        self.assertTrue(
            'The exponent can not have a unit' in str(context.exception))

        E = C**D
        self.assertAlmostEqual(E.value, 745.1**0.34)
        self.assertEqual(E.unit, '1')
        self.assertAlmostEqual(E.uncert, np.sqrt(
            (0.34 * 745.1**(0.34 - 1) * 53.9)**2 + (745.1**0.34 * np.log(745.1) * 0.01)**2))

        E_vec = C_vec**D_vec
        np.testing.assert_equal(
            E_vec.value, [745.1 ** 0.34, 496.13**0.64, 120.54**0.87])
        self.assertEqual(E_vec.unit, '1')
        self.assertAlmostEqual(E_vec.uncert[0], np.sqrt(
            (0.34 * 745.1**(0.34 - 1) * 53.9)**2 + (745.1**0.34 * np.log(745.1) * 0.01)**2))

        F = A**2
        self.assertAlmostEqual(F.value, (12.3)**2)
        self.assertEqual(F.unit, 'L2/min2')
        self.assertAlmostEqual(F.uncert, np.sqrt((2 * 12.3**(2 - 1) * 2.6)**2))

        F.convert('m6/s2')
        self.assertAlmostEqual(F.value, (12.3 / 1000 / 60)**2)
        self.assertEqual(F.unit, 'm6/s2')
        self.assertAlmostEqual(F.uncert, np.sqrt(
            (2 * (12.3 / 1000 / 60)**(2 - 1) * 2.6 / 1000 / 60)**2))

        F_vec = A_vec**2
        np.testing.assert_array_almost_equal(
            F_vec.value, np.array([(12.3)**2, 54.3**2, 91.3**2]))
        self.assertEqual(F_vec.unit, 'L2/min2')
        np.testing.assert_array_almost_equal(
            F_vec.uncert,
            np.array([
                np.sqrt((2 * 12.3**(2 - 1) * 2.6)**2),
                np.sqrt((2 * 54.3**(2 - 1) * 5.4)**2),
                np.sqrt((2 * 91.3**(2 - 1) * 10.56)**2)
            ]))

        F_vec.convert('m6 / s2')
        np.testing.assert_array_almost_equal(F_vec.value, np.array(
            [(12.3 / 1000 / 60)**2, (54.3 / 1000 / 60)**2, (91.3 / 1000 / 60)**2]))
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
        self.assertAlmostEqual(G.uncert, np.sqrt(
            (2.54**0.34 * np.log(2.54) * 0.01)**2))

        G_vec = 2.54**D_vec
        np.testing.assert_equal(
            G_vec.value, [2.54**0.34, 2.54 ** 0.64, 2.54**0.87])
        self.assertEqual(G_vec.unit, '1')
        self.assertAlmostEqual(G_vec.uncert[0], np.sqrt(
            (2.54**0.34 * np.log(2.54) * 0.01)**2))

    def test_log(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        C = variable(745.1, '1', uncert=53.9)

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        C_vec = variable([745.1, 496.13, 120.54], '1',
                         uncert=[53.9, 24.75, 6.4])

        with self.assertRaises(Exception) as context:
            np.log(A)
        self.assertTrue('You can only take the natural log of a variable if it has no unit' in str(
            context.exception))

        with self.assertRaises(Exception) as context:
            np.log10(A)
        self.assertTrue('You can only take the base 10 log of a variable if it has no unit' in str(
            context.exception))

        with self.assertRaises(Exception) as context:
            np.log(A_vec)
        self.assertTrue('You can only take the natural log of a variable if it has no unit' in str(
            context.exception))

        with self.assertRaises(Exception) as context:
            np.log10(A_vec)
        self.assertTrue('You can only take the base 10 log of a variable if it has no unit' in str(
            context.exception))

        D = np.log(C)
        self.assertAlmostEqual(D.value, np.log(745.1))
        self.assertEqual(D.unit, '1')
        self.assertAlmostEqual(D.uncert, np.sqrt((1 / 745.1) * 53.9)**2)

        D_vec = np.log(C_vec)
        np.testing.assert_array_equal(D_vec.value, np.array(
            [np.log(745.1), np.log(496.13), np.log(120.54)]))
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
        self.assertAlmostEqual(E.uncert, np.sqrt(
            (1 / (745.1 * np.log10(745.1))) * 53.9)**2)

        E_vec = np.log10(C_vec)
        np.testing.assert_array_equal(E_vec.value, np.array(
            [np.log10(745.1), np.log10(496.13), np.log10(120.54)]))
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
        self.assertTrue(
            'The exponent can not have a unit' in str(context.exception))

        c_vec = np.exp(C_vec)
        np.testing.assert_equal(
            c_vec.value, [np.e**12.3, np.e**54.3, np.e**91.3])
        self.assertEqual(c_vec.unit, '1')
        self.assertEqual(c_vec.uncert[0], np.sqrt(
            (np.e**12.3 * np.log(np.e) * 2.6)**2))

        D = np.exp(C)
        self.assertAlmostEqual(D.value, np.e**12.3)
        self.assertEqual(D.unit, '1')
        self.assertAlmostEqual(D.uncert, np.sqrt(
            (np.e**12.3 * np.log(np.e) * 5.39)**2))

        with self.assertRaises(Exception) as context:
            np.exp(A_vec)
        self.assertTrue(
            'The exponent can not have a unit' in str(context.exception))

    def testIndex(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])

        with self.assertRaises(Exception) as context:
            A[0]
        self.assertTrue(
            "'scalarVariable' object is not subscriptable" in str(context.exception))

        with self.assertRaises(Exception) as context:
            A[1]
        self.assertTrue(
            "'scalarVariable' object is not subscriptable" in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec[23]
        self.assertTrue('list index out of range' in str(context.exception))

        a_vec = A_vec[0]
        self.assertEqual(a_vec.value, 12.3)
        self.assertEqual(a_vec.unit, 'L/min')
        self.assertEqual(a_vec.uncert, 2.6)

        a_vec = A_vec[0:2]
        np.testing.assert_array_equal(a_vec.value, [12.3, 54.3])
        self.assertEqual(a_vec.unit, 'L/min')
        np.testing.assert_array_equal(a_vec.uncert, [2.6, 5.4])

        a_vec = A_vec[1:3]
        np.testing.assert_array_equal(a_vec.value, [54.3, 91.3])
        self.assertEqual(a_vec.unit, 'L/min')
        np.testing.assert_array_equal(a_vec.uncert, [5.4, 10.56])

        a_vec = A_vec[[0, 2]]
        np.testing.assert_array_equal(a_vec.value, [12.3, 91.3])
        self.assertEqual(a_vec.unit, 'L/min')
        np.testing.assert_array_equal(a_vec.uncert, [2.6, 10.56])

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
        B_vec = variable([745.1, 496.13, 120.54], 'L/min',
                         uncert=[53.9, 24.75, 6.4])

        A_vec += B_vec
        np.testing.assert_almost_equal(A_vec.value, np.array(
            [12.3 + 745.1, 54.3 + 496.13, 91.3 + 120.54]))
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
        np.testing.assert_almost_equal(A_vec.value, np.array(
            [12.3 + 12.3, 54.3 + 12.3, 91.3 + 12.3]))
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
        B_vec = variable([745.1, 496.13, 120.54], 'm',
                         uncert=[53.9, 24.75, 6.4])

        with self.assertRaises(Exception) as context:
            A += B
        self.assertTrue(
            'You tried to add a variable in [L/min] to a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B += A
        self.assertTrue(
            'You tried to add a variable in [m] to a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec += B_vec
        self.assertTrue(
            'You tried to add a variable in [L/min] to a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B_vec += A_vec
        self.assertTrue(
            'You tried to add a variable in [m] to a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec += B
        self.assertTrue(
            'You tried to add a variable in [L/min] to a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B_vec += A
        self.assertTrue(
            'You tried to add a variable in [m] to a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A += B_vec
        self.assertTrue(
            'You tried to add a variable in [L/min] to a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B += A_vec
        self.assertTrue(
            'You tried to add a variable in [m] to a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

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
        B_vec = variable([745.1, 496.13, 120.54], 'L/min',
                         uncert=[53.9, 24.75, 6.4])

        A_vec -= B_vec
        np.testing.assert_almost_equal(A_vec.value, np.array(
            [12.3 - 745.1, 54.3 - 496.13, 91.3 - 120.54]))
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
        np.testing.assert_almost_equal(A_vec.value, np.array(
            [12.3 - 12.3, 54.3 - 12.3, 91.3 - 12.3]))
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
        B_vec = variable([745.1, 496.13, 120.54], 'm',
                         uncert=[53.9, 24.75, 6.4])

        with self.assertRaises(Exception) as context:
            A -= B
        self.assertTrue(
            'You tried to subtract a variable in [m] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B -= A
        self.assertTrue(
            'You tried to subtract a variable in [L/min] from a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec -= B_vec
        self.assertTrue(
            'You tried to subtract a variable in [m] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B_vec -= A_vec
        self.assertTrue(
            'You tried to subtract a variable in [L/min] from a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec -= B
        self.assertTrue(
            'You tried to subtract a variable in [m] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B_vec -= A
        self.assertTrue(
            'You tried to subtract a variable in [L/min] from a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            A -= B_vec
        self.assertTrue(
            'You tried to subtract a variable in [m] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            B -= A_vec
        self.assertTrue(
            'You tried to subtract a variable in [L/min] from a variable in [m], but the units do not have the same SI base unit' in str(context.exception))

    def testMultiEqual(self):
        A = variable(12.3, 'L/min', uncert=2.6)
        B = variable(745.1, 'm', uncert=53.9)

        A *= B
        self.assertAlmostEqual(A.value, 12.3 * 745.1)
        self.assertTrue(A._unitObject, unit('L-m/min'))
        self.assertAlmostEqual(A.uncert, np.sqrt(
            (745.1 * 2.6)**2 + (12.3 * 53.9)**2))

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
        B_vec = variable([745.1, 496.13, 120.54], 'm',
                         uncert=[53.9, 24.75, 6.4])

        A_vec *= B_vec
        np.testing.assert_array_almost_equal(A_vec.value, np.array(
            [12.3 * 745.1, 54.3 * 496.13, 91.3 * 120.54]))
        self.assertTrue(A_vec._unitObject, unit('L-m/min'))
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
        np.testing.assert_array_almost_equal(
            A_vec.value, np.array([12.3 * 12.3, 54.3 * 12.3, 91.3 * 12.3]))
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
        self.assertTrue(A._unitObject == unit('L/min-m'))
        self.assertAlmostEqual(A.uncert, np.sqrt(
            (1 / 745.1 * 2.6)**2 + (12.3 / (745.1**2) * 53.9)**2))

        A = variable(12.3, 'L/min', uncert=2.6)
        A /= 2
        self.assertAlmostEqual(A.value, 12.3 / 2)
        self.assertTrue(A._unitObject == unit('L/min'))
        self.assertAlmostEqual(A.uncert, np.sqrt((1 / 2 * 2.6)**2))

        A = variable(12.3, 'L/min', uncert=2.6)
        B = 2
        B /= A
        self.assertAlmostEqual(B.value, 2 / 12.3)
        self.assertTrue(B._unitObject == unit('min/L'))
        self.assertAlmostEqual(B.uncert, np.sqrt((2 / (12.3**2) * 2.6)**2))

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        B_vec = variable([745.1, 496.13, 120.54], 'm',
                         uncert=[53.9, 24.75, 6.4])

        A_vec /= B_vec
        np.testing.assert_array_almost_equal(A_vec.value, np.array(
            [12.3 / 745.1, 54.3 / 496.13, 91.3 / 120.54]))
        self.assertTrue(A_vec._unitObject == unit('L/min-m'))
        np.testing.assert_array_almost_equal(
            A_vec.uncert,
            np.array([
                np.sqrt((1 / 745.1 * 2.6)**2 + (12.3 / (745.1**2) * 53.9)**2),
                np.sqrt((1 / 496.13 * 5.4)**2 +
                        (54.3 / (496.13**2) * 24.75)**2),
                np.sqrt((1 / 120.54 * 10.56)**2 +
                        (91.3 / (120.54**2) * 6.4)**2),
            ]))

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        A = variable(12.3, 'L/min', uncert=2.6)
        A_vec /= A
        np.testing.assert_array_almost_equal(
            A_vec.value, np.array([12.3 / 12.3, 54.3 / 12.3, 91.3 / 12.3]))
        self.assertTrue(A_vec._unitObject, unit('1'))
        np.testing.assert_array_almost_equal(
            A_vec.uncert,
            np.array([
                np.sqrt((1 / 12.3 * 2.6)**2 + (12.3 / (12.3**2) * 2.6)**2),
                np.sqrt((1 / 12.3 * 5.4)**2 + (54.3 / (12.3**2) * 2.6)**2),
                np.sqrt((1 / 12.3 * 10.56)**2 + (91.3 / (12.3**2) * 2.6)**2),
            ]))

    def testPrintValueAndUncertScalar(self):
        A = variable(123456789 * 10**(0), 'm',
                     uncert=123456789 * 10**(-2))
        self.assertEqual(str(A), '123000000 +/- 1000000 [m]')

        A = variable(123456789 * 10**(-2), 'm',
                     uncert=123456789 * 10**(-4))
        self.assertEqual(str(A), '1230000 +/- 10000 [m]')

        A = variable(123456789 * 10**(-4), 'm',
                     uncert=123456789 * 10**(-6))
        self.assertEqual(str(A), '12300 +/- 100 [m]')

        A = variable(123456789 * 10**(-6), 'm',
                     uncert=123456789 * 10**(-8))
        self.assertEqual(str(A), '123 +/- 1 [m]')

        A = variable(123456789 * 10**(-7), 'm',
                     uncert=123456789 * 10**(-9))
        self.assertEqual(str(A), '12.3 +/- 0.1 [m]')

        A = variable(123456789 * 10**(-8), 'm',
                     uncert=123456789 * 10**(-10))
        self.assertEqual(str(A), '1.23 +/- 0.01 [m]')

        A = variable(123456789 * 10**(-9), 'm',
                     uncert=123456789 * 10**(-11))
        self.assertEqual(str(A), '0.123 +/- 0.001 [m]')

        A = variable(123456789 * 10**(-10), 'm',
                     uncert=123456789 * 10**(-12))
        self.assertEqual(str(A), '0.0123 +/- 0.0001 [m]')

        A = variable(123456789 * 10**(-12), 'm',
                     uncert=123456789 * 10**(-14))
        self.assertEqual(str(A), '0.000123 +/- 1e-06 [m]')

        A = variable(123456789 * 10**(-14), 'm',
                     uncert=123456789 * 10**(-16))
        self.assertEqual(str(A), '0.00000123 +/- 1e-08 [m]')

        A = variable(123456789 * 10**(-16), 'm',
                     uncert=123456789 * 10**(-18))
        self.assertEqual(str(A), '0.0000000123 +/- 1e-10 [m]')

        A = variable(10.0, 'm', uncert=0.1)
        self.assertEqual(str(A), '10.0 +/- 0.1 [m]')

        A = variable(102.59573439096775, 'm', uncert=0.94)
        self.assertEqual(str(A), '102.6 +/- 0.9 [m]')

        A = variable(102.59573439096775, 'm', uncert=0.96)
        self.assertEqual(str(A), '103 +/- 1 [m]')

        A = variable(102.59573439096775, 'm', uncert=0.951)
        self.assertEqual(str(A), '103 +/- 1 [m]')

    def testPrintValueScalar(self):
        A = variable(123456789 * 10**(0), 'm')
        self.assertEqual(str(A), f'{str(float(123456789 * 10**(0)))} [m]')

        A = variable(123456789 * 10**(-2), 'm')
        self.assertEqual(str(A), f'{str(float(123456789 * 10**(-2)))} [m]')

        A = variable(123456789 * 10**(-4), 'm')
        self.assertEqual(str(A), f'{str(float(123456789 * 10**(-4)))} [m]')

        A = variable(123456789 * 10**(-6), 'm')
        self.assertEqual(str(A), f'{str(float(123456789 * 10**(-6)))} [m]')

        A = variable(123456789 * 10**(-7), 'm')
        self.assertEqual(str(A), f'{str(float(123456789 * 10**(-7)))} [m]')

        A = variable(123456789 * 10**(-8), 'm')
        self.assertEqual(str(A), f'{str(float(123456789 * 10**(-8)))} [m]')

        A = variable(123456789 * 10**(-9), 'm')
        self.assertEqual(str(A), f'{str(float(123456789 * 10**(-9)))} [m]')

        A = variable(123456789 * 10**(-10), 'm')
        self.assertEqual(str(A), f'{str(float(123456789 * 10**(-10)))} [m]')

        A = variable(123456789 * 10**(-12), 'm')
        self.assertEqual(str(A), f'{str(float(123456789 * 10**(-12)))} [m]')

        A = variable(123456789 * 10**(-14), 'm')
        self.assertEqual(str(A), f'{str(float(123456789 * 10**(-14)))} [m]')

        A = variable(123456789 * 10**(-16), 'm')
        self.assertEqual(str(A), f'{str(float(123456789 * 10**(-16)))} [m]')

    def testRoot(self):
        from random import uniform
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

        A = variable(10, 'L2/m')
        a = np.sqrt(A)
        self.assertEqual(a.value, np.sqrt(10))
        self.assertEqual(a.unit, 'L/m0.5')
        
        dP = variable(10, 'Pa')
        rho = variable(2.5, 'kg/m3')
        v = np.sqrt(2 * dP / rho)
        self.assertEqual(v.value, np.sqrt(2 * 10 / 2.5))

        dP = variable(1, 'bar')
        rho = variable(2.5, 'kg/L')
        v = np.sqrt(2 * dP / rho)
        
        self.assertEqual(v.value, np.sqrt(2 * 1 / 2.5))
        v.convert('m/s')
        self.assertAlmostEqual(v.value, 8.944271909999158554072096194 )

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
        self.assertEqual(str(A), '0.1 +/- 0.7 [L/min]')

        A = variable(0.9, 'L/min', 3)
        self.assertEqual(str(A), '1 +/- 3 [L/min]')

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
        B_vec = variable([745.1, 496.13, 120.54], 'm',
                         uncert=[53.9, 24.75, 6.4])
        C_vec = variable([745.1, 496.13, 120.54], '1',
                         uncert=[53.9, 24.75, 6.4])
        D_vec = variable([0.34, 0.64, 0.87], '1', uncert=[0.01, 0.084, 0.12])

        with self.assertRaises(Exception) as context:
            2**A
        self.assertTrue(
            'The exponent can not have a unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            2**B
        self.assertTrue(
            'The exponent can not have a unit' in str(context.exception))

        c = 2**C
        self.assertEqual(c.value, 2**74.51)
        self.assertEqual(c.unit, '1')
        self.assertEqual(c.uncert, np.sqrt(
            (2**74.51 * np.log(2) * 5.39)**2 + (74.51 * 2**(74.51 - 1) * 0)**2))

        d = 2**D
        self.assertEqual(d.value, 2**0.34)
        self.assertEqual(d.unit, '1')
        self.assertEqual(d.uncert, np.sqrt(
            (2**0.34 * np.log(2) * 0.01)**2 + (0.34 * 2**(0.34 - 1) * 0)**2))

        with self.assertRaises(Exception) as context:
            2**A_vec
        self.assertTrue(
            'The exponent can not have a unit' in str(context.exception))

        with self.assertRaises(Exception) as context:
            2**B_vec
        self.assertTrue(
            'The exponent can not have a unit' in str(context.exception))

        d_vec = 2**D_vec
        np.testing.assert_equal(d_vec.value, [2**0.34, 2**0.64, 2**0.87])
        self.assertEqual(d_vec.unit, '1')
        self.assertEqual(d_vec.uncert[0], np.sqrt(
            (2**0.34 * np.log(2) * 0.01)**2 + (0.34 * 2**(0.34 - 1) * 0)**2))

    def testPrettyPrint(self):
        a = variable(12.3, 'm')
        b = variable(12.3, 'm', 2.5)
        c = variable([12.3, 56.2], 'm')
        d = variable([12.3, 56.2], 'm', [2.5, 7.3])
        e = variable(12.3, 'DELTAC')
        
        self.assertEqual(a.__str__(pretty=False), '12.3 [m]')
        self.assertEqual(b.__str__(pretty=False), '12 +/- 2 [m]')
        self.assertEqual(c.__str__(pretty=False), '[12.3, 56.2] [m]')
        self.assertEqual(d.__str__(pretty=False), '[12, 56] +/- [2, 7] [m]')
        self.assertEqual(e.__str__(pretty=False), '12.3 [DELTAC]')

        self.assertEqual(a.__str__(pretty=True), r'12.3\ \left [m\right ]')
        self.assertEqual(b.__str__(pretty=True), r'12 \pm 2\ \left [m\right ]')
        self.assertEqual(c.__str__(pretty=True), r'[12.3, 56.2]\ \left [m\right ]')
        self.assertEqual(d.__str__(pretty=True), r'[12, 56] \pm [2, 7]\ \left [m\right ]')
        self.assertEqual(e.__str__(pretty=True), r'12.3\ \left [\Delta C\right ]')
        
        a = variable(23.1, 'L/min')
        b = variable(0.83, 'm3/h')
        c = a - b
        self.assertTrue(c.unit, 'm3/s')
        self.assertEqual(c.__str__(pretty=True), rf'{23.1 / 1000 / 60 - 0.83 / 3600}\ \left [\frac{{m^{{3}}}}{{s}}\right ]')

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

    def testArgMax(self):

        A = variable(10, 'm', 2.3)
        A = np.argmax(A)
        self.assertEqual(A, 0)

        A = variable(10, 'm')
        A = np.argmax(A)
        self.assertEqual(A, 0)

        A = variable([10, 15.7], 'm', [2.3, 5.6])
        A = np.argmax(A)
        self.assertEqual(A, 1)

        A = variable([10, 15.7], 'm')
        A = np.argmax(A)
        self.assertEqual(A, 1)

    def testArgMin(self):

        A = variable(10, 'm', 2.3)
        A = np.argmin(A)
        self.assertEqual(A, 0)

        A = variable(10, 'm')
        A = np.argmin(A)
        self.assertEqual(A, 0)

        A = variable([10, 15.7], 'm', [2.3, 5.6])
        A = np.argmin(A)
        self.assertEqual(A, 0)

        A = variable([10, 15.7], 'm')
        A = np.argmin(A)
        self.assertEqual(A, 0)

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
        self.assertEqual(A.uncert, np.sqrt(
            (1 / 2 * 2.3) ** 2 + (1 / 2 * 5.6) ** 2))

        A = variable([10, 15.7], 'm')
        A = np.mean(A)
        self.assertEqual(A.value, (10 + 15.7) / 2)
        self.assertEqual(A.unit, 'm')
        self.assertEqual(A.uncert, 0)
        
        A = variable(10, 'm', 1)
        B = variable(12, 'm', 2)
        C = np.mean([A,B])
        self.assertEqual(C.value, (10+12) / 2)
        self.assertEqual(C.unit, 'm')
        self.assertEqual(C.uncert, np.sqrt((1 * 1/2)**2 + (2 * 1/2)**2))

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
        self.assertAlmostEqual(b.uncert, np.sqrt(
            ((1 * np.pi / 180) * (np.cos(75 * np.pi / 180)))**2))

        a = variable(75, 'deg', 1)
        b = np.cos(a)
        self.assertAlmostEqual(b.value, 0.2588190451)
        self.assertEqual(b.unit, '1')
        self.assertAlmostEqual(b.uncert, np.sqrt(
            ((1 * np.pi / 180) * (-np.sin(75 * np.pi / 180)))**2))

        a = variable(75, 'deg', 1)
        b = np.tan(a)
        self.assertAlmostEqual(b.value, 3.73205080757)
        self.assertEqual(b.unit, '1')
        self.assertAlmostEqual(b.uncert, np.sqrt(
            ((1 * np.pi / 180) * (2 / (np.cos(2 * 75 * np.pi / 180) + 1)))**2))

        a = variable(0.367, 'rad', 0.0796)
        b = np.sin(a)
        self.assertAlmostEqual(b.value, 0.35881682685)
        self.assertEqual(b.unit, '1')
        self.assertAlmostEqual(b.uncert, np.sqrt(
            ((0.0796) * (np.cos(0.367)))**2))

        a = variable(0.367, 'rad', 0.0796)
        b = np.cos(a)
        self.assertAlmostEqual(b.value, 0.9334079948)
        self.assertEqual(b.unit, '1')
        self.assertAlmostEqual(b.uncert, np.sqrt(
            ((0.0796) * (-np.sin(0.367)))**2))

        a = variable(0.367, 'rad', 0.0796)
        b = np.tan(a)
        self.assertAlmostEqual(b.value, 0.38441584907)
        self.assertEqual(b.unit, '1')
        self.assertAlmostEqual(b.uncert, np.sqrt(
            ((0.0796) * (2 / (np.cos(2 * 0.367) + 1)))**2))

    def testProductRule(self):

        a = variable(23, 'deg', 2)
        b = np.sin(a)
        c = a * b
        val = 23
        unc = 2
        self.assertEqual(c.value, val * np.sin(np.pi / 180 * val))
        self.assertEqual(c.unit, 'deg')
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (unc * (np.sin(np.pi / 180 * val) + (np.pi / 180 * val) * np.cos(np.pi / 180 * val)))**2))

        a = variable(23, 'deg', 2)
        a.convert('rad')
        b = np.sin(a)
        c = a * b
        val = np.pi / 180 * 23
        unc = np.pi / 180 * 2
        self.assertEqual(c.value, val * np.sin(val))
        self.assertEqual(c.unit, 'rad')
        self.assertEqual(c.uncert, np.sqrt(
            (unc * (np.sin(val) + val * np.cos(val)))**2))

        a = variable(23, 'deg', 2)
        b = np.sin(a)
        a.convert('rad')
        c = a * b
        val = np.pi / 180 * 23
        unc = np.pi / 180 * 2
        self.assertAlmostEqual(c.value, val * np.sin(val))
        self.assertEqual(c.unit, 'rad')
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (unc * (np.sin(val) + val * np.cos(val)))**2))

        a = variable(np.pi / 180 * 23, 'rad', np.pi / 180 * 2)
        b = np.sin(a)
        a.convert('deg')
        c = a * b
        val = 23
        unc = 2
        self.assertEqual(c.value, val * np.sin(np.pi / 180 * val))
        self.assertEqual(c.unit, 'deg')
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (unc * (np.sin(np.pi / 180 * val) + (np.pi / 180 * val) * np.cos(np.pi / 180 * val)))**2))

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
        self.assertAlmostEqual(b.uncert, np.sqrt(
            (1.5 * 0.16806722689075630252100840336134 * 200)**2))

        b *= np.sin(a * variable(1, 'rad-min/L'))
        self.assertAlmostEqual(
            b.value, -2935.453099878973383976532508069948132551783965504369163751)
        self.assertEqual(b.unit, 'L2/min2')
        self.assertAlmostEqual(b.uncert, np.sqrt(
            (1.5 * 2 * 200 * (2 * np.sin(200) + 200 * np.cos(200)) / 23.8)**2))

        a /= variable(100, 'L/min')
        a.convert('')
        b /= np.exp(a)
        self.assertAlmostEqual(
            b.value, -397.2703766999135885608809478456258749790006977070134430245)
        self.assertEqual(b.unit, 'L2/min2')
        self.assertAlmostEqual(b.uncert, np.sqrt((1.5 * 2 * 200 * np.exp(-200 / 100) * (
            (2 * 100 - 200) * np.sin(200) + 100 * 200 * np.cos(200)) / (23.8 * 100))**2))

        a = variable(37, 'deg', 2.3)
        b = a**2 * np.cos(3 * a * np.sin(a))
        self.assertAlmostEqual(b.value, 539.274244145)
        self.assertEqual(b.unit, 'deg2')
        self.assertAlmostEqual(b.uncert, np.sqrt(
            (2.3 * (-44.47986837334018052364896281900061654705050149285482386191581710))**2))

        a = variable(37, 'deg', 2.3)
        b = np.cos(3 * a * np.sin(a))
        a.convert('rad')
        b *= a**2
        self.assertAlmostEqual(b.value, 0.1642723288)
        self.assertEqual(b.unit, 'rad2')
        self.assertAlmostEqual(b.uncert, np.sqrt(
            (np.pi / 180 * 2.3 * (-0.776320153968480543428298272676994570718859842395105372525387644))**2))

        a = variable(2.3, '', 0.11)
        b = variable(1.5, 'rad', 0.89)
        d = np.exp(a**2) * np.cos(b * np.tan(b / 9) + 17.5)
        self.assertAlmostEqual(
            d.value, 90.459733187853914019107237427714051178688422865118118182659)
        self.assertEqual(d.unit, '1')
        dd_da = 416.11477266412800448789329216748463542196674517954334364023
        dd_db = 59.945989031664355557893562375983977678359283356335782472980
        self.assertAlmostEqual(d.uncert, np.sqrt(
            (dd_da * 0.11)**2 + (dd_db * 0.89)**2))

        r"""
        e = \sum^{\infty}_{n=0} 1/(n!)
        b = \sum^{\infty}_{n=0} 1/(n!) * a = e*a
        \frac{\partial b}{\partial a} = e
        """
        a = variable(2.3, 'L/min', 0.0237)
        b = variable(0, 'L/min')
        for i in range(15):
            b += 1 / math.factorial(i) * a
        self.assertAlmostEqual(b.value, np.e * 2.3)
        self.assertEqual(b.unit, 'L/min')
        self.assertAlmostEqual(b.uncert, np.sqrt((np.e * 0.0237)**2))

        b = variable(64.73976386561031, 'dB')
        a0 = variable(11.0625, '', 0.4337698122276376)
        a0.convert('dB')
        c0 = a0 + b
        
        a1 = variable(11.0625, 'm2', 0.4337698122276376)
        a1 /= variable(1, 'm2') 
        a1.convert('dB')
        c1 = a1 + b
        self.assertEqual(c0, c1)
        
        a = variable([65,66], 'dB', [1,2])
        b = logarithmic.mean(a)
        c = variable(0.8, 'dB')
        d = b - c    
        
        aVal = 65
        bVal = 66
        ua = 1
        ub = 2
        cVal = 0.8
        uc = 0
        
        dVal = 10 * np.log10(10**(aVal/10) / 2 + 10**(bVal/10)/2) - cVal
        grada = 10**(aVal/10) / (10**(aVal/10) + 10**(bVal/10))
        gradb = 10**(bVal/10) / (10**(aVal/10) + 10**(bVal/10))
        gradc = -1 
        ud = np.sqrt((ua * grada)**2 + (ub * gradb)**2 + (uc * gradc)**2)
        
        self.assertAlmostEqual(d.value, dVal)
        self.assertEqual(d.unit, 'dB')
        self.assertAlmostEqual(d.uncert, ud)
        
        
        c = variable(1, 'cm/m', 0.1)
        d = np.sin(c)
        self.assertEqual(d.value, np.sin(1/100))
        self.assertEqual(d.unit, '1')
        self.assertEqual(d.uncert, np.sqrt((0.1 * 1/100 * np.cos(1/100 * 1 + 0))**2))
        
        c.convert('1')
        e = np.sin(c)
        self.assertEqual(e.value, np.sin(1/100))
        self.assertEqual(e.unit, '1')
        self.assertEqual(e.uncert, np.sqrt((0.1 * 1/100 * np.cos(1/100 * 1 + 0))**2))
        
    def testCovariance(self):
        a = variable(123, 'L/min', 9.7)
        b = variable(93, 'Pa', 1.2)
        a.addCovariance(b, 23, 'L-Pa/min')
        c = a * b
        self.assertEqual(c.value, 123 * 93)
        self.assertTrue(c._unitObject == unit('L-Pa/min'))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (123 * 1.2)**2 + (93 * 9.7)**2 + 2 * 93 * 123 * 23))

        a = variable(123, 'L/min', 9.7)
        b = variable(93, 'Pa', 1.2)
        a.addCovariance(b, 23, 'L-Pa/min')
        a.convert('m3/s')
        c = a * b
        self.assertEqual(c.value, 123 * 93 / 1000 / 60)
        self.assertTrue(c._unitObject == unit('m3-Pa/s'))
        self.assertAlmostEqual(c.uncert, np.sqrt((123 / 1000 / 60 * 1.2)**2 + (
            93 * 9.7 / 1000 / 60)**2 + 2 * 93 * 123 / 1000 / 60 * 23 / 1000 / 60))

        a = variable(123, 'L/min', 9.7)
        b = variable(93, 'Pa', 1.2)
        a.addCovariance(b, 23, 'm3-Pa/s')
        a.convert('m3/s')
        c = a * b
        self.assertEqual(c.value, 123 * 93 / 1000 / 60)
        self.assertTrue(c._unitObject == unit('m3-Pa/s'))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (123 / 1000 / 60 * 1.2)**2 + (93 * 9.7 / 1000 / 60)**2 + 2 * 93 * 123 / 1000 / 60 * 23))

        a = variable([1, 2, 3], 'L/min', [0.1, 0.2, 0.3])
        b = variable([93, 97, 102], 'Pa', [1.2, 2.4, 4.7])
        a.addCovariance(b, [2, 3, 4], 'L-Pa/min')
        c = a * b
        np.testing.assert_equal(c.value, [1*93, 2*97, 3*102])
        self.assertTrue(c._unitObject == unit('L-Pa/min'))
        dcda = np.array([93, 97, 102], dtype=float)
        dcdb = np.array([1, 2, 3], dtype=float)
        ua = np.array([0.1, 0.2, 0.3], dtype=float)
        ub = np.array([1.2, 2.4, 4.7], dtype=float)
        uab = np.array([2, 3, 4], dtype=float)

        uc = np.sqrt((dcda * ua)**2 + (dcdb * ub)**2 + 2 * dcda * dcdb * uab)
        np.testing.assert_array_almost_equal(c.uncert, uc)

    def testConvert(self):
        a = variable(1, 'km', 0.1)
        b = variable(1, 'm', 0.1)
        c = a * b
        self.assertEqual(c._unitObject.unitStrPretty, r'{km} \cdot {m}')
        c.convert('mm2')
        self.assertEqual(c._unitObject.unitStrPretty, r'mm^{2}')
        self.assertEqual(c.value, 1e9)
        self.assertEqual(c.unit, 'mm2')
        self.assertEqual(c.uncert,  np.sqrt(
            (1 * 1000 * 0.1 * 1000*1000)**2 + (1 * 1000*1000 * 0.1 * 1000)**2))

        a = variable([1,2,3], 'km', [0.1, 0.2, 0.3])
        b = variable([1,2,3], 'm', [0.1, 0.2, 0.3])
        c = a * b
        self.assertEqual(c._unitObject.unitStrPretty, r'{km} \cdot {m}')
        for elem in c:
            self.assertEqual(elem._unitObject.unitStrPretty, r'{km} \cdot {m}')
        c.convert('mm2')
        self.assertEqual(c._unitObject.unitStrPretty, r'mm^{2}')
        for elem in c:
            self.assertEqual(elem._unitObject.unitStrPretty, r'mm^{2}')
        

        air_flow = variable([0.551, 0.681, 0.817, 0.960, 1.099, 1.211], 'm3/s', [0.004, 0.003, 0.003, 0.002, 0.003, 0.004])
        rho = variable(1.2, 'kg/m3')

        mass_flow = air_flow * rho
        self.assertEqual(mass_flow._unitObject.unitStrPretty, r'{\frac{m^{3}}{s}} \cdot {\frac{kg}{m^{3}}}')
        mass_flow.convert('kg/s')
        self.assertEqual(mass_flow._unitObject.unitStrPretty, r'\frac{kg}{s}')


        diameter = variable(40, 'cm', 0.2)
        area = np.pi / 4 * diameter ** 2
        area.convert('m2')
        self.assertEqual(area.value, 0.12566370614359172953850573)
        self.assertEqual(area.unit, 'm2')
        self.assertEqual(area.uncert, np.sqrt(
            (2 * np.pi / 4 * 0.4 * 0.002)**2))

        time = variable([1, 2, 3], 's', [0.1, 0.2, 0.3])
        time.convert('min')
        np.testing.assert_array_equal(time.value, np.array([1, 2, 3]) / 60)
        self.assertEqual(time.unit, 'min')
        np.testing.assert_array_equal(
            time.uncert, np.array([0.1, 0.2, 0.3]) / 60)
        self.assertEqual(time[0].value, 1/60)
        self.assertEqual(time[1].value, 2/60)
        self.assertEqual(time[2].value, 3/60)
        self.assertEqual(time[0].unit, 'min')
        self.assertEqual(time[1].unit, 'min')
        self.assertEqual(time[2].unit, 'min')
        self.assertEqual(time[0].uncert, 0.1/60)
        self.assertEqual(time[1].uncert, 0.2/60)
        self.assertEqual(time[2].uncert, 0.3/60)

        t1 = variable([1, 2, 3], 'min')
        t1[1].convert('s')
        dt = variable(10, 'min')
        with self.assertRaises(Exception) as context:
            t1 + dt
        self.assertTrue("Some of the scalarvariables in [1.0, 120.0, 3.0] [min] did not have the unit [min] as they should. This could happen if the user has converted a scalarVaraible instead of the arrayVaraible." in str(
            context.exception))

        a = variable(19, 'dB', 1.2)
        a.convert('1')
        self.assertAlmostEqual(
            a.value, 79.432823472428150206591828283638793258896063)
        self.assertEqual(a.unit, '1')
        self.assertAlmostEqual(a.uncert, 10**(19/10 - 1) * np.log(10) * 1.2)
        a.convert('dB')
        self.assertEqual(a.value, 19)
        self.assertEqual(a.unit, 'dB')
        self.assertEqual(a.uncert, 1.2)

        a = variable(1.9, 'Np', 0.12)
        a.convert('1')
        self.assertAlmostEqual(
            a.value, 44.701184493300823037557828729065328038051563047543533720765133464)
        self.assertEqual(a.unit, '1')
        self.assertAlmostEqual(a.uncert, 2 * np.exp(2*1.9) * 0.12)
        a.convert('Np')
        self.assertEqual(a.value, 1.9)
        self.assertEqual(a.unit, 'Np')
        self.assertEqual(a.uncert, 0.12)

        a = variable(19, 'doct', 1.2)
        a.convert('1')
        self.assertAlmostEqual(
            a.value, 3.7321319661472296639253730645997686681089198574059761556019895419)
        self.assertEqual(a.unit, '1')
        self.assertAlmostEqual(
            a.uncert, 1/5 * 2**(19/10 - 1) * np.log(2) * 1.2)
        a.convert('doct')
        self.assertEqual(a.value, 19)
        self.assertEqual(a.unit, 'doct')
        self.assertEqual(a.uncert, 1.2)

        a = variable(19, 'ddec', 1.2)
        a.convert('1')
        self.assertAlmostEqual(
            a.value, 79.432823472428150206591828283638793258896063)
        self.assertEqual(a.unit, '1')
        self.assertAlmostEqual(a.uncert, 10**(19/10 - 1) * np.log(10) * 1.2)
        a.convert('ddec')
        self.assertEqual(a.value, 19)
        self.assertEqual(a.unit, 'ddec')
        self.assertEqual(a.uncert, 1.2)

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
            a < b
        self.assertTrue("operands could not be broadcast together with shapes (2,) (3,)" in str(
            context.exception))

        with self.assertRaises(Exception) as context:
            a <= b
        self.assertTrue("operands could not be broadcast together with shapes (2,) (3,)" in str(
            context.exception))

        with self.assertRaises(Exception) as context:
            a > b
        self.assertTrue("operands could not be broadcast together with shapes (2,) (3,)" in str(
            context.exception))

        with self.assertRaises(Exception) as context:
            a >= b
        self.assertTrue("operands could not be broadcast together with shapes (2,) (3,)" in str(
            context.exception))

        with self.assertRaises(Exception) as context:
            a == b
        self.assertTrue("operands could not be broadcast together with shapes (2,) (3,)" in str(
            context.exception))

        with self.assertRaises(Exception) as context:
            a != b
        self.assertTrue("operands could not be broadcast together with shapes (2,) (3,)" in str(
            context.exception))

        a = variable(1, 'm')
        b = variable(2, 'C')
        
        self.assertFalse(a < b)
        self.assertFalse(a <= b)
        self.assertFalse(a > b)
        self.assertFalse(a >= b)
        self.assertFalse(a == b)
        self.assertFalse(a != b)
       
        a = variable([1, 2, 3], 'm')
        b = variable([2, 3, 4], 'm')
        np.testing.assert_equal(a < b, [True, True, True])
        np.testing.assert_equal(a <= b, [True, True, True])
        np.testing.assert_equal(a > b, [False, False, False])
        np.testing.assert_equal(a >= b, [False, False, False])
        np.testing.assert_equal(a == b, [False, False, False])
        np.testing.assert_equal(a != b, [True, True, True])

        a = variable(10, 'L/min')
        b = variable(1, 'm3/h')
        np.testing.assert_equal(a > b, False)
        np.testing.assert_equal(a < b, True)
        np.testing.assert_equal(a >= b, False)
        np.testing.assert_equal(a <= b, True)
        np.testing.assert_equal(a == b, False)
        np.testing.assert_equal(a != b, True)

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
        self.assertTrue(
            "You can not set an element of [12, 12, 90] +/- [3, 3, 10] [L/min] with 45 +/- 1 [Pa] as they do not have the same unit" in str(context.exception))

        with self.assertRaises(Exception) as context:
            A_vec[0] = A_vec
        self.assertTrue(
            "You can only set an element with a scalar variable" in str(context.exception))

        with self.assertRaises(Exception) as context:
            A[0] = B
        self.assertTrue("'scalarVariable' object does not support item assignment" in str(
            context.exception))

        a = variable([1, 2, 3], 'm', [0.1, 0.2, 0.3])
        b = a**2
        a[1] = variable(5, 'm', 0.5)
        c = b * a

        a0 = variable(1, 'm', 0.1)
        a1 = variable(2, 'm', 0.2)
        a2 = variable(3, 'm', 0.3)
        b0 = a0**2
        b1 = a1**2
        b2 = a2**2
        a1 = variable(5, 'm', 0.5)
        c0 = b0 * a0
        c1 = b1 * a1
        c2 = b2 * a2

        self.assertEqual(c[0].value, c0.value)
        self.assertEqual(c[1].value, c1.value)
        self.assertEqual(c[2].value, c2.value)
        self.assertEqual(c.unit, c0.unit)
        self.assertEqual(c[0].uncert, c0.uncert)
        self.assertEqual(c[1].uncert, c1.uncert)
        self.assertEqual(c[2].uncert, c2.uncert)

        A = variable([1, 2, 3], 'L/min', [0.1, 0.2, 0.3])
        B = variable([93, 97, 102], 'Pa', [1.2, 2.4, 4.7])
        A.addCovariance(B, [2, 3, 4], 'L-Pa/min')
        C = A * B
        A[1] = variable(2.5, 'L/min', 0.25)
        C *= A

        a0 = variable(1, 'L/min', 0.1)
        b0 = variable(93, 'Pa', 1.2)
        a0.addCovariance(b0, 2, 'L-Pa/min')
        c0 = a0 * b0
        c0 *= a0

        a1 = variable(2, 'L/min', 0.2)
        b1 = variable(97, 'Pa', 2.4)
        a1.addCovariance(b1, 3, 'L-Pa/min')
        c1 = a1 * b1
        a11 = variable(2.5, 'L/min', 0.25)
        c1 *= a11

        a2 = variable(3, 'L/min', 0.3)
        b2 = variable(102, 'Pa', 4.7)
        a2.addCovariance(b2, 4, 'L-Pa/min')
        c2 = a2 * b2
        c2 *= a2

        np.testing.assert_almost_equal(C[0].value, c0.value)
        self.assertTrue(C._unitObject == unit(c0.unit))
        np.testing.assert_almost_equal(C[0].uncert, c0.uncert)

        np.testing.assert_almost_equal(C[1].value, c1.value)
        self.assertTrue(C._unitObject == unit(c0.unit))
        np.testing.assert_almost_equal(C[1].uncert, c1.uncert)

        np.testing.assert_almost_equal(C[2].value, c2.value)
        self.assertTrue(C._unitObject == unit(c0.unit))
        np.testing.assert_almost_equal(C[2].uncert, c2.uncert)

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
        self.assertTrue(
            "You can not set an element of [12, 54, 90, 12] +/- [3, 5, 10, 3] [L/min] with 45 +/- 1 [Pa] as they do not have the same unit" in str(context.exception))

        A_vec.append(A_vec)
        np.testing.assert_equal(A_vec.value, [12.3, 54.3, 91.3, 12.3] * 2)
        self.assertEqual(A_vec.unit, 'L/min')
        np.testing.assert_equal(A_vec.uncert, [2.6, 5.4, 10.56, 2.6] * 2)

        with self.assertRaises(Exception) as context:
            A.append(B)
        self.assertTrue(
            "'scalarVariable' object has no attribute 'append'" in str(context.exception))

        a = variable([1, 2, 3], 'm')
        b = variable([4, 5, 6], 'm')
        c = variable([10, 11, 12], 'Pa')
        d = variable([13, 14, 15], 'Pa')
        b.addCovariance(d, [0.1, 0.2, 0.3], 'm-Pa')
        a.append(b)
        c.append(d)
        e = a * c

        np.testing.assert_equal(e.value, np.array(
            [1, 2, 3, 4, 5, 6]) * np.array([10, 11, 12, 13, 14, 15]))
        self.assertTrue(e._unitObject == unit('m-Pa'))
        np.testing.assert_array_almost_equal(e.uncert, np.sqrt(
            2 * np.array([1, 2, 3, 4, 5, 6]) * np.array([10, 11, 12, 13, 14, 15]) * np.array([0, 0, 0, 0.1, 0.2, 0.3])))

        a = variable([1, 2, 3], 'm')
        b = variable([4, 5, 6], 'm')
        c = variable([10, 11, 12], 'Pa')
        d = variable([13, 14, 15], 'Pa')
        b.addCovariance(c, [0.1, 0.2, 0.3], 'm-Pa')
        a.append(b)
        c.append(d)
        d = a * c
        np.testing.assert_equal(d.value, np.array(
            [1, 2, 3, 4, 5, 6]) * np.array([10, 11, 12, 13, 14, 15]))
        self.assertTrue(d._unitObject == unit('m-Pa'))
        np.testing.assert_equal(d.uncert, np.array([0, 0, 0, 0, 0, 0]))

    def testAverageBel(self):
        a = variable([66.62, 68.91, 65.22, 63.86, 60.74, 63.36],
                     'dB', [1, 2, 3, 4, 5, 6])
        b = logarithmic.mean(a)
        self.assertAlmostEqual(b.value, 65.53976386561031)
        self.assertEqual(b.unit, 'dB')
        val = 10**(66.62/10) + 10**(68.91/10) + 10 ** (65.22/10) + \
            10**(63.86/10) + 10**(60.74/10) + 10**(63.36/10)
        grads = [
            (10**(66.62/10)) / val,
            (10**(68.91/10)) / val,
            (10**(65.22/10)) / val,
            (10**(63.86/10)) / val,
            (10**(60.74/10)) / val,
            (10**(63.36/10)) / val
        ]

        bUncert = 0
        for i in range(len(grads)):
            bUncert += (grads[i] * a.uncert[i])**2
        bUncert = np.sqrt(bUncert)

        self.assertAlmostEqual(b.uncert, bUncert)

    def testAddBel(self):
        a = variable(11, 'dB', 0.1)
        b = variable(19, 'dB', 1.2)
        c = logarithmic.add(a, b)
        self.assertEqual(
            c.value, 19.638920341433795986775635083534144311728776386508569289294)
        self.assertEqual(c.unit, 'dB')
        gradA = (10**(11/10)) / (10**(19/10) + 10**(11/10))
        gradB = (10**(19/10)) / (10**(19/10) + 10**(11/10))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(11, 'B', 0.1)
        b = variable(19, 'B', 1.2)
        c = logarithmic.add(a, b)
        self.assertEqual(
            c.value, 19.000000004342944797317794326113524021957351355250018803653068412)
        self.assertEqual(c.unit, 'B')
        gradA = (10**(11)) / (10**(19) + 10**(11))
        gradB = (10**(19)) / (10**(19) + 10**(11))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(1.1, 'B', 0.1)
        b = variable(19, 'dB', 1.2)
        c = logarithmic.add(a, b)
        self.assertAlmostEqual(
            c.value, 1.9638920341433795986775635083534144311728776386508569289294)
        self.assertEqual(c.unit, 'B')
        gradA = 10**1.1 / (10**(19/10) + 10**1.1)
        gradB = 10**(19/10-1) / (10**(19/10) + 10**1.1)
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

    def testSubtractBel(self):
        a = variable(11, 'dB', 0.1)
        b = variable(19, 'dB', 1.2)
        c = logarithmic.sub(b, a)
        self.assertEqual(
            c.value, 18.250596325673850704123951198937009709734608031896818185442)
        self.assertEqual(c.unit, 'dB')
        gradA = -(10**(11/10)) / (10**(19/10) - 10**(11/10))
        gradB = (10**(19/10)) / (10**(19/10) - 10**(11/10))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(11, 'B', 0.1)
        b = variable(19, 'B', 1.2)
        c = logarithmic.sub(b, a)
        self.assertEqual(
            c.value, 18.99999999565705515925275748356129104145734723683018994643533534)
        self.assertEqual(c.unit, 'B')
        gradA = -(10**(11)) / (10**(19) - 10**(11))
        gradB = (10**(19)) / (10**(19) - 10**(11))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(1.1, 'B', 0.1)
        b = variable(19, 'dB', 1.2)
        c = logarithmic.sub(b, a)
        self.assertAlmostEqual(
            c.value, 1.8250596325673850704123951198937009709734608031896818185442)
        self.assertEqual(c.unit, 'B')
        gradA = -10**(1.1+1) / (10**(19/10) - 10**1.1) / 10
        gradB = 10**(19/10) / (10**(19/10) - 10**1.1) / 10
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

    def testAddNeper(self):
        a = variable(1.1, 'Np', 1.2)
        b = variable(1.9, 'Np', 0.1)
        c = logarithmic.add(a, b)
        gradA = np.exp(2*1.1) / (np.exp(2*1.1) + np.exp(2*1.9))
        gradB = np.exp(2*1.9) / (np.exp(2*1.1) + np.exp(2*1.9))
        self.assertAlmostEqual(c.value, 1.9919503704442)
        self.assertEqual(c.unit, 'Np')
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * 1.2)**2 + (gradB * 0.1)**2))

        a = variable(11, 'dNp', 0.1)
        b = variable(19, 'dNp', 1.2)
        c = logarithmic.add(a, b)
        self.assertEqual(
            c.value, 19.919503704441694148537476059644226712824512622876065)
        self.assertEqual(c.unit, 'dNp')
        gradA = np.exp(11/5) / ((np.exp(11/5) + np.exp(19/5)))
        gradB = np.exp(19/5) / ((np.exp(11/5) + np.exp(19/5)))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(1.1, 'B', 0.2)
        b = variable(1.9, 'Np', 0.1)
        c = logarithmic.add(a, b)
        self.assertAlmostEqual(
            c.value, 2.0240668721868840255988083259625703882692927909254311205261)
        self.assertEqual(c.unit, 'Np')
        gradA = 2**(1.1-1) * 5 ** 1.1 * np.log(10) / \
            (10 ** 1.1 + np.exp(2*1.9))
        gradB = np.exp(2*1.9) / (10 ** 1.1 + np.exp(2*1.9))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

    def testSubtractNeper(self):
        a = variable(1.1, 'Np', 1.2)
        b = variable(1.9, 'Np', 0.1)
        c = logarithmic.sub(b, a)
        gradA = -np.exp(2*1.1) / (-np.exp(2*1.1) + np.exp(2*1.9))
        gradB = np.exp(2*1.9) / (-np.exp(2*1.1) + np.exp(2*1.9))
        self.assertAlmostEqual(
            c.value, 1.7872414933794011376350282497774605052189199203232206053472)
        self.assertEqual(c.unit, 'Np')
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * 1.2)**2 + (gradB * 0.1)**2))

        a = variable(11, 'dNp', 0.1)
        b = variable(19, 'dNp', 1.2)
        c = logarithmic.sub(b, a)
        self.assertEqual(
            c.value, 17.872414933794011376350282497774605052189199203232206053472)
        self.assertEqual(c.unit, 'dNp')
        gradA = -np.exp(11/5) / ((-np.exp(11/5) + np.exp(19/5)))
        gradB = np.exp(19/5) / ((-np.exp(11/5) + np.exp(19/5)))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(1.1, 'B', 0.2)
        b = variable(1.9, 'Np', 0.1)
        c = logarithmic.sub(b, a)
        self.assertAlmostEqual(
            c.value, 1.7346138119351200779277098049685724168821500468817797823749)
        self.assertEqual(c.unit, 'Np')
        gradA = - (2**(1.1-1) * 5**1.1 * np.log(10)) / \
            (np.exp(2 * 1.9) - 10**1.1)
        gradB = np.exp(2*1.9) / (-10**1.1 + np.exp(2*1.9))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

    def testAddOctave(self):
        a = variable(11, 'oct', 0.1)
        b = variable(19, 'oct', 1.2)
        c = logarithmic.add(a, b)
        self.assertEqual(
            c.value, 19.005624549193878106919859102674066601721109681538352035907295778)
        self.assertEqual(c.unit, 'oct')
        gradA = (2**(11)) / (2**(19) + 2**(11))
        gradB = (2**(19)) / (2**(19) + 2**(11))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(11, 'doct', 0.1)
        b = variable(19, 'doct', 1.2)
        c = logarithmic.add(a, b)
        self.assertEqual(
            c.value, 25.547555540454389776996741016863742223173494327085729223977)
        self.assertEqual(c.unit, 'doct')
        gradA = (2**(11/10)) / (2**(19/10) + 2**(11/10))
        gradB = (2**(19/10)) / (2**(19/10) + 2**(11/10))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(1.1, 'oct', 0.1)
        b = variable(19, 'doct', 1.2)
        c = logarithmic.add(a, b)
        self.assertAlmostEqual(
            c.value, 2.5547555540454389776996741016863742223173494327085729223977)
        self.assertEqual(c.unit, 'oct')
        gradA = 2**1.1 / (2**(19/10) + 2**1.1)
        gradB = 1 / (10*(2**(1.1 - 19/10) + 1))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

    def testSubtractOctave(self):
        a = variable(11, 'oct', 0.1)
        b = variable(19, 'oct', 1.2)
        c = logarithmic.sub(b, a)
        self.assertEqual(
            c.value, 18.994353436858857937578124384247611038635913083061882360637118114)
        self.assertEqual(c.unit, 'oct')
        gradA = -(2**(11)) / (2**(19) - 2**(11))
        gradB = (2**(19)) / (2**(19) - 2**(11))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(11, 'doct', 0.1)
        b = variable(19, 'doct', 1.2)
        c = logarithmic.sub(b, a)
        self.assertAlmostEqual(
            c.value, 6.677423233234791893663882614756435176500756327730140063397)
        self.assertEqual(c.unit, 'doct')
        gradA = -(2**(11/10)) / (2**(19/10) - 2**(11/10))
        gradB = (2**(19/10)) / (2**(19/10) - 2**(11/10))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(1.1, 'oct', 0.1)
        b = variable(19, 'doct', 1.2)
        c = logarithmic.sub(b, a)
        self.assertAlmostEqual(
            c.value, 0.6677423233234791893663882614756435176500756327730140063397)
        self.assertEqual(c.unit, 'oct')
        gradA = -5 * 2**(1.1+1) / (2**(19/10) - 2**1.1) / 10
        gradB = 2**(19/10) / (2**(19/10) - 2**1.1) / 10
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

    def testAddDecade(self):
        a = variable(11, 'ddec', 0.1)
        b = variable(19, 'ddec', 1.2)
        c = logarithmic.add(a, b)
        self.assertEqual(
            c.value, 19.638920341433795986775635083534144311728776386508569289294)
        self.assertEqual(c.unit, 'ddec')
        gradA = (10**(11/10)) / (10**(19/10) + 10**(11/10))
        gradB = (10**(19/10)) / (10**(19/10) + 10**(11/10))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(11, 'dec', 0.1)
        b = variable(19, 'dec', 1.2)
        c = logarithmic.add(a, b)
        self.assertEqual(
            c.value, 19.000000004342944797317794326113524021957351355250018803653068412)
        self.assertEqual(c.unit, 'dec')
        gradA = (10**(11)) / (10**(19) + 10**(11))
        gradB = (10**(19)) / (10**(19) + 10**(11))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(1.1, 'dec', 0.1)
        b = variable(19, 'ddec', 1.2)
        c = logarithmic.add(a, b)
        self.assertAlmostEqual(
            c.value, 1.9638920341433795986775635083534144311728776386508569289294)
        self.assertEqual(c.unit, 'dec')
        gradA = 10**1.1 / (10**(19/10) + 10**1.1)
        gradB = 10**(19/10-1) / (10**(19/10) + 10**1.1)
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

    def testSubtractDecade(self):
        a = variable(11, 'ddec', 0.1)
        b = variable(19, 'ddec', 1.2)
        c = logarithmic.sub(b, a)
        self.assertEqual(
            c.value, 18.250596325673850704123951198937009709734608031896818185442)
        self.assertEqual(c.unit, 'ddec')
        gradA = -(10**(11/10)) / (10**(19/10) - 10**(11/10))
        gradB = (10**(19/10)) / (10**(19/10) - 10**(11/10))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(11, 'dec', 0.1)
        b = variable(19, 'dec', 1.2)
        c = logarithmic.sub(b, a)
        self.assertEqual(
            c.value, 18.99999999565705515925275748356129104145734723683018994643533534)
        self.assertEqual(c.unit, 'dec')
        gradA = -(10**(11)) / (10**(19) - 10**(11))
        gradB = (10**(19)) / (10**(19) - 10**(11))
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

        a = variable(1.1, 'dec', 0.1)
        b = variable(19, 'ddec', 1.2)
        c = logarithmic.sub(b, a)
        self.assertAlmostEqual(
            c.value, 1.8250596325673850704123951198937009709734608031896818185442)
        self.assertEqual(c.unit, 'dec')
        gradA = -10**(1.1+1) / (10**(19/10) - 10**1.1) / 10
        gradB = 10**(19/10) / (10**(19/10) - 10**1.1) / 10
        self.assertAlmostEqual(c.uncert, np.sqrt(
            (gradA * a.uncert)**2 + (gradB * b.uncert)**2))

    def testTemperatureAddition(self):
        a = variable(20, 'C')
        b = variable(30, 'C')
        c = a + b
        self.assertEqual(c.value, 50)
        self.assertEqual(c.unit, 'C')
        a = variable(20, 'C')
        b = variable(30, 'DELTAC')
        c = a + b
        self.assertEqual(c.value, 50)
        self.assertEqual(c.unit, 'C')

        a = variable(20, 'DELTAC')
        b = variable(30, 'C')
        c = a + b
        self.assertEqual(c.value, 50)
        self.assertEqual(c.unit, 'C')

        a = variable(20, 'DELTAC')
        b = variable(30, 'DELTAC')
        c = a + b
        self.assertEqual(c.value, 50)
        self.assertEqual(c.unit, 'DELTAC')

        a = variable(100, 'K')
        b = variable(20, 'C')
        c = a + b
        return
        self.assertEqual(c.value, 100 + 273.15 + 20)
        self.assertEqual(c.unit, 'K')

        a = variable(100, 'DELTAK')
        b = variable(20, 'C')
        c = a + b
        self.assertEqual(c.value, 100+273.15 + 20)
        self.assertEqual(c.unit, 'K')

        a = variable(100, 'K')
        b = variable(20, 'DELTAC')
        c = a + b
        self.assertEqual(c.value, 120)
        self.assertEqual(c.unit, 'K')

        a = variable(100, 'DELTAK')
        b = variable(20, 'DELTAC')
        c = a + b
        self.assertEqual(c.value, 120)
        self.assertEqual(c.unit, 'DELTAK')

    def testTemperatureSubtraction(self):
        a = variable(20, 'C')
        b = variable(30, 'C')
        c = a - b
        self.assertEqual(c.value, -10)
        self.assertEqual(c.unit, 'DELTAC')

        a = variable(20, 'C')
        b = variable(30, 'DELTAC')
        c = a - b
        self.assertEqual(c.value, -10)
        self.assertEqual(c.unit, 'C')

        a = variable(20, 'DELTAC')
        b = variable(30, 'C')
        with self.assertRaises(Exception) as context:
            c = a - b
        self.assertTrue("You tried to subtract a temperature from a temperature differnce. This is not possible." in str(
            context.exception))

        a = variable(20, 'DELTAC')
        b = variable(30, 'DELTAC')
        c = a - b
        self.assertEqual(c.value, -10)
        self.assertEqual(c.unit, 'DELTAC')

        a = variable(100, 'K')
        b = variable(20, 'C')
        c = a - b
        self.assertEqual(c.value, 100 - (273.15 + 20))
        self.assertEqual(c.unit, 'DELTAK')

        a = variable(20, 'C')
        b = variable(100, 'DELTAK')
        c = a-b
        self.assertEqual(c.value, 20 + 273.15 - 100)
        self.assertEqual(c.unit, 'K')

        a = variable(100, 'DELTAK')
        b = variable(20, 'C')
        with self.assertRaises(Exception) as context:
            c = a - b
        self.assertTrue("You tried to subtract a temperature from a temperature differnce. This is not possible." in str(
            context.exception))

        a = variable(100, 'K')
        b = variable(20, 'DELTAC')
        c = a - b
        self.assertEqual(c.value, 80)
        self.assertEqual(c.unit, 'K')

        a = variable(100, 'DELTAK')
        b = variable(20, 'DELTAC')
        c = a - b
        self.assertEqual(c.value, 100 - 20)
        self.assertEqual(c.unit, 'DELTAK')

    def testTemperatureMean(self):
        a = variable(20, 'C')
        b = variable(30, 'C')
        c = np.mean([a, b])
        self.assertEqual(c.value, 25)
        self.assertEqual(c.unit, 'C')

        a = variable([20, 30], 'C')
        c = np.mean(a)
        self.assertEqual(c.value, 25)
        self.assertEqual(c.unit, 'C')

    def testTemperatureMultiplication(self):
        a = variable(1, '1/K')
        b = variable(1, 'K')
        c = a * b
        self.assertEqual(c.value, 1)
        self.assertEqual(c.unit, '1')
        self.assertEqual(c.uncert, 0)

        a = variable(1, '1/K')
        b = variable(1, 'DELTAC')
        c = a * b
        self.assertEqual(c.value, 1)
        self.assertEqual(c.unit, '1')
        self.assertEqual(c.uncert, 0)

    def testTemperatureDivide(self):
        a = variable(1, '1/K')
        b = variable(1, '1/K')
        c = a / b
        self.assertEqual(c.value, 1)
        self.assertEqual(c.unit, '1')
        self.assertEqual(c.uncert, 0)

        a = variable(1, '1/K')
        b = variable(1, '1/DELTAC')
        c = a / b
        self.assertEqual(c.value, 1)
        self.assertEqual(c.unit, '1')
        self.assertEqual(c.uncert, 0)

    def testPop(self):

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        A_vec.pop(0)
        np.testing.assert_equal(A_vec.value, [54.3, 91.3])
        self.assertEqual(A_vec.unit, 'L/min')
        np.testing.assert_equal(A_vec.uncert, [5.4, 10.56])

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        A_vec.pop(1)
        np.testing.assert_equal(A_vec.value, [12.3, 91.3])
        self.assertEqual(A_vec.unit, 'L/min')
        np.testing.assert_equal(A_vec.uncert, [2.6, 10.56])

        A_vec = variable([12.3, 54.3, 91.3], 'L/min', uncert=[2.6, 5.4, 10.56])
        A_vec.pop(2)
        np.testing.assert_equal(A_vec.value, [12.3, 54.3])
        self.assertEqual(A_vec.unit, 'L/min')
        np.testing.assert_equal(A_vec.uncert, [2.6, 5.4])

    def testSignificance(self):
        a = variable(23, 'L/min', 2.3)
        b = variable(11, 'mbar', 1.1)
        c = a * b
        vars, sigs = c.getUncertantyContributors()
        self.assertTrue([a] in vars)
        self.assertTrue([b] in vars)
        self.assertAlmostEqual(sigs[vars.index([a])].value, (11 * 2.3)**2 / ((11 * 2.3)**2 + (23 * 1.1)**2) * 100 )
        self.assertAlmostEqual(sigs[vars.index([b])].value, (23 * 1.1)**2 / ((11 * 2.3)**2 + (23 * 1.1)**2) * 100 )
        
        
        a = variable(23, 'L/min', 2.3)
        b = variable(11, 'mbar', 1.1)
        c = a * b
        c.convert('Pa-m3/s')
        vars, sigs = c.getUncertantyContributors()
        self.assertTrue([a] in vars)
        self.assertTrue([b] in vars)
        scale = 100 / 1000 * 60
        self.assertAlmostEqual(sigs[vars.index([a])].value, (11 * 2.3 * scale)**2 / ((11 * 2.3 * scale)**2 + (23 * 1.1 * scale)**2) * 100 )
        self.assertAlmostEqual(sigs[vars.index([b])].value, (23 * 1.1 * scale)**2 / ((11 * 2.3 * scale)**2 + (23 * 1.1 * scale)**2) * 100 )
        
        a = variable(23, 'L/min', 2.3)
        b = variable(11, 'mbar', 1.1)
        a.addCovariance(b,  -0.02, 'L-mbar/min')
        c = a * b
        c.convert('Pa-m3/s')
        scale = 100 / 1000 * 60
        vars, sigs = c.getUncertantyContributors()
        self.assertTrue([a] in vars)
        self.assertTrue([b] in vars)
        self.assertTrue([a,b] in vars)
        self.assertAlmostEqual(sigs[vars.index([a])].value, (11 * 2.3 * scale)**2 / ((11 * 2.3 * scale)**2 + (23 * 1.1 * scale)**2 + 2 * 11 * 23 * 0.02 * scale**2) * 100 )
        self.assertAlmostEqual(sigs[vars.index([b])].value, (23 * 1.1 * scale)**2 / ((11 * 2.3 * scale)**2 + (23 * 1.1 * scale)**2 + 2 * 11 * 23 * 0.02 * scale**2) * 100 )
        self.assertAlmostEqual(sigs[vars.index([a,b])].value, (2 * 11 * 23 * 0.02 * scale**2) / ((11 * 2.3 * scale)**2 + (23 * 1.1 * scale)**2 + 2 * 11 * 23 * 0.02 * scale**2) * 100 )
        
        
        a = variable(23, 'C', 2.3)
        b = variable(11, 'C', 1.1)
        a.addCovariance(b,  -0.02, 'C2')
        c = a - b
        vars, sigs = c.getUncertantyContributors()
        self.assertTrue([a] in vars)
        self.assertTrue([b] in vars)
        self.assertTrue([a,b] in vars)
        self.assertAlmostEqual(sigs[vars.index([a])].value, (1 * 2.3 * scale)**2 / ((1 * 2.3 * scale)**2 + (1 * 1.1 * scale)**2 + 2 * 1 * 1 * 0.02 * scale**2) * 100 )
        self.assertAlmostEqual(sigs[vars.index([b])].value, (1 * 1.1 * scale)**2 / ((1 * 2.3 * scale)**2 + (1 * 1.1 * scale)**2 + 2 * 1 * 1 * 0.02 * scale**2) * 100 )
        self.assertAlmostEqual(sigs[vars.index([a,b])].value, (2 * 1 * 1 * 0.02 * scale**2) / ((1 * 2.3 * scale)**2 + (1 * 1.1 * scale)**2 + 2 * 1 * 1 * 0.02 * scale**2) * 100 )
        
        

if __name__ == '__main__':
    unittest.main()
