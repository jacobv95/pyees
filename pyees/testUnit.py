import unittest
import numpy as np
try:
    from unit import unit, addNewUnit
except ImportError:
    from pyees.unit import unit, addNewUnit

class test(unittest.TestCase):

    def testPower(self):
        A = unit('L-kg/min')
        a = A**2
        self.assertTrue(a == unit('L2-kg2/min2'))

        B = unit('m/s2')
        b = B**0
        self.assertTrue(b == unit('1'))

        C = unit('L2/h2')
        c = C**0.5
        self.assertTrue(c == unit('L/h'))

    def testMultiply(self):
        a = unit('L/min')
        b = unit('kg-m/L')
        """ (L/min) * (kg-m/L)
            L-kg-m/min-L
            kg-m/min    """
        c = a * b
        self.assertTrue(c == unit('kg-m/min'))

        a = unit('L/min')
        b = unit('L/min')
        c = a * b
        self.assertTrue(c == unit('L2/min2'))

        a = unit('km')
        b = unit('1/m')
        c = a * b
        self.assertTrue(c == unit('km/m'))
        
        a = unit('m/N')
        b = unit('N/cm')
        c = a * b
        self.assertTrue(c == unit('m/cm'))
        

    def testDivide(self):

        a = unit('L/min')
        b = unit('L/min')
        c = a / b
        self.assertTrue(c == unit('1'))

        a = unit('L/min')
        b = unit('kg-m/L')
        """ (L/min) / (kg-m/L)
            (L/min) * (L/kg-m)
            L2 / min-kg-m """
        c = a / b
        self.assertTrue(c == unit('L2/min-kg-m'))

        A = unit('m')
        B = unit('cm')
        c = A / B
        self.assertTrue(c == unit('m/cm'))
    
    def testAdd(self):
        a = unit('L/min')
        b = unit('kg-m/L')

        with self.assertRaises(Exception) as context:
            a + b
        self.assertTrue('You tried to add a variable in [L/min] to a variable in [kg-m/L], but the units do not have the same SI base unit' in str(context.exception))
        
        a = unit('L/min')
        b = unit('L/min')
        outputUnit = a + b
        self.assertTrue(outputUnit == unit('L/min'))
        self.assertFalse(outputUnit.isLogarithmicUnit())

        a = unit('m-K/L-bar')
        b = unit('K-m/bar-L')
        outputUnit = a + b
        self.assertTrue(outputUnit == unit('DELTAK-m/bar-L'))
        self.assertFalse(outputUnit.isLogarithmicUnit())
        
        a = unit('J/kg-DELTAK')
        b = unit('J/kg-DELTAK')
        outputUnit = a + b
        self.assertTrue(outputUnit == unit('J/kg-DELTAK'))
        self.assertFalse(outputUnit.isLogarithmicUnit())
        
        a = unit('J/g-DELTAK')
        b = unit('J/kg-DELTAK')
        outputUnit = a + b
        self.assertTrue(outputUnit == unit('J/g-DELTAK'))
        self.assertFalse(outputUnit.isLogarithmicUnit())
        
        a = unit('dB')
        b = unit('dB')
        outputUnit = a + b
        self.assertTrue(outputUnit == unit('dB'))
        self.assertTrue(outputUnit.isLogarithmicUnit())
        
        a = unit('mB')
        b = unit('dB')
        outputUnit = a + b
        self.assertTrue(outputUnit == unit('B'))
        self.assertTrue(outputUnit.isLogarithmicUnit())
        
        a = unit('L/min')
        b = unit('m3/h')
        outputUnit = a + b
        self.assertTrue(outputUnit == unit('m3/s'))
        self.assertFalse(outputUnit.isLogarithmicUnit())
        
    def testSub(self):
        a = unit('L/min')
        b = unit('kg-m/L')

        with self.assertRaises(Exception) as context:
            a - b
        self.assertTrue('You tried to subtract a variable in [kg-m/L] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))
        
        a = unit('L/min')
        b = unit('L/min')
        outputUnit = a - b
        self.assertTrue(outputUnit == unit('L/min'))
        self.assertFalse(outputUnit.isLogarithmicUnit())

        a = unit('m-K/L-bar')
        b = unit('K-m/bar-L')
        outputUnit = a - b
        self.assertTrue(outputUnit == unit('DELTAK-m/bar-L'))
        self.assertFalse(outputUnit.isLogarithmicUnit())
        
        a = unit('J/kg-DELTAK')
        b = unit('J/kg-DELTAK')
        outputUnit = a - b
        self.assertTrue(outputUnit == unit('J/kg-DELTAK'))
        self.assertFalse(outputUnit.isLogarithmicUnit())
        
        a = unit('J/g-DELTAK')
        b = unit('J/kg-DELTAK')
        outputUnit = a - b
        self.assertTrue(outputUnit == unit('J/g-DELTAK'))
        self.assertFalse(outputUnit.isLogarithmicUnit())
        
        a = unit('dB')
        b = unit('dB')
        outputUnit = a - b
        self.assertTrue(outputUnit == unit('dB'))
        self.assertTrue(outputUnit.isLogarithmicUnit())
        
        a = unit('mB')
        b = unit('dB')
        outputUnit = a - b
        self.assertTrue(outputUnit == unit('B'))
        self.assertTrue(outputUnit.isLogarithmicUnit())
        
        a = unit('L/min')
        b = unit('m3/h')
        outputUnit = a - b
        self.assertTrue(outputUnit == unit('m3/s'))
        self.assertFalse(outputUnit.isLogarithmicUnit())

    def testConvert(self):
        a = unit('L/min')
        converter = a.getConverter('m3/h')
        np.testing.assert_array_almost_equal(converter(1,1,1), [0.06, 0.06, 1])

        a = unit('K')
        converter = a.getConverter('C')
        np.testing.assert_array_almost_equal(converter(300,1,1), [26.85, 1,1])

        a = unit('C')
        converter = a.getConverter('K')
        np.testing.assert_array_almost_equal(converter(0,1,1), [273.15,1,1])

        a = unit('kJ/kg-C')
        converter = a.getConverter('J/kg-K')
        np.testing.assert_array_almost_equal(converter(1,1,1, useOffset=False), [1000,1000,1])

        a = unit('kJ/kg-K')
        converter = a.getConverter('J/kg-F')
        np.testing.assert_array_almost_equal(converter(1,1,1, useOffset=False), [555.555555555555,555.555555555555,1])

        a = unit('K')
        converter = a.getConverter('F')
        np.testing.assert_array_almost_equal(converter(300,1,1, useOffset=True), [80.33, 1.7999999999999998, 1])

        a = unit('mm2')
        converter = a.getConverter('m2')
        np.testing.assert_array_almost_equal(converter(1,1,1), [1/1000 * 1/1000, 1/1000 * 1/1000, 1])

        a = unit('A')
        b = unit('V')
        c = a * b
        converter = c.getConverter('W')
        
        a = unit('A')
        b = unit('ohm')
        c = a * b
        converter = c.getConverter('V')
        
        np.testing.assert_array_almost_equal(converter(1,1,1, useOffset=True), [1,1,1])

        with self.assertRaises(Exception) as context:
            a = unit('mu')
        self.assertEqual('''The unit (mu) was not found. Therefore it was interpreted as a prefix and a unit. The prefix was identified as "mu" and the unit was identified as "1". However, the unit "1" cannot have a prefix''', str(context.exception))

        with self.assertRaises(Exception) as context:
            a = unit('1/M')
        self.assertEqual('The unit (M) was not found. Therefore it was interpreted as a prefix and a unit. The prefix was identified as "M" and the unit was identified as "1". However, the unit "1" cannot have a prefix', str(context.exception))

        with self.assertRaises(Exception) as context:
            a = unit('1/k')
        self.assertEqual('The unit (k) was not found. Therefore it was interpreted as a prefix and a unit. The prefix was identified as "k" and the unit was identified as "1". However, the unit "1" cannot have a prefix', str(context.exception))

        a = unit('L/min')
        with self.assertRaises(Exception) as context:
            converter = a.getConverter('m')
        self.assertEqual('You tried to convert from L/min to m. But these do not have the same base units', str(context.exception))

    def testInput(self):

        a = unit(None)
        self.assertTrue(a == unit('1'))
        
        a = unit('m / s')
        self.assertTrue(a == unit('m/s'))

        with self.assertRaises(Exception) as context:
            a = unit('m!/s')
        self.assertTrue('The character ! is not used within the unitsystem', str(context.exception))

        with self.assertRaises(Exception) as context:
            a = unit('m/s/bar')
        self.assertTrue('A unit can only have a single slash (/)', str(context.exception))

        self.assertTrue(unit(' ') == unit('1'))
        self.assertTrue(unit('-') == unit('1'))
        self.assertTrue(unit('') == unit('1'))
        self.assertTrue(unit('--') == unit('1'))
        self.assertTrue(unit('- -') == unit('1'))
        
        self.assertTrue(unit('()') == unit('1'))
        self.assertTrue(unit('( )') == unit('1'))
        self.assertTrue(unit('(  )') == unit('1'))  
        self.assertTrue(unit('(-)') == unit('1'))
        self.assertTrue(unit('(--)') == unit('1'))
        self.assertTrue(unit('(- -)') == unit('1'))
        self.assertTrue(unit('( -)') == unit('1'))
        self.assertTrue(unit('( - (- ))') == unit('1'))
        
        self.assertTrue(unit('(m/s2)2/Hz') == unit('m2/s4-Hz'))
        self.assertTrue(unit('(m1/s2)2/Hz') == unit('m2/s4-Hz'))
        self.assertTrue(unit('(m1/s2)1/Hz') == unit('m/s2-Hz'))

    def testAddNewUnit(self):
        addNewUnit('gnA', 9.81, 'm/s2')
        converter = unit('gnA').getConverter('m/s2')
        np.testing.assert_array_almost_equal(converter(1,1,1), [9.81, 9.81, 1])
        
        addNewUnit('gnB', 9.81, 'm/s2', 0)
        converter = unit('gnB').getConverter('m/s2')
        np.testing.assert_array_almost_equal(converter(1,1,1), [9.81, 9.81, 1])

        converter = unit('gnB').getConverter('mm/h2')
        np.testing.assert_array_almost_equal(converter(1,1,1), [127137600000, 127137600000, 1])
        
        addNewUnit('Rø', 40/21, 'C', -7.5 * 40/21)
        converter = unit('Rø').getConverter('C')
        np.testing.assert_array_almost_equal(converter(100,1,1), [176.190476190476190476, 1.9047619047619047619048, 1])
        
        converter = unit('Rø').getConverter('F')
        np.testing.assert_array_almost_equal(converter(100,1,1), [349.142857142857142857142857142857, 3.4285714285714285714286 , 1])
            
        converter = unit('Rø').getConverter('K')
        np.testing.assert_array_almost_equal(converter(100,1,1), [449.340476190476190476, 1.9047619047619047619048, 1] )
        
        addNewUnit('Ra', 5/9 ,'C', -491.67 * 5 / 9)
        converter = unit('Ra').getConverter('C')
        np.testing.assert_array_almost_equal(converter(83.1,1,1), [-226.98333333333333333333333, 0.5555555555555556, 1])
        
        converter = unit('Ra').getConverter('F')
        np.testing.assert_array_almost_equal(converter(83.1,1,1), [-376.57, 1, 1])
        
        converter = unit('Ra').getConverter('K')
        np.testing.assert_array_almost_equal(converter(83.1,1,1), [46.1666666666666666666667, 0.5555555555555556, 1])
        
        converter = unit('Ra').getConverter('Rø')
        np.testing.assert_array_almost_equal(converter(83.1,1,1), [-111.66625, 0.29166666666666666666667, 1])
        

    

if __name__ == '__main__':
    unittest.main()
