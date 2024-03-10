import unittest
import numpy as np
try:
    from unit import unit, addNewUnit
    from variable import variable
except ImportError:
    from pyees.unit import unit, addNewUnit
    from pyees.variable import variable

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
        A = variable(1,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 0.06)
        self.assertAlmostEqual(A.uncert, 0.06)
        self.assertAlmostEqual(A._uncertSI, 1)

        a = unit('K')
        converter = a.getConverter('C')
        A = variable(300,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 26.85)
        self.assertAlmostEqual(A.uncert, 1)
        self.assertAlmostEqual(A._uncertSI, 1)

        a = unit('C')
        converter = a.getConverter('K')
        A = variable(0,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 273.15)
        self.assertAlmostEqual(A.uncert, 1)
        self.assertAlmostEqual(A._uncertSI, 1)

        a = unit('kJ/kg-C')
        converter = a.getConverter('J/kg-K')
        A = variable(1,'',1)
        converter(A, useOffset=False)
        self.assertAlmostEqual(A.value, 1000)
        self.assertAlmostEqual(A.uncert, 1000)
        self.assertAlmostEqual(A._uncertSI, 1)

        a = unit('kJ/kg-K')
        converter = a.getConverter('J/kg-F')
        A = variable(1,'',1)
        converter(A, useOffset=False)
        self.assertAlmostEqual(A.value, 555.555555555555)
        self.assertAlmostEqual(A.uncert, 555.555555555555)
        self.assertAlmostEqual(A._uncertSI, 1)

        a = unit('K')
        converter = a.getConverter('F')
        A = variable(300,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 80.33)
        self.assertAlmostEqual(A.uncert, 1.7999999999999998)
        self.assertAlmostEqual(A._uncertSI, 1)

        a = unit('mm2')
        converter = a.getConverter('m2')
        A = variable(1,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 1/1000 * 1/1000)
        self.assertAlmostEqual(A.uncert, 1/1000 * 1/1000)
        self.assertAlmostEqual(A._uncertSI, 1)

        a = unit('A')
        b = unit('V')
        c = a * b
        converter = c.getConverter('W')
        
        a = unit('A')
        b = unit('ohm')
        c = a * b
        converter = c.getConverter('V')
        A = variable(1,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 1)
        self.assertAlmostEqual(A.uncert, 1)
        self.assertAlmostEqual(A._uncertSI, 1)

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
        
        a = unit('(m/s2)2/Hz')
        self.assertEqual(a, unit('m2/s4-Hz'))
        self.assertEqual(a.unitStrPretty, rf'\frac{{\left( \frac{{m}}{{s^{{2}}}} \right)^{{2}}}}{{Hz}}')
        
        a = unit('(m/s2)1/Hz')
        self.assertEqual(a, unit('m/s2-Hz'))
        self.assertEqual(a.unitStrPretty, rf'\frac{{\frac{{m}}{{s^{{2}}}}}}{{Hz}}')
        
        a = unit('((m-s2)2/(Hz))2')
        self.assertEqual(a, unit('m4-s8/Hz2'))
        self.assertEqual(a.unitStrPretty, rf'\left( \frac{{\left( m\cdot s^{{2}} \right)^{{2}}}}{{Hz}} \right)^{{2}}')
        
        a = unit('((m/s2)2/(Hz))2')
        self.assertEqual(a, unit('m4/s8-Hz2'))
        self.assertEqual(a.unitStrPretty, rf'\left( \frac{{\left( \frac{{m}}{{s^{{2}}}} \right)^{{2}}}}{{Hz}} \right)^{{2}}')
        
        a = unit('((m-s2)2-(Hz))2')
        self.assertEqual(a, unit('m4-s8-Hz2'))
        self.assertEqual(a.unitStrPretty, rf'\left( \left( m\cdot s^{{2}} \right)^{{2}} \cdot Hz \right)^{{2}}')
        
        a = unit('((m/s2)2-(Hz))2')
        self.assertEqual(a, unit('m4-Hz2/s8'))
        self.assertEqual(a.unitStrPretty, rf'\left( \left( \frac{{m}}{{s^{{2}}}} \right)^{{2}} \cdot Hz \right)^{{2}}')
        
        a = unit('((m/s2)3-(Hz))2')       
        self.assertEqual(a, unit('m6-Hz2/s12'))
        self.assertEqual(a.unitStrPretty, rf'\left( \left( \frac{{m}}{{s^{{2}}}} \right)^{{3}} \cdot Hz \right)^{{2}}')
        
        a = unit('s12')
        self.assertEqual(a, unit('s12'))
        self.assertEqual(a.unitStrPretty, rf's^{{12}}')
        
        
        
        

    def testAddNewUnit(self):
        addNewUnit('gnA', 9.81, 'm/s2')
        converter = unit('gnA').getConverter('m/s2')
        A = variable(1,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 9.81)
        self.assertAlmostEqual(A.uncert, 9.81)
        self.assertAlmostEqual(A._uncertSI, 1)
        
        addNewUnit('gnB', 9.81, 'm/s2', 0)
        converter = unit('gnB').getConverter('m/s2')
        A = variable(1,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 9.81)
        self.assertAlmostEqual(A.uncert, 9.81)
        self.assertAlmostEqual(A._uncertSI, 1)

        converter = unit('gnB').getConverter('mm/h2')
        A = variable(1,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 127137600000)
        self.assertAlmostEqual(A.uncert, 127137600000)
        self.assertAlmostEqual(A._uncertSI, 1)
        
        addNewUnit('Rø', 40/21, 'C', -7.5 * 40/21)
        converter = unit('Rø').getConverter('C')
        A = variable(100,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 176.190476190476190476)
        self.assertAlmostEqual(A.uncert, 1.9047619047619047619048)
        self.assertAlmostEqual(A._uncertSI, 1)
        
        converter = unit('Rø').getConverter('F')
        A = variable(100,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 349.142857142857142857142857142857)
        self.assertAlmostEqual(A.uncert,  3.4285714285714285714286)
        self.assertAlmostEqual(A._uncertSI, 1)
            
        converter = unit('Rø').getConverter('K')
        A = variable(100,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 449.340476190476190476)
        self.assertAlmostEqual(A.uncert,  1.9047619047619047619048)
        self.assertAlmostEqual(A._uncertSI, 1)
        
        addNewUnit('Ra', 5/9 ,'C', -491.67 * 5 / 9)
        converter = unit('Ra').getConverter('C')
        A = variable(83.1,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, -226.98333333333333333333333)
        self.assertAlmostEqual(A.uncert,  0.5555555555555556)
        self.assertAlmostEqual(A._uncertSI, 1)
        
        converter = unit('Ra').getConverter('F')
        A = variable(83.1,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, -376.57)
        self.assertAlmostEqual(A.uncert,  1)
        self.assertAlmostEqual(A._uncertSI, 1)
        
        converter = unit('Ra').getConverter('K')
        A = variable(83.1,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, 46.1666666666666666666667)
        self.assertAlmostEqual(A.uncert,  0.5555555555555556)
        self.assertAlmostEqual(A._uncertSI, 1)
          
        converter = unit('Ra').getConverter('Rø')
        A = variable(83.1,'',1)
        converter(A)
        self.assertAlmostEqual(A.value, -111.66625)
        self.assertAlmostEqual(A.uncert,  0.29166666666666666666667)
        self.assertAlmostEqual(A._uncertSI, 1)
        

    

if __name__ == '__main__':
    unittest.main()
