import unittest
try:
    from unit import unit, addNewUnit
except ImportError:
    from pyees.unit import unit, addNewUnit

class test(unittest.TestCase):

    def testPower(self):
        A = unit('L-kg/min')
        with self.assertRaises(Exception) as context:
            A**1.5
        self.assertEqual('The power has to be an integer', str(context.exception))
        a, _ = A**2
        self.assertEqual(str(a), 'L2-kg2/min2')

        B = unit('m/s2')
        b, _ = B**0
        self.assertEqual(str(b), '1')

        C = unit('L2/h2')
        c, _ = C**0.5
        self.assertEqual(str(c), 'L/h')

    def testMultiply(self):
        a = unit('L/min')
        b = unit('kg-m/L')
        """ (L/min) * (kg-m/L)
            L-kg-m/min-L
            kg-m/min    """
        c = a * b
        self.assertTrue(unit(c)._assertEqual('kg-m/min'))

        a = unit('L/min')
        b = unit('L/min')
        c = a * b
        self.assertEqual(str(c), 'L2/min2')

        a = unit('km')
        b = unit('1/m')
        c = a * b
        self.assertEqual(str(c), 'km/m')

    def testDivide(self):

        a = unit('L/min')
        b = unit('L/min')
        c = a / b
        self.assertEqual(str(c), '1')

        a = unit('L/min')
        b = unit('kg-m/L')
        """ L/min / (kg-m/L)
            L/min * L/kg-m
            L2 / min-kg-m """
        c = a / b

        self.assertTrue(unit('L2/min-kg-m')._assertEqual(unit(c)))

        A = unit('m')
        B = unit('cm')
        c = A / B
        self.assertEqual(c, 'm/cm')

    def testAdd(self):
        a = unit('L/min')
        b = unit('kg-m/L')

        with self.assertRaises(Exception) as context:
            a + b
        self.assertTrue('You tried to add a variable in [L/min] to a variable in [kg-m/L], but the units do not have the same SI base unit' in str(context.exception))
        
        a = unit('L/min')
        b = unit('L/min')
        _,cUnit,_,_,_ = a + b
        self.assertEqual(cUnit, 'L/min')

        a = unit('m-K/L-bar')
        b = unit('K-m/bar-L')
        _,cUnit,_,_,_ = a + b
        self.assertTrue(unit._assertEqualStatic(cUnit, 'DELTAK-m/bar-L'))
        
        a = unit('J/kg-DELTAK')
        b = unit('J/kg-DELTAK')
        _,cUnit,_,_,_ = a + b
        self.assertTrue(unit._assertEqualStatic(cUnit, 'J/kg-DELTAK'))
        
        a = unit('J/g-DELTAK')
        b = unit('J/kg-DELTAK')
        _,cUnit,_,_,_ = a + b
        self.assertTrue(unit._assertEqualStatic(cUnit, 'J/g-DELTAK'))
        
    def testSub(self):
        a = unit('L/min')
        b = unit('kg-m/L')
        
        with self.assertRaises(Exception) as context:
            a-b
        self.assertTrue('You tried to subtract a variable in [kg-m/L] from a variable in [L/min], but the units do not have the same SI base unit' in str(context.exception))
        
        a = unit('L/min')
        b = unit('L/min')
        _,cUnit,_,_,_ = a - b
        self.assertEqual(str(cUnit), 'L/min')
        
        a = unit('J/kg-DELTAK')
        b = unit('J/kg-DELTAK')
        _,cUnit,_,_,_ = a - b
        self.assertTrue(unit._assertEqualStatic(cUnit, 'J/kg-DELTAK'))
        
        a = unit('J/g-DELTAK')
        b = unit('J/kg-DELTAK')
        _,cUnit,_,_,_ = a - b
        self.assertTrue(unit._assertEqualStatic(cUnit, 'J/g-DELTAK'))

    def testConvert(self):
        a = unit('L/min')
        converter = a.getConverter('m3/h')
        self.assertAlmostEqual(converter.convert(1), 0.06)

        a = unit('K')
        converter = a.getConverter('C')
        self.assertAlmostEqual(converter.convert(300), 26.85)

        a = unit('C')
        converter = a.getConverter('K')
        self.assertAlmostEqual(converter.convert(0), 273.15)

        a = unit('kJ/kg-C')
        converter = a.getConverter('J/kg-K')
        self.assertAlmostEqual(converter.convert(1, useOffset=False), 1000)

        a = unit('kJ/kg-K')
        converter = a.getConverter('J/kg-F')
        self.assertAlmostEqual(converter.convert(1, useOffset=False), 555.555555555555)

        a = unit('K')
        converter = a.getConverter('F')
        self.assertAlmostEqual(converter.convert(300, useOffset=True), 80.33)

        a = unit('A')
        b = unit('V')
        c = a * b
        c = unit(c)
        converter = c.getConverter('W')
        
        a = unit('A')
        b = unit('ohm')
        c = a * b
        c = unit(c)
        converter = c.getConverter('V')
        
        self.assertAlmostEqual(converter.convert(1, useOffset=True), 1)

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

        a = unit('m / s')
        self.assertEqual(str(a), 'm/s')

        with self.assertRaises(Exception) as context:
            a = unit('m!/s')
        self.assertEqual('The character ! is not used within the unitsystem', str(context.exception))

        with self.assertRaises(Exception) as context:
            a = unit('m/s/bar')
        self.assertEqual('A unit can only have a single slash (/)', str(context.exception))

        self.assertEqual(str(unit(' ')), '1')
        self.assertEqual(str(unit('-')), '1')
        self.assertEqual(str(unit('')), '1')
        self.assertEqual(str(unit('--')), '1')
        self.assertEqual(str(unit('()')), '1')
        self.assertEqual(str(unit('( )')), '1')
        self.assertEqual(str(unit('(  )')), '1')  
        self.assertEqual(str(unit('(-)')), '1')
        self.assertEqual(str(unit('(--)')), '1')
        self.assertEqual(str(unit('( -)')), '1')

    def testAddNewUnit(self):
        addNewUnit('gnA', 9.81, 'm/s2')
        converter = unit('gnA').getConverter('m/s2')
        self.assertEqual(converter.convert(1), 9.81)
        
        addNewUnit('gnB', 9.81, 'm/s2', 0)
        converter = unit('gnB').getConverter('m/s2')
        self.assertEqual(converter.convert(1), 9.81)

        converter = unit('gnB').getConverter('mm/h2')
        self.assertEqual(converter.convert(1), 127137600000)
        
        addNewUnit('Rø', 40/21, 'C', -7.5 * 40/21)
        converter = unit('Rø').getConverter('C')
        self.assertAlmostEqual(converter.convert(100), 176.190476190476190476)
        
        converter = unit('Rø').getConverter('F')
        self.assertAlmostEqual(converter.convert(100), 349.142857142857142857142857142857)
            
        converter = unit('Rø').getConverter('K')
        self.assertAlmostEqual(converter.convert(100), 449.340476190476190476 )
        
        addNewUnit('Ra', 5/9 ,'C', -491.67 * 5 / 9)
        converter = unit('Ra').getConverter('C')
        self.assertAlmostEqual(converter.convert(83.1), -226.98333333333333333333333)
        
        converter = unit('Ra').getConverter('F')
        self.assertAlmostEqual(converter.convert(83.1), -376.57)
        
        converter = unit('Ra').getConverter('K')
        self.assertAlmostEqual(converter.convert(83.1), 46.1666666666666666666667)
        
        converter = unit('Ra').getConverter('Rø')
        self.assertAlmostEqual(converter.convert(83.1), -111.66625)
        
        
        
        

if __name__ == '__main__':
    unittest.main()
