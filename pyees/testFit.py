import unittest
import numpy as np
import matplotlib.pyplot as plt
try:
    from fit import *
except ImportError:
    from pyees.fit import *


class test(unittest.TestCase):
   
    def assertRelativeDifference(self, a, b, r):
        assert abs(a-b) < abs(b * r), f"The value {a} and {b} has a greater relative difference than {r}. The difference was {abs(a-b)} and was allowed to be {b*r}"
  
    def testUncertanty(self):

        # constant fit with uncertanty
        for i in range(2, 15):
            x = list(range(1, i + 1))
            y = [10] * i
            x = variable(x, 'm')
            y = variable(y, 'C', uncert=[1] * i)
            F = pol_fit(x, y, deg=0)
            Fa = F.coefficients[0]
            self.assertAlmostEqual(Fa.value, 10)
            self.assertEqual(str(Fa.unit), 'C')
            self.assertAlmostEqual(Fa.uncert, 1 / np.sqrt(i))
            self.assertAlmostEqual(F.r_squared, 1)

        ## fit fromtable 6.1 in "Measurements and their uncertanties" by Ifan G. Hughes
        frequency = variable([10,20,30,40,50,60,70,80,90,100,110], 'Hz')
        voltage = variable([16, 45, 64, 75,70,115, 142, 167, 183, 160, 221], 'mV', [5,5,5,5,30,5,5,5,5,30,5])
        fit = lin_fit(frequency, voltage)
        
        a,b = fit.coefficients
        self.assertEqual(str(a), '2.03 +/- 0.05 [mV/Hz]')
        self.assertEqual(str(b), '-1 +/- 3 [mV]')
        
        # ## TODO compare this to the example in the book
        # fig1, ax1 = plt.subplots()
        # fit.plotResiduals(ax1, label = 'residuals')
        # np.testing.assert_array_equal(ax1.lines[0].get_ydata(), [])


        # ## TODO compare this to the example in the book
        # fig2, ax2 = plt.subplots()
        # fit.plotNormalizedResiduals(ax2, label = 'normalized resudials')
        # np.testing.assert_array_equal(ax2.lines[0].get_ydata(), [])


        
        

    def testLinFit(self):
        a = 2
        b = 10
        n = 100
        x = np.linspace(0, 100, n)
        y = a * x + b

        x = variable(x, 'm')
        y = variable(y, 'C')

        F = lin_fit(x, y)
        Fa = F.coefficients[0]
        Fb = F.coefficients[1]

        self.assertAlmostEqual(Fa.value, 2)
        self.assertEqual(str(Fa.unit), 'DELTAC/m')

        self.assertAlmostEqual(Fb.value, 10)
        self.assertEqual(str(Fb.unit), 'C')

        self.assertAlmostEqual(F.r_squared, 1)

    def testPolFit2(self):
        a = 2
        b = 10
        c = 15
        n = 100
        x = np.linspace(0, 100, n)
        y = a * x**2 + b * x + c

        x = variable(x, 'm')
        y = variable(y, 'C')

        F = pol_fit(x, y)
        Fa = F.coefficients[0]
        Fb = F.coefficients[1]
        Fc = F.coefficients[2]

        self.assertAlmostEqual(Fa.value, 2)
        self.assertEqual(str(Fa.unit), 'DELTAC/m2')

        self.assertAlmostEqual(Fb.value, 10)
        self.assertEqual(str(Fb.unit), 'DELTAC/m')

        self.assertAlmostEqual(Fc.value, 15)
        self.assertEqual(str(Fc.unit), 'C')

        self.assertAlmostEqual(F.r_squared, 1)

    def testPolFit3(self):
        a = 2
        b = 10
        c = 15
        d = 50
        n = 100
        x = np.linspace(0, 100, n)
        y = a * x**3 + b * x**2 + c * x + d

        x = variable(x, 'm')
        y = variable(y, 'C')

        F = pol_fit(x, y, deg=3)
        Fa = F.coefficients[0]
        Fb = F.coefficients[1]
        Fc = F.coefficients[2]
        Fd = F.coefficients[3]

        self.assertAlmostEqual(Fa.value, 2)
        self.assertEqual(str(Fa.unit), 'DELTAC/m3')

        self.assertAlmostEqual(Fb.value, 10)
        self.assertEqual(str(Fb.unit), 'DELTAC/m2')

        self.assertAlmostEqual(Fc.value, 15)
        self.assertEqual(str(Fc.unit), 'DELTAC/m')

        self.assertAlmostEqual(Fd.value, 50)
        self.assertEqual(str(Fd.unit), 'C')

        self.assertAlmostEqual(F.r_squared, 1)

    def testExpFit(self):   
        x = variable([20, 30, 40, 50, 60, 70, 80, 90, 100])
        y = variable([2.7331291071103,4.83637470698903,7.76023628649736,12.92164947233590,19.26005212361100,29.98037228450110,58.70407550133760,82.8915749115424,144.581793442337], 'kg/m3') 
        f = exp_fit(x,y)
        
        a,b,c  = f.coefficients
        self.assertEqual(b.unit, '1')
        self.assertEqual(a.unit, y.unit)
        self.assertEqual(c.unit, y.unit)
        self.assertRelativeDifference(a.value, 0.9875635849080608, 5e-2)
        self.assertRelativeDifference(b.value, 0.04932224945938845, 1e-3)
        self.assertEqual(c.value,0)
        
        x = variable([20, 30, 40, 50, 60, 70, 80, 90, 100], 'C')
        y = variable([2.7331291071103,4.83637470698903,7.76023628649736,12.92164947233590,19.26005212361100,29.98037228450110,58.70407550133760,82.8915749115424,144.581793442337], 'kg/m3') 

        with self.assertRaises(Exception) as context:
            f = exp_fit(x,y, p0 = [1,1,1,1])
        self.assertTrue('The variable "x" cannot have a unit' in str(context.exception))
       
        
    

    def testPowFit(self):    
        x = variable([20, 30, 40, 50, 60, 70, 80, 90, 100])
        y = variable([2735.968626, 5013.971519, 6501.553987, 10229.42877, 10745.50817, 14982.43969, 17657.04326, 20032.85742, 23085.80822], 'kg/m3') 
        f = pow_fit(x,y, p0 = [1,1,0])
        
        a,b,c  = f.coefficients
        self.assertEqual(b.unit, '1')
        self.assertEqual(a.unit, y.unit)
        self.assertEqual(c.unit, y.unit)
        self.assertRelativeDifference(a.value, 54.1146149130, 5e-2)
        self.assertRelativeDifference(b.value, 1.3161605224, 1e-2)
        self.assertEqual(c.value, 0)
        
        
        x = variable([20, 30, 40, 50, 60, 70, 80, 90, 100], 'C')
        y = variable([2.7331291071103,4.83637470698903,7.76023628649736,12.92164947233590,19.26005212361100,29.98037228450110,58.70407550133760,82.8915749115424,144.581793442337], 'kg/m3') 

        with self.assertRaises(Exception) as context:
            f = pow_fit(x,y)
        self.assertTrue('The variable "x" cannot have a unit' in str(context.exception))
       

    

    def testLogisticFit(self):    
        
        x = variable([20, 30, 40, 50, 60, 70, 80, 90, 100])
        y = variable([2.007250271, 5.427429172, 14.89534516, 35.10172629, 64.20820482, 98.38607319, 101.7200864, 114.0915575, 132.427977], 'kg/m3') 
        f = logistic_fit(x,y)

        a,b,c = f.coefficients

        self.assertEqual(a.unit, y.unit)
        self.assertEqual(b.unit, '1')
        
        self.assertRelativeDifference(a.value, 123.4, 2e-1)
        self.assertRelativeDifference(b.value, 0.1, 2e-1)
        self.assertRelativeDifference(c.value, 60, 1e-1)
        
    
        x = variable([20, 30, 40, 50, 60, 70, 80, 90, 100], 'C')
        y = variable([2.7331291071103,4.83637470698903,7.76023628649736,12.92164947233590,19.26005212361100,29.98037228450110,58.70407550133760,82.8915749115424,144.581793442337], 'kg/m3') 

        with self.assertRaises(Exception) as context:
            f = logistic_fit(x,y, p0 = [1,1,1])
        self.assertTrue('The variable "x" cannot have a unit' in str(context.exception))
    




if __name__ == '__main__':
    unittest.main()
