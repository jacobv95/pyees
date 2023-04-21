import unittest
import numpy as np
try:
    from fit import *
except ImportError:
    from pyees.fit import *


class test(unittest.TestCase):

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
        self.assertAlmostEqual(Fa.uncert, 0)

        self.assertAlmostEqual(Fb.value, 10)
        self.assertEqual(str(Fb.unit), 'C')
        self.assertAlmostEqual(Fb.uncert, 0)

        self.assertAlmostEqual(F.r_squared, 1)

    def testPolFit2(self):
        a = 2
        b = 10
        c = 15
        n = 100
        x = np.linspace(0, 100, n)
        y = a * x**2 + b * x + c
        # y += 10 * np.random.rand(n)

        x = variable(x, 'm')
        y = variable(y, 'C')

        F = pol_fit(x, y)
        Fa = F.coefficients[0]
        Fb = F.coefficients[1]
        Fc = F.coefficients[2]

        self.assertAlmostEqual(Fa.value, 2)
        self.assertEqual(str(Fa.unit), 'DELTAC/m2')
        self.assertAlmostEqual(Fa.uncert, 0)

        self.assertAlmostEqual(Fb.value, 10)
        self.assertEqual(str(Fb.unit), 'DELTAC/m')
        self.assertAlmostEqual(Fb.uncert, 0)

        self.assertAlmostEqual(Fc.value, 15)
        self.assertEqual(str(Fc.unit), 'C')
        self.assertAlmostEqual(Fc.uncert, 0)

        self.assertAlmostEqual(F.r_squared, 1)

    def testPolFit3(self):
        a = 2
        b = 10
        c = 15
        d = 50
        n = 100
        x = np.linspace(0, 100, n)
        y = a * x**3 + b * x**2 + c * x + d
        # y += 10 * np.random.rand(n)

        x = variable(x, 'm')
        y = variable(y, 'C')

        F = pol_fit(x, y, deg=3)
        Fa = F.coefficients[0]
        Fb = F.coefficients[1]
        Fc = F.coefficients[2]
        Fd = F.coefficients[3]

        self.assertAlmostEqual(Fa.value, 2)
        self.assertEqual(str(Fa.unit), 'DELTAC/m3')
        self.assertAlmostEqual(Fa.uncert, 0)

        self.assertAlmostEqual(Fb.value, 10)
        self.assertEqual(str(Fb.unit), 'DELTAC/m2')
        self.assertAlmostEqual(Fb.uncert, 0)

        self.assertAlmostEqual(Fc.value, 15)
        self.assertEqual(str(Fc.unit), 'DELTAC/m')
        self.assertAlmostEqual(Fc.uncert, 0)

        self.assertAlmostEqual(Fd.value, 50)
        self.assertEqual(str(Fd.unit), 'C')
        self.assertAlmostEqual(Fd.uncert, 0)

        self.assertAlmostEqual(F.r_squared, 1)


## TODO test exp_fit
## TODO test pow_fit
## TODO test logistic_fit    


if __name__ == '__main__':
    unittest.main()
