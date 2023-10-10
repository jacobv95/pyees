import unittest
import numpy as np
import matplotlib.pyplot as plt
try:
    from fit import lin_fit, pow_fit, pol_fit, exp_fit, logistic_fit, variable, _fit
except ImportError:
    from pyees.fit import lin_fit, pow_fit, pol_fit, exp_fit, logistic_fit, variable, _fit



class gaussian(_fit):
    def __init__(self, x : variable, y: variable, p0 : list[float] = None, useParameters : list[bool] = [True, True, True]):
        self._nParameters = 3
        _fit.__init__(self, self.func, x, y, p0=p0, useParameters = useParameters)

    def getVariableUnits(self):
        return ['', '', '']

    def _func(self, B, x):
        from numpy import exp
        mu,sigma,y_mu = self.getOnlyUsedTerms(B)
        return y_mu*exp(-(x-mu)**2/(2*sigma**2))

    def func_name(self):
        mu,sigma,y_mu = self.coefficients
        return f'$y_mu \cdot exp(-(x-mu)^2/(2 sigma ^ 2)),\quad mu={mu.__str__(pretty = True)}, \quad sigma={sigma.__str__(pretty = True)}, \quad y_mu={y_mu.__str__(pretty = True)}$'

class absolute_power(_fit):
    def __init__(self, x : variable, y: variable, p0 : list[float] = None, useParameters : list[bool] = [True, True, True]):
        self._nParameters = 3
        _fit.__init__(self, self.func, x, y, p0=p0, useParameters = useParameters)

    def getVariableUnits(self):
        return ['', '', '']

    def _func(self, B, x):
        mu,nu,y_mu = self.getOnlyUsedTerms(B)
        return  y_mu*abs(x-mu)**nu

    def func_name(self):
        mu,nu,y_mu = self.coefficients
        return f'$ y_mu*abs(x-mu)^*u,\quad mu={mu.__str__(pretty = True)}, \quad nu={nu.__str__(pretty = True)}, \quad y_mu={y_mu.__str__(pretty = True)}$'


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


    def testRegression(self):
        # https://www.physics.utoronto.ca/apl/python/ODR_Fitter_Description.pdf
        
        x = variable([11.9, 8.9, 6.3, 14.0, 8.0, 12.7, 10.2, 18.2, 20.8, 17.8, 17.0, 19.8], '', [0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.2])
        y = variable([26.1, 9.3, 2.9, 42.0, 7.0, 32.8, 16.8, 46.4, 31.3, 49.4, 50.6, 38.0], '', [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2])       

        f = gaussian(x,y, p0 = [15, 5, 10])
        mu, sigma, y_mu = f.coefficients

        epsilon = 1e-4
        self.assertRelativeDifference(mu.value, 16.643, epsilon)
        self.assertRelativeDifference(mu.uncert, 0.05991, epsilon)
        self.assertRelativeDifference(sigma.value, 4.2778, epsilon)
        self.assertRelativeDifference(sigma.uncert, 0.042811, epsilon)
        self.assertRelativeDifference(y_mu.value, 50.686, epsilon)
        self.assertRelativeDifference(y_mu.uncert, 0.27017, epsilon)
        
        
        ## Asymmetric uncertainties
        x = variable([0.305, 0.356, 0.468, 0.659, 0.859, 1.028, 1.091, 1.369, 1.583, 1.646, 1.823, 1.934], '', [0.047, 0.048, 0.040, 0.031, 0.031, 0.031, 0.057, 0.060, 0.041, 0.036, 0.036, 0.058])
        y = variable([25.127, 7.440, 3.345, 2.274, 2.088, 1.045, -0.662, 0.493, 0.927, 0.401, -0.562, -0.405], '', [0.686, 0.501, 0.828, 0.575, 0.926, 0.895, 0.570, 0.715, 0.663, 0.734, 0.918, 0.547])
        
        f = absolute_power(x,y, p0 = [0, -1, 1])
        mu, nu, y_mu = f.coefficients
        epsilon = 1e-4
        self.assertRelativeDifference(mu.value, 0.20045, epsilon)
        self.assertRelativeDifference(mu.uncert, 0.13953, epsilon)
        self.assertRelativeDifference(nu.value, -1.5997, epsilon)
        self.assertRelativeDifference(nu.uncert, 0.89354, epsilon)
        self.assertRelativeDifference(y_mu.value, 0.47992, epsilon)
        self.assertRelativeDifference(y_mu.uncert, 0.23242, epsilon)
        
        
        
        
        

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
        f = pow_fit(x,y, p0 = [50,1,0])
        
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
