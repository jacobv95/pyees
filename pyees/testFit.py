import unittest
import numpy as np
import matplotlib.pyplot as plt

from fit import lin_fit, pow_fit, pol_fit, exp_fit, logistic_fit, variable, _fit, crateNewFitClass, multi_variable_lin_fit, crateNewMultiVariableFitClass

showPlots = False



def gaussianFunc(B, x):
    from numpy import exp
    mu,sigma,y_mu = B
    return y_mu*exp(-(x-mu)**2/(2*sigma**2))

def gaussianFuncVariableUnits(xUnit, yUnit):
    return ['', '', '']

def gaussianFuncName(B):
    mu,sigma,y_mu = B
    return fr'$y_mu \cdot exp(-(x-mu)^2/(2 sigma ^ 2)),\quad mu={mu.__str__(pretty = True)}, \quad sigma={sigma.__str__(pretty = True)}, \quad y_mu={y_mu.__str__(pretty = True)}$'

gaussian = crateNewFitClass(gaussianFunc, gaussianFuncName, gaussianFuncVariableUnits, 3)


def absolutePowerFuncVariableUnits(xUnit, yUnit):
    return ['', '', '']

def absolutePowerFunc(B, x):
    mu,nu,y_mu = B
    return  y_mu*abs(x-mu)**nu

def absolutePowerFuncName(B):
    mu,nu,y_mu = B
    return fr'$ y_mu*abs(x-mu)^*u,\quad mu={mu.__str__(pretty = True)}, \quad nu={nu.__str__(pretty = True)}, \quad y_mu={y_mu.__str__(pretty = True)}$'

absolute_power = crateNewFitClass(absolutePowerFunc, absolutePowerFuncName, absolutePowerFuncVariableUnits, 3)



class test(unittest.TestCase):
   
    def assertRelativeDifference(self, a, b, r):
        assert abs(a-b) < abs(b * r), f"The value {a} and {b} has a greater relative difference than {r}. The difference was {abs(a-b)} and was allowed to be {abs(b*r)}"
  
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
        
        self.assertRelativeDifference(a.uncert, np.sqrt(0.0027), 0.005)
        self.assertRelativeDifference(b.uncert, np.sqrt(11.5), 0.005)
        cov = variable(-0.153, 'mV2/Hz')
        cov.convert('V2/Hz')
        self.assertRelativeDifference(a.covariance[b], cov.value , 0.005)


    def testRegression(self):
        # https://www.physics.utoronto.ca/apl/python/ODR_Fitter_Description.pdf
        
        ## this reference scales the uncertanties based on res_Var
        x = variable([11.9, 8.9, 6.3, 14.0, 8.0, 12.7, 10.2, 18.2, 20.8, 17.8, 17.0, 19.8], '', [0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.2])
        y = variable([26.1, 9.3, 2.9, 42.0, 7.0, 32.8, 16.8, 46.4, 31.3, 49.4, 50.6, 38.0], '', [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2])       

        f = gaussian(x,y, p0 = [15, 5, 10])
        mu, sigma, y_mu = f.coefficients

        epsilon = 1e-4
        self.assertRelativeDifference(mu.value, 16.643, epsilon)
        # self.assertRelativeDifference(mu.uncert, 0.05991, epsilon)
        self.assertRelativeDifference(sigma.value, 4.2778, epsilon)
        # self.assertRelativeDifference(sigma.uncert, 0.042811, epsilon)
        self.assertRelativeDifference(y_mu.value, 50.686, epsilon)
        # self.assertRelativeDifference(y_mu.uncert, 0.27017, epsilon)
        
        
        ## Asymmetric uncertainties
        x = variable([0.305, 0.356, 0.468, 0.659, 0.859, 1.028, 1.091, 1.369, 1.583, 1.646, 1.823, 1.934], '', [0.047, 0.048, 0.040, 0.031, 0.031, 0.031, 0.057, 0.060, 0.041, 0.036, 0.036, 0.058])
        y = variable([25.127, 7.440, 3.345, 2.274, 2.088, 1.045, -0.662, 0.493, 0.927, 0.401, -0.562, -0.405], '', [0.686, 0.501, 0.828, 0.575, 0.926, 0.895, 0.570, 0.715, 0.663, 0.734, 0.918, 0.547])
        
        f = absolute_power(x,y, p0 = [0, -1, 1])
        mu, nu, y_mu = f.coefficients
        epsilon = 1e-4
        self.assertRelativeDifference(mu.value, 0.20045, epsilon)
        # self.assertRelativeDifference(mu.uncert, 0.13953, epsilon)
        self.assertRelativeDifference(nu.value, -1.5997, epsilon)
        # self.assertRelativeDifference(nu.uncert, 0.89354, epsilon)
        self.assertRelativeDifference(y_mu.value, 0.47992, epsilon)
        # self.assertRelativeDifference(y_mu.uncert, 0.23242, epsilon)   
        

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
       
    def testPolFitUnits(self):
        a = 2
        c = 15
        n = 100
        x = np.linspace(0, 100, n)
        y = a * x**2 + c

        x = variable(x, 'm')
        y = variable(y, 'C')

        F = pol_fit(x, y, useParameters=[True, False, True])

        F.predict(x)

    def testMultiVariableFit(self):
        from prop import prop

        ## determine the inputs for the regression
        temperature = variable([-20, -10, 0, 10, 20, 30, 40, 50, 60], 'C')
        temperature.convert('K')
        pressures = variable(np.array([0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]) * 101325, 'Pa')
        humidity = variable([10, 20, 30, 40, 50, 60, 70, 80, 90], '%')
        

        ## determine the density for each combination of inputs
        ## and store the maxtrix of inputs and outputs
        T, P, RH, D = variable([], 'K'),variable([], 'Pa'),variable([], '%'),variable([], 'kg/m3')
        for t in temperature:
            for p in pressures:
                for h in humidity:
                    
                    d = prop('density', 'air', T = t, P = p, rh = h)
                    
                    ## store the values in lists
                    T.append(t)
                    P.append(p)
                    RH.append(h)
                    D.append(d)



        f = multi_variable_lin_fit( [1 / T, P, RH, T * P, T * RH], D)

        res = [
            variable(20.20928106, 'kg-DELTAK/m3', 7e-08),
            variable(0.000023512096813, 'kg/m3-Pa', 2e-15),
            variable(0.002157800109, 'kg/m3-%', 2e-12),
            variable(-0.000000039310176563, 'kg/m3-Pa-DELTAK', 7e-18),
            variable(-0.000008111835383, 'kg/m3-%-DELTAK', 6e-15),
            variable(-0.0700479212, 'kg/m3', 2e-10),
        ]


        for i in range(len(res)):
            self.assertRelativeDifference(res[i].value, f.coefficients[i].value, 0.2)
            self.assertEqual(res[i].unit, f.coefficients[i].unit)
            self.assertRelativeDifference(res[i].uncert, f.coefficients[i].uncert, 0.2)




    def testCreateNewMultiVariableFit(self):
        from prop import prop

        ## determine the inputs for the regression
        temperature = variable([-20, -10, 0, 10, 20, 30, 40, 50, 60], 'C')
        temperature.convert('K')
        pressures = variable(np.array([0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]) * 101325, 'Pa')
        humidity = variable([10, 20, 30, 40, 50, 60, 70, 80, 90], '%')
        

        ## determine the density for each combination of inputs
        ## and store the maxtrix of inputs and outputs
        T, P, RH, D = variable([], 'K'),variable([], 'Pa'),variable([], '%'),variable([], 'kg/m3')
        for t in temperature:
            for p in pressures:
                for h in humidity:
                    
                    d = prop('density', 'air', T = t, P = p, rh = h)
                    
                    ## store the values in lists
                    T.append(t)
                    P.append(p)
                    RH.append(h)
                    D.append(d)

        def getVariableUnits(xUnit, Dunit):
            Tunit, Punit, RHunit = xUnit
            
            out = [
                Dunit * Tunit,
                Dunit / Punit,
                Dunit / RHunit,
                Dunit / (Tunit * Punit),
                Dunit / (Tunit * RHunit),
                Dunit
            ]
            
            return out
        
        def func(B, X):
            T, P, RH = X
            a,b,c,d,e,f = B

            return a / T + b * P + c * RH + d * T * P + e * T * RH + f

        def func_name(coefficients):
            a,b,c,d,e,f = coefficients
            return f"a / T + b * P + c * RH + d * T * P + e * T * RH + f, a = {a}, b = {b}, c = {c}, d = {d}, e = {e}, f = {f}"

        fit = crateNewMultiVariableFitClass(func, func_name, getVariableUnits, 6)
        f = fit([T,P,RH], D)


        res = [
            variable(20.20928106, 'kg-DELTAK/m3', 7e-08),
            variable(0.000023512096813, 'kg/m3-Pa', 2e-15),
            variable(0.002157800109, 'kg/m3-%', 2e-12),
            variable(-0.000000039310176563, 'kg/m3-Pa-DELTAK', 7e-18),
            variable(-0.000008111835383, 'kg/m3-%-DELTAK', 6e-15),
            variable(-0.0700479212, 'kg/m3', 2e-10),
        ]

        for i in range(len(res)):
            self.assertRelativeDifference(res[i].value, f.coefficients[i].value, 0.2)
            self.assertEqual(res[i].unit, f.coefficients[i].unit)
            self.assertRelativeDifference(res[i].uncert, f.coefficients[i].uncert, 0.2)


    

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
    
    def testNewFit(self):
        def func(B, x):
            a = B[0]
            return a*x**2

        def funcName(coefficients):
            a = coefficients[0]
            return f'a*x, a={a}'
        
        def getVariableUnitsFunc(xUnit, yUnit):
            return [yUnit / (xUnit**2)]
        
        nParameters = 1
        
        newFit = crateNewFitClass(func, funcName, getVariableUnitsFunc, nParameters)


        x = variable([1,2,3], 'm')
        y = variable([2,4,6], 'C')
        f1 = newFit(x,y)

        f2 = pol_fit(x, y, useParameters=[True, False, False])
        self.assertEqual(f1.coefficients[0], f2.coefficients[0])
        

        
        
        def func(B, x):
            a,c  = B
            return a*x**2 + c

        def funcName(coefficients):
            a,c = coefficients
            return f'a*x + c, a={a}, c = {c}'
        
        def getVariableUnitsFunc(xUnit, yUnit):
            return [yUnit / (xUnit**2), yUnit]
        
        nParameters = 2
        
        newFit = crateNewFitClass(func, funcName, getVariableUnitsFunc, nParameters)


        x = variable([1,2,3], 'm')
        y = variable([2,4,6], 'C')
        f1 = newFit(x,y)

        f2 = pol_fit(x, y, useParameters=[True, False, True])
        self.assertEqual(f1.coefficients[0], f2.coefficients[0])
        self.assertEqual(f1.coefficients[1], f2.coefficients[2])
        
        
        
    def testMonteCarloPolFit(self):   
        import scipy.odr as odr     
        def func(B,x):
            a,b,c = B
            return a*x**2 + b*x + c


        a = np.random.uniform(0.01, 0.02)
        b = np.random.uniform(-5, -3)
        c = np.random.uniform(-15, 15)

        uncertScale = 0.01
        noiseScale = 0.01

        xMin = 10
        xMax = 200

        xValue = list(np.linspace(xMin, xMax, 10))
        xNoise = [np.random.uniform(-elem * noiseScale, elem * noiseScale) for elem in xValue]
        xValue = [val + noise for val, noise in zip(xValue, xNoise)]
        xUncert = [np.random.uniform(np.abs(elem * uncertScale)) for elem in xValue]

        yValue = func([a,b,c], np.array(xValue))
        yNoise = [np.random.uniform(-elem * noiseScale, elem * noiseScale) for elem in yValue]
        yValue = [val + noise for val, noise in zip(yValue, yNoise)]
        yUncert = [np.random.uniform(np.abs(elem * uncertScale)) for elem in yValue]

        wx = [1 / elem**2 for elem in xUncert]
        wy = [1 / elem**2 for elem in yUncert]

        n = 10000

        if showPlots:
            fig, ax = plt.subplots()
            x_s = []
            y_s = []

        nParameters = 3
        parameterMatrix = np.zeros([n,nParameters])

        for i in range(n):
            x = [np.random.normal(loc = val, scale = unc) for val, unc in zip(xValue, xUncert)]
            y = [np.random.normal(loc = val, scale = unc) for val, unc in zip(yValue, yUncert)]
            if showPlots:
                x_s.append(x)
                y_s.append(y)
            data = odr.Data(x,y, we = wy, wd = wx)
            regression = odr.ODR(data, odr.Model(func), beta0 = [0] * nParameters)
            regression = regression.run()       
            popt = regression.beta
            parameterMatrix[i,:] = popt

        if showPlots:
            plt.scatter(x_s,y_s, color = 'black', marker = '.')


        parameterValues = np.zeros(nParameters)
        covarianceMatrix = np.zeros([nParameters, nParameters])
        for i in range(nParameters):
            parameterValues[i] = np.mean(parameterMatrix[:,i])

        for i in range(nParameters):
            for j in range(nParameters):
                mu_i = parameterValues[i]
                mu_j = parameterValues[j]
                out = 0
                for ii in range(n):
                    out += (parameterMatrix[ii,i] - mu_i) * (parameterMatrix[ii,j] - mu_j)
                out /= n-1
                covarianceMatrix[i,j] = out 

        parameters = []
        for i in range(nParameters):
            parameters.append(variable(parameterValues[i], '', np.sqrt(covarianceMatrix[i,i])))

        for i in range(nParameters):
            for j in range(nParameters):
                if i == j: continue
                parameters[i].addCovariance(parameters[j], covarianceMatrix[i,j], '')

        if showPlots:
            x = np.linspace(min(xValue), max(xValue), 100)
            y = func(parameters, x)
            plt.plot(x, y.value, color = 'b', label = 'Monte Carlo')
            plt.plot(x, y.value + y.uncert, color = 'b', linestyle = 'dashed')
            plt.plot(x, y.value - y.uncert, color = 'b', linestyle = 'dashed')


        ## using pyees
        x = variable(xValue, '1', xUncert)
        y = variable(yValue, '1', yUncert)
        f = pol_fit(x,y)


        for i in range(nParameters):
            self.assertRelativeDifference(f.coefficients[i].value, parameters[i].value, 5e-2)
            self.assertRelativeDifference(f.coefficients[i].uncert, parameters[i].uncert, 5e-2)
            
            for j in range(nParameters):
                if i == j: continue
                self.assertRelativeDifference(
                    f.coefficients[i].covariance[f.coefficients[j]],
                    covarianceMatrix[i,j],
                    5e-2
                )
            

        if showPlots:
            f.scatter(ax, color = 'green', label = None)
            f.plot(ax, color = 'r', label='Pyees')
            f.plotUncertanty(ax, color = 'r', linestyle = 'dashed')
            ax.legend()
            fig.tight_layout()
            plt.show()
                        

    def testPlotly(self):
            
        ## define some data and create a fit-object
        x = variable([1,2,3], 'min', [0.3, 1.2, 2.1])
        y = variable([5,9,11], 'L', [2.5, 5.3, 7.8])
        f = lin_fit(x,y)
        
        ## create a figure using matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        f.scatter(ax)
        f.plot(ax)
        f.plotData(ax)
        f.plotUncertanty(ax)
        f.scatterResiduals(ax)
        f.scatterNormalizedResiduals(ax)
        f.plotUncertantyOfInputs(ax)
        f.scatterUncertatyAsEllipses(ax)
        f.plotResiduals(ax, color = 'black', linestyle = 'dashed')
        ax.set_xlabel('Time')
        ax.set_ylabel('Volume')
        f.addUnitToLabels(ax)
        if showPlots:
            plt.show()
        
        
        ## create a figure using plotly
        import plotly.graph_objects as go
        fig = go.Figure()
        f.scatter(fig)
        f.plot(fig)
        f.plotData(fig)
        f.plotUncertanty(fig)
        f.scatterResiduals(fig)
        f.scatterNormalizedResiduals(fig)
        f.plotUncertantyOfInputs(fig)
        f.scatterUncertatyAsEllipses(fig)
        f.plotResiduals(ax, color = 'black', linestyle = 'dashed')
        fig.update_yaxes(title = 'Volume')
        fig.update_xaxes(title = "Time")
        f.addUnitToLabels(fig)
        if showPlots:
            fig.show()
        
        ## create a figure using plotly
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols = 1)
        f.scatter(fig, row = 2, col = 1)
        f.plot(fig, row = 2, col = 1)
        f.plotData(fig, row = 2, col = 1),
        f.plotUncertanty(fig, row = 2, col = 1)
        f.scatterResiduals(fig, row = 1, col = 1)
        f.scatterNormalizedResiduals(fig, row = 1, col = 1)
        f.plotUncertantyOfInputs(fig, row = 2, col = 1)
        f.scatterUncertatyAsEllipses(fig, row = 2, col = 1)
        fig.get_subplot(col = 1, row = 2).xaxis.title = "Time"
        fig.get_subplot(col = 1, row = 2).yaxis.title = "Volume"
        f.addUnitToLabels(fig, col = 1, row = 2)
        if showPlots:
            fig.show()

if __name__ == '__main__':
    showPlots = False
    unittest.main()
