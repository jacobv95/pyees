import numpy as np
import scipy.odr as odr
import string
import sys
import warnings
try:
    from variable import variable, arrayVariable, scalarVariable
    from unit import unit
except ImportError:
    from pyees.variable import variable, arrayVariable, scalarVariable
    from pyees.unit import unit



    
class _fit():
    def __init__(self, func, x, y, p0 = None, useParameters = None) -> None:

        if p0 is None:
            p0 = [0] * self._nParameters
        if len(p0) != self._nParameters:
            raise ValueError(f'The input "p0" has to be None or have a length of {self._nParameters}')

        if useParameters is None:
            useParameters = [True] * self._nParameters   
        if len(useParameters) != self._nParameters:
            raise ValueError(f'The input "useParameters" has to have a length of {self._nParameters}')
        
        self.func = func
        self.p0 = p0
        self.useParameters = useParameters

        if not (isinstance(x, arrayVariable) and isinstance(y, arrayVariable)):
            raise ValueError('The inputs has to be variables')

        self.xVal = x.value
        self.yVal = y.value
        self.xUnit = x._unitObject
        self.yUnit = y._unitObject
        self.xUncert = x.uncert
        self.yUncert = y.uncert

        indexesNotToUse = []
        for i in range(len(self.xVal)):
            if np.isnan(self.xVal[i]):
                indexesNotToUse.append(i)
                continue
            if np.isnan(self.xUncert[i]):
                indexesNotToUse.append(i)
                continue
            if np.isnan(self.yVal[i]):
                indexesNotToUse.append(i)
                continue
            if np.isnan(self.yUncert[i]):
                indexesNotToUse.append(i)
                continue
        

        if indexesNotToUse:
            indexesToUse = [i for i in range(len(self.xVal)) if not i in indexesNotToUse]
            self.xVal = self.xVal[indexesToUse]
            self.xUncert = self.xUncert[indexesToUse]
            self.yVal = self.yVal[indexesToUse]
            self.yUncert = self.yUncert[indexesToUse]

        # uncertanties can not be 0
        self._sx = [elem if elem != 0 else 1e-10 for elem in self.xUncert]
        self._sy = [elem if elem != 0 else 1e-10 for elem in self.yUncert]

        # create the regression
        np.seterr('ignore')
        data = odr.RealData(self.xVal, self.yVal, sx=self._sx, sy=self._sy)
        regression = odr.ODR(data, odr.Model(self._func), beta0=self.p0)
        regression = regression.run()       
        
        ## create a list of coefficients
        self.coefficients = []
        ## the diagonal of the covariance matrix is scaled according to the residual variance
        ## if the residual variance is zero then set the scale to 1
        scale = 1 if regression.res_var == 0 else regression.res_var
        units = self.getVariableUnits()
        for i in range(len(regression.beta)):
            var = variable(regression.beta[i], units[i], np.sqrt(regression.cov_beta[i,i] * scale))
            self.coefficients.append(var)
        
        ## add the covariance between the coefficients
        for i in range(len(self.coefficients)):
            for j in range(len(self.coefficients)):
                if i == j:
                    continue
                self.coefficients[i].addCovariance(
                    var = self.coefficients[j],
                    covariance = regression.cov_beta[i,j],
                    unitStr = str(self.coefficients[i]._unitObject * self.coefficients[j]._unitObject)
                )

        # determine r-squared
        residuals = np.array([yi - pi for yi, pi in zip(self.yVal, self._predict(regression.beta, self.xVal))])# y.value - self._predict(x.value)
        ss_res = sum(residuals**2)
        ss_tot = sum((self.yVal - np.mean(self.yVal))**2)
        if ss_tot != 0:
            self.r_squared = variable(1 - (ss_res / ss_tot))
        else:
            self.r_squared = variable(1)
        
        
        delta, epsilon = regression.delta, regression.eps
        dx_star = ( self.xUncert*np.sqrt( ((self.yUncert*delta)**2) /
                ( (self.yUncert*delta)**2 + (self.xUncert*epsilon)**2 ) ) )
        dy_star = ( self.yUncert*np.sqrt( ((self.xUncert*epsilon)**2) /
                ( (self.yUncert*delta)**2 + (self.xUncert*epsilon)**2 ) ) )
        sigma_odr = np.sqrt(dx_star**2 + dy_star**2)
        residuals = ( np.sign(self.yVal-self._func(regression.beta, self.xVal))
              * np.sqrt(delta**2 + epsilon**2) )
        self._residualY = variable(residuals, self.yUnit, sigma_odr)
        
        
    def getOnlyUsedTerms(self, B):
        for i in range(len(B)):
            if not self.useParameters[i]:
                B[i] = self.p0[i]
        return B
        
    def __str__(self):
        return self.func_name() + f', $R^2 = {self.r_squared.value:.5f}$'

    def scatter(self, ax, label=True, showUncert=True, **kwargs):

        if all(self.xUncert == 0) and all(self.yUncert == 0):
            showUncert = False

        # parse label
        if isinstance(label, str):
            label = label
        elif label == True:
            label = 'Data'
        elif label == False:
            label = None
        elif label is None:
            label = None
        else:
            raise ValueError('The label has to be a string, a bool or None')

        # scatter
        if showUncert:
            return ax.errorbar(self.xVal, self.yVal, xerr=self.xUncert, yerr=self.yUncert, linestyle='', label=label, **kwargs)
        else:
            return ax.scatter(self.xVal, self.yVal, label=label, **kwargs)

    def scatterNormalizedResiduals(self, ax, label = True, **kwargs):
        
        # parse label
        if isinstance(label, str):
            label = label
        elif label == True:
            label = 'Normalized residuals'
        elif label == False:
            label = None
        elif label is None:
            label = None
        else:
            raise ValueError('The label has to be a string, a bool or None')
        
        np.seterr('ignore')
        scale = variable(np.array([1 / ((elemX**2 + elemY**2)**(1/2)) for elemX, elemY in zip(self._sx, self._sy)]))
        normRes = scale * self._residualY
        np.seterr('warn')
        
        return ax.errorbar(self.xVal, normRes.value, xerr=self.xUncert, yerr=normRes.uncert, linestyle='', label=label, **kwargs)

    def scatterResiduals(self, ax, label = True, **kwargs):
        
        # parse label
        if isinstance(label, str):
            label = label
        elif label == True:
            label = 'Normalized residuals'
        elif label == False:
            label = None
        elif label is None:
            label = None
        else:
            raise ValueError('The label has to be a string, a bool or None')
        
        return ax.errorbar(self.xVal, self._residualY.value, xerr=self.xUncert, yerr=self._residualY.uncert, linestyle='', label=label, **kwargs)

    def plotData(self, ax, label=True, **kwargs):

        # parse label
        if isinstance(label, str):
            label = label
        elif label == True:
            label = 'Data'
        elif label == False:
            label = None
        elif label is None:
            label = None
        else:
            raise ValueError('The label has to be a string, a bool or None')

        return ax.plot(self.xVal, self.yVal, label=label, **kwargs)

    def predict(self, x):
        if not isinstance(x, scalarVariable):
            raise ValueError('The input "x" has to be a variable')
        return self.func(x)
    
    def _predict(self, coeffs, x):
        # if not isinstance(x, scalarVariable):
        #     x = variable(x, self.xUnit)
        return self._func(coeffs, x)

    def plotUncertanty(self, ax, x = None, **kwargs):
        
        if x is None:
            x = variable(np.linspace(np.min(self.xVal), np.max(self.xVal), 100), self.xUnit)
        else:
            if not isinstance(x, arrayVariable):
                raise ValueError('The input "x" has to be a variable')
            
        y = self.predict(x)
        y = list(y.value + y.uncert) + [np.nan] + list(y.value - y.uncert)
        x = list(x.value) + [np.nan] + list(x.value)
        return ax.plot(x, y, **kwargs)

    def plot(self, ax, label=True, x=None, **kwargs):

        # parse label
        if isinstance(label, str):
            label = label
        elif label == True:
            label = self.__str__()
        elif label == False:
            label = None
        elif label is None:
            label = None
        else:
            raise ValueError('The label has to be a string, a bool or None')

        if x is None:
            x = variable(np.linspace(np.min(self.xVal), np.max(self.xVal), 100), self.xUnit)
        else:
            if not isinstance(x, arrayVariable):
                raise ValueError('The input "x" has to be a variable')

        y = self.predict(x).value
        x = x.value
        
        return ax.plot(x, y, label=label, **kwargs)
          
    def addUnitToLabels(self, ax):
        self.addUnitToXLabel(ax)
        self.addUnitToYLabel(ax)

    def addUnitToXLabel(self, ax):
        xLabel = ax.get_xlabel()
        if xLabel:
            xLabel += ' '
        xLabel += f'[{self.xUnit}]'
        ax.set_xlabel(xLabel)

    def addUnitToYLabel(self, ax):
        yLabel = ax.get_ylabel()
        if yLabel:
            yLabel += ' '
        yLabel += f'[{self.yUnit}]'
        ax.set_ylabel(yLabel)

    def func(self, x):
        return self._func(self.coefficients, x)

    def evaluateFit(self):
        ## TODO evaluate the fit - use the degrees of freedom as in the book
        
        pred = self._predict(self.xVal)
        
        count = 0
        for yi, uyi, pi in zip(self.yVal, self.yUncert, pred):
            
            ## count the number of datapoints were the prediction is within 1 standard deviation
            if yi == pi:
                count += 1
            elif yi > pi:
                if yi - uyi < pi:
                    count += 1
            else:
                if yi + uyi > pi:
                    count += 1
            
            
        ## determine the relative number of datapoints where the prediction is within 1 standard deviation
        r = count / len(self.xVal)
        
        if r < 0.68:
            warnings.warn(f'{(1-r)*100:.2f}% of the datapoints were more than 1 standard deviation away from the regression. This might indicate that the regression is poor', category=Warning, stacklevel=1)
            


class dummy_fit(_fit):
    """Create a dummy fit of the input data. No regression is performed when creating a dummy fit. The dummy fit object is onely used to easily plot the data
    
    Parameters:
    x : variable
        The independent variable
    y : variable
        The dependents variable
    """
    def __init__(self, x : variable,  y : variable):
        
        if not (isinstance(x, arrayVariable) and isinstance(y, arrayVariable)):
            raise ValueError('The inputs has to be variables')

        self.xVal = x.value
        self.yVal = y.value
        self.xUnit = x.unit
        self.yUnit = y.unit
        self.xUncert = x.uncert
        self.yUncert = y.uncert

        self.r_squared = variable(1)

    def func(self, x):
        if isinstance(x, arrayVariable):
            return variable([1] * len(x))
        return variable(1)
    
    def func_name(self):
        return '1'


class exp_fit(_fit):
    """Create an exponential fit of the input data and easily plot the regression.
    
    f(x) = a * exp(b * x) + c

    Parameters:
    x : variable
        The independent variable
    y : variable
        The dependents variable
    p0 : List(float) = None
        The initial guesses in order (a,b,c). If p0 is set to None, then all parameters are initialized to 0.
    useParameters : List(double) = None
        Wheater or not to use the parameters in order (a,b,c). If an element is set to "False", then the parameter is fixed to the initial guess
    """
    
    def __init__(self, x : variable, y: variable, p0 : list[float] = None, useParameters : list[bool] = [True, True, False]):
        self._nParameters = 3
        if x.unit != '1':
            raise ValueError('The variable "x" cannot have a unit')
        _fit.__init__(self, self.func, x, y, p0=p0, useParameters = useParameters)

    def getVariableUnits(self):
        return [self.yUnit, '', self.yUnit]

    def _func(self, B, x):
        a,b,c = self.getOnlyUsedTerms(B)
        return a * np.exp(b * x) + c

    def func_name(self):
        a,b,c = self.coefficients
        return f'$a\cdot b^(b\cdot x)+d,\quad a={a.__str__(pretty = True)}, \quad b={b.__str__(pretty = True)}, \quad c={c.__str__(pretty = True)}$'


class pow_fit(_fit):
    """Create a power fit of the input data and easily plot the regression.
    
    f(x) = a * b**x + c

    Parameters:
    x : variable
        The independent variable
    y : variable
        The dependents variable
    p0 : List(float) = None
        The initial guesses in order (a,b,c). If p0 is set to None, then all parameters are initialized to 0.
    useParameters : List(double) = None
        Wheater or not to use the parameters in order (a,b,c). If an element is set to "False", then the parameter is fixed to the initial guess
    """
    def __init__(self, x : variable, y: variable, p0 : list[float] = None, useParameters : list[bool] = [True,True,False]):
        self._nParameters = 3
        
        if x.unit != '1':
            raise ValueError('The variable "x" cannot have a unit')
        
        _fit.__init__(self, self.func, x, y, p0=p0, useParameters=useParameters)

    def getVariableUnits(self):
        return [self.yUnit, '1', self.yUnit]

    def _func(self, B, x):
        a,b,c = self.getOnlyUsedTerms(B)
        return a * x**b+c

    def func_name(self):
        return f'$a x^b+c,\quad a={self.coefficients[0].__str__(pretty = True)}, \quad b={self.coefficients[1].__str__(pretty = True)}, \quad b={self.coefficients[1].__str__(pretty = True)}$'


def lin_fit(x : variable, y: variable, p0 : list[float] = None, useParameters : list[bool] = None):
    """Create a linear fit of the input data and easily plot the regression.
    
    f(x) = a*x + b

    Parameters:
    x : variable
        The independent variable
    y : variable
        The dependents variable
    p0 : List(float) = None
        The initial guesses in order (a,b). If p0 is set to None, then all parameters are initialized to 0.
    useParameters : List(double) = None
        Wheater or not to use the parameters in order (a,b). If an element is set to "False", then the parameter is fixed to the initial guess
    """
    return pol_fit(x, y, deg=1, p0=p0, useParameters=useParameters)


class pol_fit(_fit):
    """Create a polynomial fit of the input data and easily plot the regression.
    
    f(x) = sum_0^n ( a_i * x**(n-i) )

    Parameters:
    x : variable
        The independent variable
    y : variable
        The dependents variable
    p0 : List(float) = None
        The initial guesses in order (a_n, ..., a_1, a_0). If p0 is set to None, then all parameters are initialized to 0.
    useParameters : List(double) = None
        Wheater or not to use the parameters in order (a_n, ..., a_1, a_0). If an element is set to "False", then the parameter is fixed to the initial guess
    deg: int
        The degree of the polynomial
    """
    def __init__(self, x : variable, y: variable, p0 : list[float] = None, useParameters : list[bool] = None, deg :int = 2):
        self._nParameters = deg + 1
        self.deg = deg

        _fit.__init__(self, self.func, x, y, p0=p0, useParameters=useParameters)

    def getVariableUnits(self):
        units = []
        n = self.deg
        for i in range(n + 1):
            u = self.yUnit
            if i != n:
                ui = self.xUnit ** (n - i)
                u /= ui
            units.append(u)
        return units

    def _func(self, B, x):
        B = self.getOnlyUsedTerms(B)
        out = 0
        for i,b in enumerate(B):
            out +=  b * x**(self.deg - i)
        return out

    def func_name(self):
        out = '$'
        n = self.deg
        for i in range(n + 1):
            exponent = n - i
            if i == 0:
                out += f'{string.ascii_lowercase[i]}'
            else:
                out += f'+{string.ascii_lowercase[i]}'
            if exponent != 0:
                out += f'x'
            if exponent > 1:
                    out += f'^{exponent}'
       
        for i in range(n + 1):
            out += f', {string.ascii_lowercase[i]}={self.coefficients[i].__str__(pretty = True)}'
        out += '$'
        return out


class logistic_fit(_fit):
    """Create a logistic fit of the input data and easily plot the regression.
    
    f(x) = L / (1 + exp(-k * (x - x0)))

    Parameters:
    x : variable
        The independent variable
    y : variable
        The dependents variable
    p0 : List(float) = None
        The initial guesses in order (L,k,x0). If p0 is set to None, then all parameters are initialized to 0.
    useParameters : List(double) = None
        Wheater or not to use the parameters in order (L,k,x0). If an element is set to "False", then the parameter is fixed to the initial guess
    """
    def __init__(self, x : variable, y: variable, p0 : list[float] = None, useParameters : list[bool] = None):

        self._nParameters = 3

        if x.unit != '1':
            raise ValueError('The variable "x" cannot have a unit')
        _fit.__init__(self, self.func, x, y, p0=p0, useParameters=useParameters)

    def getVariableUnits(self):
        return [self.yUnit,'1','1']
    
    def _func(self, B, x):
        L,k,x0 = self.getOnlyUsedTerms(B)
        return L / (1 + np.exp(-k * (x - x0)))

    def func_name(self):
        L = self.coefficients[0].__str__(pretty=True)
        k = self.coefficients[1].__str__(pretty=True)
        x0 = self.coefficients[2].__str__(pretty=True)

        out = f'$\\frac{{L}}{{1 + e^{{-k\cdot (x-x_0)}}}}'
        out += f'\quad L={L}'
        out += f'\quad k={k}'
        out += f'\quad x_0={x0}$'
        return out



## TODO fit - få det til at virke som i bogen - brug tests - mangler solutions


if __name__ == "__main__":
    x = variable([1,2])
    y = variable([10,10])
    F = pol_fit(x, y, deg=0)
    Fa = F.coefficients[0]
    print(Fa)
    
        