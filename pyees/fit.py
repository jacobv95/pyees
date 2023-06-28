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
    def __init__(self, func, x, y, p0) -> None:
        self.func = func

        if not (isinstance(x, arrayVariable) and isinstance(y, arrayVariable)):
            raise ValueError('The inputs has to be variables')

        self.xVal = x.value
        self.yVal = y.value
        self.xUnit = x._unitObject
        self.yUnit = y._unitObject
        self.xUncert = x.uncert
        self.yUncert = y.uncert

        # uncertanties can not be 0
        if len(self.xVal) == 1:
            self._sx = self.xUncert if self.xUncert != 0 else 1e-10
        else:
            self._sx = [elem if elem != 0 else 1e-10 for elem in self.xUncert]
        if len(self.yVal) == 1:
            self._sy = self.yUncert if self.yUncert != 0 else 1e-10
        else:
            self._sy = [elem if elem != 0 else 1e-10 for elem in self.yUncert]

        # create the regression
        np.seterr('ignore')
        data = odr.RealData(self.xVal, self.yVal, sx=self._sx, sy=self._sy)
        regression = odr.ODR(data, odr.Model(self._func), beta0=p0)
        regression = regression.run()
        popt = regression.beta
        popt = [0.9 * elem for elem in popt]
        regression = odr.ODR(data, odr.Model(self._func), beta0=popt)
        regression = regression.run()
        

        ## create a list of coefficients
        self.coefficients = []
        units = self.getVariableUnits()
        for i in range(len(regression.beta)):
            var = variable(regression.beta[i], units[i], np.sqrt(regression.cov_beta[i,i]))
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
        self._residuals = y - self.predict(x)
        ss_res = sum(self._residuals**2)
        ss_tot = sum((y - np.mean(y))**2)
        if ss_tot != 0:
            self.r_squared = 1 - (ss_res / ss_tot)
        else:
            self.r_squared = 1
        np.seterr('warn')

    def __str__(self):
        return self.func_name() + ',  ' + self._r2_name()

    def _r2_name(self):
        return f'$R^2 = {self.r_squared.value:.5f}$'

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
            ax.errorbar(self.xVal, self.yVal, xerr=self.xUncert, yerr=self.yUncert, linestyle='', label=label, **kwargs)
        else:
            ax.scatter(self.xVal, self.yVal, label=label, **kwargs)

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
        normRes = scale * self._residuals
        np.seterr('warn')
        ax.scatter(self.xVal, normRes.value, label=label, **kwargs)

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
        
        ax.scatter(self.xVal, self._residuals.value, label=label, **kwargs)

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

        ax.plot(self.xVal, self.yVal, label=label, **kwargs)

    def predict(self, x):
        if not isinstance(x, scalarVariable):
            x = variable(x, self.xUnit)
        return self.func(x)

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
            x = np.linspace(np.min(self.xVal), np.max(self.xVal), 100)
        y = self.predict(x).value
        ax.plot(x, y, label=label, **kwargs)
          
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
        
        pred = self.predict(self.xVal)
        
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
    def __init__(self, x, y):
        
        if not (isinstance(x, arrayVariable) and isinstance(y, arrayVariable)):
            raise ValueError('The inputs has to be variables')

        self.xVal = x.value
        self.yVal = y.value
        self.xUnit = x.unit
        self.yUnit = y.unit
        self.xUncert = x.uncert
        self.yUncert = y.uncert

        self.r_squared = 0

    def _func(self, x):
        return 1

    def func_name(self):
        return f'{self.popt[0]}'

class exp_fit(_fit):
    def __init__(self, x, y, p0=[1, 1]):
        if len(p0) != 2:
            raise ValueError('You have to provide initial guesses for 2 parameters')
        if x.unit != '1':
            raise ValueError('The variable "x" cannot have a unit')
        _fit.__init__(self, self.func, x, y, p0=p0)

    def getVariableUnits(self):
        return [self.yUnit, '']

    def _func(self, B, x):
        a = B[0]
        b = B[1]
        return a * b**x

    def func_name(self):
        return f'$a\cdot b^x,\quad a={self.coefficients[0].__str__(pretty = True)}, \quad b={self.coefficients[1].__str__(pretty = True)}$'


class pow_fit(_fit):
    def __init__(self, x, y, p0=[1, 1]):
        
        if len(p0) != 2:
            raise ValueError('You have to provide initial guesses for 2 parameters')
        if x.unit != '1':
            raise ValueError('The variable "x" cannot have a unit')
        _fit.__init__(self, self.func, x, y, p0=p0)

    def getVariableUnits(self):
        return [self.yUnit, '1']

    def _func(self, B, x):
        a = B[0]
        b = B[1]
        return a * x**b

    def func_name(self):
        return f'$a x^b,\quad a={self.coefficients[0].__str__(pretty = True)}, \quad b={self.coefficients[1].__str__(pretty = True)}$'


def lin_fit(x, y, terms = None, p0=None):
    return pol_fit(x, y, deg=1, terms = terms, p0=p0)


class pol_fit(_fit):
    def __init__(self, x, y, deg=2, terms=None, p0=None):
        if terms is None:
            terms = [True] * (deg + 1)
        else:
            for term in terms:
                if not str(type(term)) == "<class 'bool'>":
                    raise ValueError('All elements in "terms" has to be booleans')
            if len(terms) > deg + 1:
                raise ValueError(f'You have specified to use {len(terms)} terms, but you can only use {deg+1} using a polynomial of degree {deg}')
        self.terms = terms

        if p0 is None:
            p0 = [1] * sum(1 for elem in self.terms if elem)

        self.deg = deg

        _fit.__init__(self, self.func, x, y, p0=p0)

    def getVariableUnits(self):
        units = []
        n = self.deg
        index = 0
        for i in range(n + 1):
            if self.terms[i]:
                u = self.yUnit
                if i != n:
                    ui, _ = self.xUnit ** (n - i)
                    u /= unit(ui)
                units.append(u)
                index += 1
        return units

    def _func(self, B, x):
        out = 0
        n = self.deg
        index = 0
        for i in range(n + 1):
            if self.terms[i]:
                out +=  B[index] * x**(n - i)
                index += 1
        return out

    def func_name(self):
        out = '$'
        n = self.deg
        for i in range(n + 1):
            if self.terms[i]:
                exponent = n - i
                if i == 0:
                    out += f'{string.ascii_lowercase[i]}'
                else:
                    out += f'+{string.ascii_lowercase[i]}'
                if exponent != 0:
                    out += f'x'
                if exponent > 1:
                    out += f'^{exponent}'
        index = 0
        for i in range(n + 1):
            if self.terms[i]:
                out += f', {string.ascii_lowercase[i]}={self.coefficients[index].__str__(pretty = True)}'
                index += 1
        out += '$'
        return out


class logistic_fit(_fit):
    def __init__(self, x, y, p0=[1, 1, 1]):
        if len(p0) != 3:
            raise ValueError('You have to provide initial guesses for 3 parameters')
        if x.unit != '1':
            raise ValueError('The variable "x" cannot have a unit')
        _fit.__init__(self, self.func, x, y, p0=p0)

    def getVariableUnits(self):
        return [self.yUnit,'1','1']
    
    def _func(self, B, x):
        L = B[0]
        k = B[1]
        x0 = B[2]
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

## TODO fit - lås parametre