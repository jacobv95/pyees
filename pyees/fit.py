import numpy as np
import scipy.odr as odr
import string
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
            sx = self.xUncert if self.xUncert != 0 else 1e-10
        else:
            sx = [elem if elem != 0 else 1e-10 for elem in self.xUncert]
        if len(self.yVal) == 1:
            sy = self.yUncert if self.yUncert != 0 else 1e-10
        else:
            sy = [elem if elem != 0 else 1e-10 for elem in self.yUncert]

        # create the regression
        data = odr.RealData(self.xVal, self.yVal, sx=sx, sy=sy)
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
        np.seterr('ignore')
        residuals = self.yVal - self.predict(self.xVal).value
        np.seterr('warn')
        y_bar = np.mean(self.yVal)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.yVal - y_bar)**2)
        if ss_tot != 0:
            self.r_squared = 1 - (ss_res / ss_tot)
        else:
            self.r_squared = 1

    def __str__(self):
        return self.func_name() + ',  ' + self._r2_name()

    def _r2_name(self):
        return f'$R^2 = {self.r_squared:.5f}$'

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

    def predictDifferential(self, x):
        if not isinstance(x, arrayVariable) or isinstance(x, scalarVariable):
            x = variable(x, self.xUnit)
        return self.d_func(self.popt, x)

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
        
    def plotDifferential(self, ax, label=True, x=None, **kwargs):

        # parse label
        if isinstance(label, str):
            label = label
        elif label == True:
            label = self.d_func_name()
        elif label == False:
            label = None
        elif label is None:
            label = None
        else:
            raise ValueError('The label has to be a string, a bool or None')

        if x is None:
            x = np.linspace(np.min(self.xVal), np.max(self.xVal), 100)
        ax.plot(x, self.predDifferential(x), label=label, **kwargs)
        
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


class dummy_fit(_fit):
    def __init__(self, x, y, p0=None):
        
        if not (isinstance(x, arrayVariable) and isinstance(y, arrayVariable)):
            raise ValueError('The inputs has to be variables')

        self.xVal = x.value
        self.yVal = y.value
        self.xUnit = x.unit
        self.yUnit = y.unit
        self.xUncert = x.uncert
        self.yUncert = y.uncert

        self.r_squared = 0
        self.popt = [variable(1, self.yUnit)]

    def func(self, B, x):
        val = self.popt[0].value
        val = [val] * len(x.value)
        return variable(val, self.yUnit)

    def d_func(self, B, x):
        val = [0] * len(x.value)
        unit = (self.popt[0] / variable(1, self.xUnit)).unit
        return variable(val, unit)

    def func_name(self):
        return f'{self.popt[0]}'

    def d_func_name(self):
        unit = (self.popt[0] / variable(1, self.xUnit)).unit
        return f'{variable(0, unit)}'


class exp_fit(_fit):
    def __init__(self, x, y, p0=[1, 1]):
        if len(p0) != 2:
            raise ValueError('You have to provide initial guesses for 2 parameters')
        if x.unit != '1':
            raise ValueError('The variable "x" cannot have a unit')
        _fit.__init__(self, self.func, x, y, p0=p0)

    def getPoptVariables(self):
        a = self.popt[0]
        b = self.popt[1]

        uA = self.uPopt[0]
        uB = self.uPopt[1]

        unitA = self.yUnit
        unitB = '1'

        a = variable(a, unitA, uA)
        b = variable(b, unitB, uB)

        self.popt = [a, b]

    def func(self, B, x):
        a = B[0]
        b = B[1]
        return a * b**x

    def d_func(self, B, x):
        a = B[0]
        b = B[1]
        return a * b**x * np.log(b)

    def d_func_name(self):
        return f'$a\cdot b^x\cdot \ln(b),\quad a=${self.popt[0]}$, \quad b=${self.popt[1]}'

    def func_name(self):
        return f'$a\cdot b^x,\quad a={self.popt[0].__str__(pretty = True)}, \quad b={self.popt[1].__str__(pretty = True)}$'


class pow_fit(_fit):
    def __init__(self, x, y, p0=[1, 1]):
        
        if len(p0) != 2:
            raise ValueError('You have to provide initial guesses for 2 parameters')
        if x.unit != '1':
            raise ValueError('The variable "x" cannot have a unit')
        _fit.__init__(self, self.func, x, y, p0=p0)

    def getPoptVariables(self):
        a = self.popt[0]
        b = self.popt[1]

        uA = self.uPopt[0]
        uB = self.uPopt[1]

        unitA = self.yUnit
        unitB = '1'

        a = variable(a, unitA, uA)
        b = variable(b, unitB, uB)

        self.popt = [a, b]

    def func(self, B, x):
        a = B[0]
        b = B[1]
        return a * x**b

    def d_func(self, B, x):
        a = B[0]
        b = B[1]
        return a * b * x**(b - 1)

    def d_func_name(self):
        return f'$a b x^{{b-1}},\quad a=${self.popt[0]}$, \quad b=${self.popt[1]}'

    def func_name(self):
        return f'$a x^b,\quad a={self.popt[0].__str__(pretty = True)}, \quad b={self.popt[1].__str__(pretty = True)}$'


def lin_fit(x, y, p0=None):
    return pol_fit(x, y, deg=1, p0=p0)


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

    def func(self, x):
        return self._func(self.coefficients, x)

    def _func(self, B, x):
        out = 0
        n = self.deg
        index = 0
        for i in range(n + 1):
            if self.terms[i]:
                out +=  B[index] * x**(n - i)
                index += 1
        return out

    def d_func(self, x):
        return self._d_func(self.coefficients, x)

    def _d_func(self, B, x):
        out = 0
        n = self.deg
        index = 0
        for i in range(n):
            if self.terms[i]:
                out += (n - i) * B[index] * x**(n - i - 1)
        return out

    def d_func_name(self):
        out = ''
        n = self.deg
        for i in range(n):
            if self.terms[i]:
                exponent = n - i - 1
                coefficient = n - i
                if out:
                    out += '+'
                if coefficient != 1:
                    out += f'{coefficient}'

                out += f'{string.ascii_lowercase[i]}'

                if exponent != 0:
                    out += f'$x$'
                if exponent > 1:
                    out += f'$^{exponent}$'

        index = 0
        for i in range(n):
            if self.terms[i]:
                out += f', {string.ascii_lowercase[i]}={self.popt[index].__str__(pretty = True)}'
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

    def getPoptVariables(self):
        L = self.popt[0]
        k = self.popt[1]
        x0 = self.popt[2]

        uL = self.uPopt[0]
        uK = self.uPopt[1]
        uX0 = self.uPopt[2]

        unitL = self.yUnit
        unitK = '1'
        unitX0 = '1'

        L = variable(L, unitL, uL)
        k = variable(k, unitK, uK)
        x0 = variable(x0, unitX0, uX0)

        self.popt = [L, k, x0]

    def func(self, B, x):
        L = B[0]
        k = B[1]
        x0 = B[2]
        return L / (1 + np.exp(-k * (x - x0)))

    def d_func(self, B, x):
        L = B[0]
        k = B[1]
        x0 = B[2]
        return k * L * np.exp(-k * (x - x0)) / ((np.exp(-k * (x - x0)) + 1)**2)

    def d_func_name(self):
        L = self.popt[0]
        k = self.popt[1]
        x0 = self.popt[2]

        out = f'$\\frac{{k\cdot L \cdot e^{{-k\cdot (x-x_0)}}}}{{\\left(1 + e^{{-k\cdot (x-x_0)}}\\right)}}$'
        out += f'$\quad L={L}$, '
        out += f'$\quad k={k}$, '
        out += f'$\quad x_0={x0}$'
        return out

    def func_name(self):
        L = self.popt[0].__str__(pretty=True)
        k = self.popt[1].__str__(pretty=True)
        x0 = self.popt[2].__str__(pretty=True)

        out = f'$\\frac{{L}}{{1 + e^{{-k\cdot (x-x_0)}}}}'
        out += f'\quad L={L}'
        out += f'\quad k={k}'
        out += f'\quad x_0={x0}$'
        return out


if __name__ == "__main__":
    
    
    flow = variable([32.77384783, 41.09968726, 49.26071221, 57.3579077, 65.42974152, 73.39579152, 81.71096102, 89.5224963, 97.35493807], 'L/min', [0.09242640187777096, nan, nan, nan, 0.0997230684346466, 0.16573749880130453, 0.3306259756514912, 0.3428053760615392, 0.4003027546277681])
    dp = variable([43.310, 64.53, 90.0, 118.8, 151.92, 189.4, 231.19, 276.1, 323.4], 'mbar',[0.002, 0.08, 0.1, 0.2, 0.09, 0.6, 0.06, 0.6, 0.4])
    
    fit = pol_fit(flow, dp)
    print(fit)    



## TODO fit - få det til at virke som i bogen - brug tests - mangler solutions
## TODO fit - lav normalized residuals plot
## TODO fit - evaluer, om 68% af datapunkterne ligger 1 standard afvigelse fra fittet

## TODO fit - lås parametre