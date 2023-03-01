from copy import deepcopy
import numpy as np
try:
    from .unit import unit
except ImportError:
    from unit import unit
    
HANDLED_FUNCTIONS = {}


class variable():
    def __init__(self, value, unitStr='', uncert=None, nDigits=3) -> None:

        # create a unit object
        self._unitObject = unitStr if isinstance(unitStr, unit) else unit(unitStr)

        # number of digits to show
        self.nDigits = nDigits

        # store the value and the uncertaty
        def evaluateInput(input):
            if isinstance(input, np.ndarray):
                return input
            else:
                listLike = False
                try:
                    len(input)
                    listLike = True
                except:
                    pass
                if listLike:
                    return np.array(input, dtype=float)
                else:
                    return np.array([input], dtype=float)
        self._value = evaluateInput(value)
        if uncert is None:
            self._uncert = evaluateInput([0] * self.len())
        else:
            self._uncert = evaluateInput(uncert)

        # check the length of the uncertanty and the value
        if self.len() != len(self._uncert):
            raise ValueError('The lenght of the value has to be equal to the lenght of the uncertanty')

        # value and unit in SI. This is used when determining the gradient in the uncertanty expression
        self._getConverterToSI()

        # uncertanty
        self.dependsOn = {}
        self.covariance = {}

    def _getConverterToSI(self):
        self._converterToSI = self._unitObject._converterToSI

    @property
    def value(self):
        if self.len() == 1:
            return self._value[0]
        return self._value

    @property
    def unit(self):
        return str(self._unitObject)

    @property
    def uncert(self):
        if self.len() == 1:
            return self._uncert[0]
        return self._uncert

    def convert(self, newUnit):
        converter = self._unitObject.getConverter(newUnit)
        self._value = converter.convert(self._value, useOffset=not self._unitObject.isCombinationUnit())
        self._uncert = converter.convert(self._uncert, useOffset=False)
        self._unitObject.convert(newUnit)

        # update the converter to SI
        self._getConverterToSI()

    def len(self):
        return len(self._value)

    def __getitem__(self, index):
        if isinstance(index, int) or isinstance(index, slice):
            if index >= self.len():
                raise IndexError('Index out of bounds')
            if self.len() == 1:
                if index != 0:
                    raise IndexError('Index out of bound')
                return variable(self.value, self._unitObject, self.uncert)
            return variable(self.value[index], self._unitObject, self.uncert[index])
        else:
            val = [self.value[elem]for elem in index]
            unc = [self.uncert[elem]for elem in index]
            return variable(val, self._unitObject, unc)

    def printUncertanty(self, value, uncert):
        # function to print number
        if uncert == 0 or uncert is None or np.isnan(uncert):
            return f'{value:.{self.nDigits}g}', None

        digitsUncert = -int(np.floor(np.log10(np.abs(uncert))))

        if value != 0:
            digitsValue = -int(np.floor(np.log10(np.abs(value))))
        else:
            digitsValue = 0

        # uncertanty
        if digitsUncert > 0:
            uncert = f'{uncert:.{1}g}'
        else:
            nDecimals = len(str(int(uncert)))
            uncert = int(np.around(uncert, -nDecimals + 1))

        # value
        if digitsValue <= digitsUncert:
            if digitsUncert > 0:
                value = f'{value:.{digitsUncert}f}'
            else:
                value = int(np.around(value, - nDecimals + 1))
        else:
            value = '0'
            if digitsUncert > 0:
                value += '.' + ''.join(['0'] * digitsUncert)

        return value, uncert

    def __str__(self, pretty=None) -> str:

        # standard values
        uncert = None
        unitStr = self._unitObject.__str__(pretty=pretty)

        if pretty:
            pm = r'\pm'
            space = r'\ '
            squareBracketLeft = r'\left ['
            squareBracketRight = r'\right ]'

        else:
            pm = '+/-'
            squareBracketLeft = '['
            squareBracketRight = ']'
            space = ' '

        if unitStr == '1':
            unitStr = ''
        else:
            unitStr = rf'{squareBracketLeft}{unitStr}{squareBracketRight}'

        if self.len() == 1:
            # print a single value
            value = self.value
            if self.uncert != 0:
                uncert = self.uncert

            value, uncert = self.printUncertanty(value, uncert)
            if uncert is None:
                return rf'{value}{space}{unitStr}'
            else:
                return rf'{value} {pm} {uncert}{space}{unitStr}'

        else:
            # print array of values
            valStr = []
            uncStr = []
            for v, u in zip(self._value, self._uncert):
                v, u = self.printUncertanty(v, u)
                valStr.append(v)
                uncStr.append(u)

            if all(self._uncert == 0) or all(elem is None for elem in self._uncert):
                out = rf''
                out += rf'['
                for i, elem in enumerate(valStr):
                    out += rf'{elem}'
                    if i != len(valStr) - 1:
                        out += rf', '
                out += rf']'
                out += rf'{space}{unitStr}'
                return out
            else:
                # find number of significant digits in uncertanty
                out = rf''
                out += rf'['
                for i, elem in enumerate(valStr):
                    out += rf'{elem}'
                    if i != len(valStr) - 1:
                        out += r', '
                out += rf']'
                out += rf' {pm} '
                out += rf'['
                for i, elem in enumerate(uncStr):
                    out += rf'{elem}'
                    if i != len(uncStr) - 1:
                        out += r', '
                out += rf']'
                out += rf'{space}{unitStr}'
                return out

    def _addDependents(self, vars, grads):
        # copy the gradients
        # this prevents a pointer error as the gradients often are set to the value of a variable
        # when the gradients are scaled this causes problems
        grads = [deepcopy(elem) for elem in grads]

        # loop over the variables and their gradients
        for var, grad in zip(vars, grads):
            # scale the gradient to SI units. This is necessary if one of the variables are converted after the dependency has been noted
            scale = self._converterToSI.convert(1, useOffset=False) / var._converterToSI.convert(1, useOffset=False)
            grad *= scale

            # check if the variable depends on other variables
            if var.dependsOn:

                # loop over the dependencies of the variables and add them to the dependencies of self.
                # this ensures that the product rule is used
                for key, item in var.dependsOn.items():
                    if key in self.dependsOn:
                        self.dependsOn[key] += item * grad
                    else:
                        self.dependsOn[key] = item * grad
            else:

                # the variable did not have any dependecies. Therefore the the varaible is added to the dependencies of self
                if var in self.dependsOn:
                    self.dependsOn[var] += grad
                else:
                    self.dependsOn[var] = grad

    def _addCovariance(self, var, covariance):
        self.covariance[var] = covariance

    def _calculateUncertanty(self):

        # variance from each measurement
        if self.len() == 0:
            variance = 0
        else:
            variance = np.zeros(self.len())
        selfScaleToSI = self._converterToSI.convert(1, useOffset=False)
        for var, grad in self.dependsOn.items():
            # the gradient is scaled with the inverse of the conversion of the unit to SI units.
            # This is necessary if the variable "var" has been converted after the dependency has been noted
            scale = var._converterToSI.convert(1, useOffset=False) / selfScaleToSI
            variance += (grad * scale * var.uncert)**2

        # variance from the corralation between measurements
        n = len(self.dependsOn.keys())
        for i in range(n):
            var_i = list(self.dependsOn.keys())[i]
            for j in range(i + 1, n):
                var_j = list(self.dependsOn.keys())[j]
                if var_j in var_i.covariance.keys():
                    if not var_i in var_j.covariance.keys():
                        raise ValueError(
                            f'The variable {var_i} is correlated with the varaible {var_j}. However the variable {var_j} not not correlated with the variable {var_i}. Something is wrong.')
                    scale_i = var_i._converterToSI.convert(1, useOffset=False) / selfScaleToSI
                    scale_j = var_j._converterToSI.convert(1, useOffset=False) / selfScaleToSI
                    varianceContribution = 2 * scale_i * self.dependsOn[var_i] * scale_j * self.dependsOn[var_j] * var_i.covariance[var_j][0]
                    variance += varianceContribution

        self._uncert = np.sqrt(variance)
        
    def __add__(self, other):
        
        if not isinstance(other, variable):
            return self + variable(other, self.unit)

        # determine if the two variable can be added
        addBool, outputUnit = self._unitObject + other._unitObject
        if not addBool:
            raise ValueError(f'You tried to add a variable in [{self.unit}] to a variable in [{other.unit}], but the units do not have the same SI base unit')

        # convert self and other to the SI unit system
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(other._unitObject._SIBaseUnit)

        # determine the value and gradients
        val = self._value + other._value
        grad = [1, 1]
        vars = [self, other]

        # create the new variable
        var = variable(val, outputUnit)
        var._addDependents(vars, grad)
        var._calculateUncertanty()

        # convert self and other back
        self.convert(selfUnit)
        other.convert(otherUnit)

        # convert the output variable
        if outputUnit == 'K':
            SIBaseUnits = [self._unitObject._SIBaseUnit, other._unitObject._SIBaseUnit]
            if 'DELTAK' in SIBaseUnits:
                outputUnit = [self.unit, other.unit][SIBaseUnits.index('K')]
                var.convert(outputUnit)
        if self.unit == other.unit:
            var.convert(self.unit)
        return var

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if not isinstance(other, variable):
            return self - variable(other, self.unit)

        # determine if the variables can be subtracted
        subBool, outputUnit = self._unitObject - other._unitObject
        if not subBool:
            raise ValueError(f'You tried to subtract a variable in [{other.unit}] from a variable in [{self.unit}], but the units do not have the same SI base unit')

        # convert self and other to the SI unit system
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(other._unitObject._SIBaseUnit)

        # determine the value and gradients
        val = self.value - other.value
        grad = [1, -1]
        vars = [self, other]

        # create the new variable
        var = variable(val, outputUnit)
        var._addDependents(vars, grad)
        var._calculateUncertanty()

        # convert self and other back
        self.convert(selfUnit)
        other.convert(otherUnit)

        # convert the output variable
        if outputUnit == 'K':
            SIBaseUnits = [self._unitObject._SIBaseUnit, other._unitObject._SIBaseUnit]
            if 'DELTAK' in SIBaseUnits:
                outputUnit = [self.unit, other.unit][SIBaseUnits.index('K')]
                var.convert(outputUnit)
        if self.unit == other.unit and 'DELTA' not in outputUnit:
            var.convert(self.unit)

        return var

    def __rsub__(self, other):
        return - self + other

    def __mul__(self, other):
        if not isinstance(other, variable):
            return self * variable(other)

        val = self.value * other.value
        outputUnit = self._unitObject * other._unitObject

        grad = [other._value, self._value]
        vars = [self, other]

        var = variable(val, outputUnit)
        var._addDependents(vars, grad)
        var._calculateUncertanty()

        if var._unitObject._SIBaseUnit == '1' and var._unitObject != '1':
            var.convert('1')

        return var

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        if not isinstance(other, variable):
            return self ** variable(other)

        if other.len() != 1:
            raise ValueError('The exponent has to be a single number')
        if str(other.unit) != '1':
            raise ValueError('The exponent can not have a unit')

        val = self._value ** other.value
        outputUnit = self._unitObject ** other.value

        gradSelf = other._value * self._value ** (other._value - 1)
        def gradOther(valSelf, valOther, uncertOther):
            if uncertOther != 0:
                return valSelf ** valOther * np.log(valSelf)
            else:
                return 0
        gradOther = np.vectorize(gradOther, otypes=[float])(self._value, other._value, other._uncert)

        grad = [gradSelf, gradOther]
        vars = [self, other]

        var = variable(val, outputUnit)
        var._addDependents(vars, grad)
        var._calculateUncertanty()
        return var

    def __rpow__(self, other):
        return variable(other, '1') ** self

    def __truediv__(self, other):
        if not isinstance(other, variable):
            return self / variable(other)

        val = self._value / other._value
        outputUnit = self._unitObject / other._unitObject

        grad = [1 / other._value, -self._value / (other._value**2)]
        vars = [self, other]

        var = variable(val, outputUnit)
        var._addDependents(vars, grad)
        var._calculateUncertanty()

        if var._unitObject._SIBaseUnit == '1' and var._unitObject != '1':
            var.convert('1')

        return var

    def __rtruediv__(self, other):
        if not isinstance(other, variable):
            return variable(other) / self

        val = other._value / self._value
        outputUnit = other._unitObject / self._unitObject

        grad = [-other._value / (self._value**2), 1 / (self._value)]
        vars = [self, other]

        var = variable(val, outputUnit)
        var._addDependents(vars, grad)
        var._calculateUncertanty()

        if var._unitObject._SIBaseUnit == '1' and var._unitObject != '1':
            var.convert('1')

        return var

    def __neg__(self):
        return -1 * self

    def log(self):
        if self.unit != '1':
            raise ValueError('You can only take the natural log of a variable if it has no unit')

        val = np.log(self.value)

        vars = [self]
        grad = [1 / self.value]

        var = variable(val, '1')
        var._addDependents(vars, grad)
        var._calculateUncertanty()

        return var

    def log10(self):
        if self.unit != '1':
            raise ValueError('You can only take the base 10 log of a variable if it has no unit')
        val = np.log10(self.value)

        vars = [self]
        grad = [1 / (self.value * np.log10(self.value))]

        var = variable(val, '1')
        var._addDependents(vars, grad)
        var._calculateUncertanty()

        return var

    def exp(self):
        return np.e**self

    def sqrt(self):
        return self**(1 / 2)

    def sin(self):
        if str(self._unitObject._SIBaseUnit) != 'rad':
            raise ValueError('You can only take sin of an angle')

        outputUnit = '1'
        if self._unitObject._assertEqual('rad'):
            val = np.sin(self.value)
            grad = [np.cos(self.value)]
        else:
            val = np.sin(np.pi / 180 * self.value)
            grad = [np.pi / 180 * np.cos(np.pi / 180 * self.value)]

        vars = [self]

        var = variable(val, outputUnit)
        var._addDependents(vars, grad)
        var._calculateUncertanty()

        return var

    def cos(self):
        if str(self._unitObject._SIBaseUnit) != 'rad':
            raise ValueError('You can only take cos of an angle')

        outputUnit = '1'
        if self.unit == 'rad':
            val = np.cos(self.value)
            grad = [-np.sin(self.value)]
        else:
            val = np.cos(np.pi / 180 * self.value)
            grad = [-np.pi / 180 * np.sin(np.pi / 180 * self.value)]

        vars = [self]

        var = variable(val, outputUnit)
        var._addDependents(vars, grad)
        var._calculateUncertanty()

        return var

    def tan(self):
        if str(self._unitObject._SIBaseUnit) != 'rad':
            raise ValueError('You can only take tan of an angle')

        outputUnit = '1'
        if self.unit == 'rad':
            val = np.tan(self.value)
            grad = [2 / (np.cos(2 * self.value) + 1)]
        else:
            val = np.tan(np.pi / 180 * self.value)
            grad = [np.pi / (90 * (np.cos(np.pi / 90 * self.value) + 1))]

        vars = [self]

        var = variable(val, outputUnit)
        var._addDependents(vars, grad)
        var._calculateUncertanty()

        return var

    def __abs__(self):
        return variable(np.abs(self.value), self.unit, self.uncert)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle Physical objects
        if not all(issubclass(t, variable) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


    def _compareLengths(self, other):
        if self.len() != other.len():
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same length')

    def __lt__(self, other):
        if not isinstance(other, variable):
            return self < variable(other, self.unit)

        if not self._unitObject._SIBaseUnit == other._unitObject._SIBaseUnit:
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same SI base unit')
        
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(self._unitObject._SIBaseUnit)

        if self.len() == 1 and other.len() == 1:
            out =  self.value < other.value
        elif self.len() == 1:
            out = list(self.value < other.value)
        elif other.len() == 1:
            out = list(self.value < other.value)
        else:
            self._compareLengths(other)
            out = [elemA < elemB for elemA, elemB in zip(self.value, other.value)]
        
        self.convert(selfUnit)
        other.convert(otherUnit)
        return out

    def __le__(self, other):
        if not isinstance(other, variable):
            return self <= variable(other, self.unit)
        if not self._unitObject._SIBaseUnit == other._unitObject._SIBaseUnit:
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same SI base unit')
        
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(self._unitObject._SIBaseUnit)
        
        if self.len() == 1 and other.len() == 1:
            out = self.value <= other.value
        elif self.len() == 1:
            out = list(self.value <= other.value)
        elif other.len() == 1:
            out = list(self.value <= other.value)
        else:
            self._compareLengths(other)
            out = [elemA <= elemB for elemA, elemB in zip(self.value, other.value)]

        self.convert(selfUnit)
        other.convert(otherUnit)
        return out
    
    def __gt__(self, other):
        if not isinstance(other, variable):
            return self > variable(other, self.unit)

        if not self._unitObject._SIBaseUnit == other._unitObject._SIBaseUnit:
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same SI base unit')
        
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(self._unitObject._SIBaseUnit)

        if self.len() == 1 and other.len() == 1:
            out =  self.value > other.value
        elif self.len() == 1:
            out =  list(self.value > other.value)
        elif other.len() == 1:
            out =  list(self.value > other.value)
        else:
            self._compareLengths(other)
            out =  [elemA > elemB for elemA, elemB in zip(self.value, other.value)]
        
        self.convert(selfUnit)
        other.convert(otherUnit)
        return out
   
    def __ge__(self, other):
        if not isinstance(other, variable):
            return self >= variable(other, self.unit)

        if not self._unitObject._SIBaseUnit == other._unitObject._SIBaseUnit:
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same SI base unit')
        
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(self._unitObject._SIBaseUnit)

        if self.len() == 1 and other.len() == 1:
            out =  self.value >= other.value
        elif self.len() == 1:
            out =  list(self.value >= other.value)
        elif other.len() == 1:
            out =  list(self.value >= other.value)
        else:
            self._compareLengths(other)
            out =  [elemA >= elemB for elemA, elemB in zip(self.value, other.value)]

        self.convert(selfUnit)
        other.convert(otherUnit)
        return out

    def __eq__(self, other):
        if not isinstance(other, variable):
            return self == variable(other, self.unit)

        if not self._unitObject._SIBaseUnit == other._unitObject._SIBaseUnit:
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same SI base unit')
        
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(self._unitObject._SIBaseUnit)

        if self.len() == 1 and other.len() == 1:
            out =  self.value == other.value
        elif self.len() == 1:
            out =  list(self.value == other.value)
        elif other.len() == 1:
            out =  list(self.value == other.value)
        else:
            self._compareLengths(other)
            out =  [elemA == elemB for elemA, elemB in zip(self.value, other.value)]

        self.convert(selfUnit)
        other.convert(otherUnit)
        return out

    def __ne__(self, other):
        if not isinstance(other, variable):
            return self != variable(other, self.unit)

        if not self._unitObject._SIBaseUnit == other._unitObject._SIBaseUnit:
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same SI base unit')
        
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(self._unitObject._SIBaseUnit)

        if self.len() == 1 and other.len() == 1:
            out = self.value != other.value
        elif self.len() == 1:
            out = list(self.value != other.value)
        elif other.len() == 1:
            out = list(self.value != other.value)
        else:
            self._compareLengths(other)
            out = [elemA != elemB for elemA, elemB in zip(self.value, other.value)]

        self.convert(selfUnit)
        other.convert(otherUnit)
        return out

    def __hash__(self):
        return id(self)


def implements(numpy_function):
    """Register an __array_function__ implementation for Physical objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator


@implements(np.max)
def np_max_for_variable(x, *args, **kwargs):
    if x.len() > 1:
        index = np.argmax(x.value)
        val = x.value[index]
        if not x.uncert is None:
            unc = x.uncert[index]
        else:
            unc = None
    else:
        val = x.value
        if not x.uncert is None:
            unc = x.uncert
        else:
            unc = None
    return variable(val, x.unit, unc)


@implements(np.min)
def np_max_for_variable(x, *args, **kwargs):
    if x.len() > 1:
        index = np.argmin(x.value)
        val = x.value[index]
        if not x.uncert is None:
            unc = x.uncert[index]
        else:
            unc = None
    else:
        val = x.value
        if not x.uncert is None:
            unc = x.uncert
        else:
            unc = None
    return variable(val, x.unit, unc)


@implements(np.mean)
def np_mean_for_variable(x, *args, **kwargs):
    if x.len() > 1:
        val = np.mean(x.value)
        if not x.uncert is None:
            n = len(x.uncert)
            unc = np.sqrt(sum([(1 / n * elem)**2 for elem in x.uncert]))
        else:
            unc = None
    else:
        val = x.value
        if not x.uncert is None:
            unc = x.uncert
        else:
            unc = None
    return variable(val, x.unit, unc)

