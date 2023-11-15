
import numpy as np
try:
    from unit import unit
except ImportError:
    from pyees.unit import unit


class logarithmicVariables:
    def __init__(self):
        pass

    @staticmethod
    def __add__(a, b):
        aUnit = a.unit
        bUnit = b.unit

        aUnitWithoutPrefix = a._unitObject.getUnitWithoutPrefix()
        a.convert(aUnitWithoutPrefix)
        aConverter = a._unitObject.getLogarithmicConverter()
        a._unitStr = '1'
        a._unitObject = unit('')
        for elem in a:
            elem._unitStr = '1'
            elem._unitObject = unit('')
        aConverter.convertToSignal(a)

        bUnitWithoutPrefix = b._unitObject.getUnitWithoutPrefix()
        b.convert(bUnitWithoutPrefix)
        bConverter = b._unitObject.getLogarithmicConverter()
        b._unitStr = '1'
        b._unitObject = unit('')
        for elem in b:
            elem._unitStr = '1'
            elem._unitObject = unit('')
        bConverter.convertToSignal(b)

        c = a + b

        aConverter.convertFromSignal(a)
        a._unitStr = aUnitWithoutPrefix
        a._unitObject = unit(aUnitWithoutPrefix)
        for elem in a:
            elem._unitStr = aUnitWithoutPrefix
            elem._unitObject = unit(aUnitWithoutPrefix)
        a.convert(aUnit)

        bConverter.convertFromSignal(b)
        b._unitStr = bUnitWithoutPrefix
        b._unitObject = unit(bUnitWithoutPrefix)
        for elem in b:
            elem._unitStr = bUnitWithoutPrefix
            elem._unitObject = unit(bUnitWithoutPrefix)
        b.convert(bUnit)

        aConverter.convertFromSignal(c)
        c._unitStr = aUnitWithoutPrefix
        c._unitObject = unit(aUnitWithoutPrefix)
        for elem in c:
            elem._unitStr = aUnitWithoutPrefix
            elem._unitObject = unit(aUnitWithoutPrefix)
        c.convert(aUnit)

        return c

    @staticmethod
    def __sub__(a, b):
        aUnit = a.unit
        bUnit = b.unit

        aUnitWithoutPrefix = a._unitObject.getUnitWithoutPrefix()
        a.convert(aUnitWithoutPrefix)
        aConverter = a._unitObject.getLogarithmicConverter()
        a._unitStr = '1'
        a._unitObject = unit('')
        for elem in a:
            elem._unitStr = '1'
            elem._unitObject = unit('')
        aConverter.convertToSignal(a)

        bUnitWithoutPrefix = b._unitObject.getUnitWithoutPrefix()
        b.convert(bUnitWithoutPrefix)
        bConverter = b._unitObject.getLogarithmicConverter()
        b._unitStr = '1'
        b._unitObject = unit('')
        for elem in b:
            elem._unitStr = '1'
            elem._unitObject = unit('')
        bConverter.convertToSignal(b)

        c = a - b

        aConverter.convertFromSignal(a)
        a._unitStr = aUnitWithoutPrefix
        a._unitObject = unit(aUnitWithoutPrefix)
        for elem in a:
            elem._unitStr = aUnitWithoutPrefix
            elem._unitObject = unit(aUnitWithoutPrefix)
        a.convert(aUnit)

        bConverter.convertFromSignal(b)
        b._unitStr = bUnitWithoutPrefix
        b._unitObject = unit(bUnitWithoutPrefix)
        for elem in b:
            elem._unitStr = bUnitWithoutPrefix
            elem._unitObject = unit(bUnitWithoutPrefix)
        b.convert(bUnit)

        aConverter.convertFromSignal(c)
        c._unitStr = aUnitWithoutPrefix
        c._unitObject = unit(aUnitWithoutPrefix)
        for elem in c:
            elem._unitStr = aUnitWithoutPrefix
            elem._unitObject = unit(aUnitWithoutPrefix)
        c.convert(aUnit)

        return c

    @staticmethod
    def mean(a):
        aUnit = a.unit

        aUnitWithoutPrefix = a._unitObject.getUnitWithoutPrefix()
        a.convert(aUnitWithoutPrefix)
        aConverter = a._unitObject.getLogarithmicConverter()
        a._unitStr = '1'
        a._unitObject = unit('')
        for elem in a:
            elem._unitStr = '1'
            elem._unitObject = unit('')
        aConverter.convertToSignal(a)

        b = np.mean(a)

        aConverter.convertFromSignal(a)
        a._unitStr = aUnitWithoutPrefix
        a._unitObject = unit(aUnitWithoutPrefix)
        for elem in a:
            elem._unitStr = aUnitWithoutPrefix
            elem._unitObject = unit(aUnitWithoutPrefix)
        a.convert(aUnit)

        aConverter.convertFromSignal(b)
        b._unitStr = aUnitWithoutPrefix
        b._unitObject = unit(aUnitWithoutPrefix)
        for elem in b:
            elem._unitStr = aUnitWithoutPrefix
            elem._unitObject = unit(aUnitWithoutPrefix)
        b.convert(aUnit)

        return b


class scalarVariable():
    def __init__(self, value, unitObject: str | unit, uncert, nDigits) -> None:

        self._value = value
        self._uncert = uncert

        # create a unit object
        self._unitObject = unitObject if isinstance(
            unitObject, unit) else unit(unitObject)
        self._uncertSI = self.uncert * self._unitObject._converterToSI.scale

        # number of digits to show
        self._nDigits = nDigits

        # uncertanty
        self.dependsOn = {}
        self.covariance = {}

    @property
    def nDigits(self):
        return self._nDigits

    @nDigits.setter
    def nDigits(self, nDigits):
        self._nDigits = nDigits

    @property
    def value(self):
        return self._value

    @property
    def unit(self):
        return str(self._unitObject)

    @property
    def uncert(self):
        return self._uncert

    def convert(self, newUnit):
        converter = self._unitObject.getConverter(newUnit)
        self._value = converter.convert(
            self._value, useOffset=not self._unitObject.isCombinationUnit())
        self._uncert = self._uncert * converter.scale
        self._unitObject = unit(newUnit)

    def printUncertanty(self, value, uncert):
        # function to print number
        if uncert == 0 or uncert is None or np.isnan(uncert):
            return f'{value:.{self._nDigits}g}', None

        digitsUncert = -int(np.floor(np.log10(np.abs(uncert))))
        uncert = float(f'{uncert:.{1}g}')
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
            value = str(np.around(value, digitsUncert))
            if '.' in value and digitsUncert <= 0:
                value = value.split('.')[0]

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

        # print a single value
        value = self.value
        if self.uncert != 0:
            uncert = self.uncert

        value, uncert = self.printUncertanty(value, uncert)
        if uncert is None:
            return rf'{value}{space}{unitStr}'
        else:
            return rf'{value} {pm} {uncert}{space}{unitStr}'

    def __repr__(self) -> str:
        return self.__str__()

    def _addDependent(self, var, grad):

        # scale the gradient to SI units. This is necessary if one of the variables are converted after the dependency has been noted
        grad *= self._unitObject._converterToSI.scale / \
            var._unitObject._converterToSI.scale
        self.__addDependent(var, grad)

    def __addDependent(self, var, grad):

        if not var.dependsOn:
            # the variable does not depend on other variables
            # add the variable to the dependencies of self
            self.___addDependent(var, grad)
            return

        # the variable depends on other variables
        # loop over the dependencies of the variables and add them to the dependencies of self.
        # this ensures that the product rule is used
        for vvar, dependency in var.dependsOn.items():
            self.___addDependent(vvar, grad * dependency)

    def ___addDependent(self, var, grad):
        if var in self.dependsOn:
            # the variable is already in the dependencies of self
            # increment the gradient
            # this makes sure that the productrule of differentiation is followed
            self.dependsOn[var] += grad
            return

        # the variable is not in the dependencies of self.
        # add it to the dictionary
        self.dependsOn[var] = grad

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n == 0:
            self.n += 1
            return self
        raise StopIteration

    def addCovariance(self, var, covariance: float, unitStr: str):

        covUnit = unit(unitStr)
        selfVarUnit = self._unitObject * var._unitObject
        if not covUnit.unitDictSI == selfVarUnit.unitDictSI:
            raise ValueError(
                f'The covariance of {covariance} [{unitStr}] does not match the units of {self} and {var}')

        covariance = covariance * covUnit._converterToSI.scale

        self.covariance[var] = covariance
        var.covariance[self] = covariance

    def _calculateUncertanty(self):

        variance = 0
        for var, grad in self.dependsOn.items():
            # variance from the measurements
            variance += (var._uncertSI * grad)**2

            # variance from the corralation between measurements
            # covariance = dict(var: covariance)
            # the gradients can be found in self.dependsOn
            # loop over all variables that are in var.covariance and also self.dependsOn
            if not var.covariance:
                continue
            for var2 in filter(lambda x: x in self.dependsOn, var.covariance):
                variance += grad * self.dependsOn[var2] * var.covariance[var2]

        # all variances are determined in the SI unit system.
        # these has to be converted back in the the unit of self
        self._uncertSI = np.sqrt(variance)
        self._uncert = self._uncertSI / self._unitObject._converterToSI.scale

    def __add__(self, other):

        if not isinstance(other, scalarVariable):
            return self + variable(other, self.unit)

        # determine if the two variable can be added
        outputUnit = self._unitObject + other._unitObject

        # handle logarithmic addition seperately
        if outputUnit.isLogarithmicUnit():
            return logarithmicVariables.__add__(self, other)

        # convert self and other
        selfUnit = self._unitObject
        otherUnit = other._unitObject

        # convert the units if the SI unit is identical to that of the output unit and the unit is not equal to the output unit
        if self._unitObject.unitDictSI == outputUnit.unitDictSI and not self._unitObject == outputUnit:
            self.convert(str(outputUnit))
        if other._unitObject.unitDictSI == outputUnit.unitDictSI and not other._unitObject == outputUnit:
            other.convert(str(outputUnit))

        # determine the value
        val = self.value + other.value

        # create the new variable and add the self and other as dependencies
        var = variable(val, outputUnit)
        var._addDependent(self, 1)
        var._addDependent(other, 1)
        var._calculateUncertanty()

        # convert all units back to their original units
        if not self._unitObject == selfUnit:
            self.convert(str(selfUnit))
        if not other._unitObject == otherUnit:
            other.convert(str(otherUnit))

        # if the output unit is 'K' and one of the inputs is a temperature and the other is a tempreature difference
        # then return the output in the same unit as the temperature
        # this has no affect if the inputs are in 'K' and 'DeltaK'
        # but the output is converted to 'C' if the inputs are in 'C' and 'DeltaK'
        SIBaseUnits = [self._unitObject.unitDictSI,
                       other._unitObject.unitDictSI]
        if outputUnit.unitDict == {'K': {'': 1}} and {'DELTAK': {'': 1}} in SIBaseUnits and {'K': {'': 1}} in SIBaseUnits:
            var.convert(str([selfUnit, otherUnit]
                        [SIBaseUnits.index({'K': {'': 1}})]))

        return var

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if not isinstance(other, scalarVariable):
            return self - variable(other, self.unit)

        # determine if the variables can be subtracted
        outputUnit = self._unitObject - other._unitObject

        # handle logarithmic unit seperately
        if outputUnit.isLogarithmicUnit():
            return logarithmicVariables.__sub__(self, other)

        # convert self and other
        selfUnit = self._unitObject
        otherUnit = other._unitObject

        # convert the units if the SI unit is identical to that of the output unit and the unit is not equal to the output unit
        if self._unitObject.unitDictSI == outputUnit.unitDictSI and not self._unitObject == outputUnit:
            self.convert(str(outputUnit))
        if other._unitObject.unitDictSI == outputUnit.unitDictSI and not other._unitObject == outputUnit:
            other.convert(str(outputUnit))

        # determine the value
        val = self.value - other.value

        # create the new variable and add the self and other as dependencies
        var = variable(val, outputUnit)
        var._addDependent(self, 1)
        var._addDependent(other, -1)
        var._calculateUncertanty()

        # convert all units back to their original units
        if not self._unitObject == selfUnit:
            self.convert(str(selfUnit))
        if not other._unitObject == otherUnit:
            other.convert(str(otherUnit))

        SIBaseUnits = [self._unitObject.unitDictSI,
                       other._unitObject.unitDictSI]
        if outputUnit.unitDictSI == {'K': {'': 1}} and SIBaseUnits[0] == {'K': {'': 1}} and SIBaseUnits[1] == {'K': {'': 1}}:
            if list(self._unitObject.unitDict.keys())[0] == list(other._unitObject.unitDict.keys())[0]:
                var._unitObject = unit(
                    'DELTA' + list(self._unitObject.unitDict.keys())[0])
            else:
                var._unitObject = unit('DELTAK')
            for elem in var:
                elem._unitObject = var._unitObject
        return var

    def __rsub__(self, other):
        return - self + other

    def __mul__(self, other):

        if not isinstance(other, scalarVariable):
            return self * variable(other)

        # determine the value
        val = self.value * other.value

        # determine the output unit
        outputUnit = self._unitObject * other._unitObject

        # create a variable
        var = variable(val, outputUnit)
        var._addDependent(self, other.value)
        var._addDependent(other, self.value)
        var._calculateUncertanty()

        # if all units were cancled during the multiplication, then convert to 1
        # this will remove any remaining prefixes
        if var._unitObject.unitDictSI == {'1': {'': 1}} and var._unitObject.unitDict != {'1': {'': 1}}:
            var.convert('1')

        return var

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):

        if not isinstance(other, scalarVariable):
            return self ** variable(other, '')

        if str(other.unit) != '1':
            raise ValueError('The exponent can not have a unit')

        # determine the output unit
        outputUnit = self._unitObject ** other.value

        # if self._unitObject has the same keys, as the output unit
        # then self does not have to be scaled to the SI unit system in order for the unit to be raised to the power
        hasToBeScaledToSI = self._unitObject.unitDict.keys() != outputUnit.unitDict.keys()
        if hasToBeScaledToSI:
            selfUnit = self.unit
            self.convert(self._unitObject.unitStrSI)

        # dertermine the value
        val = self.value ** other.value

        # determine the gradients of the output with respoect to self and other
        def gradSelf(valSelf, valOTher):
            if valSelf != 0:
                return valOTher * valSelf ** (valOTher - 1)
            return 0

        def gradOther(valSelf, valOther, uncertOther):
            if uncertOther != 0:
                return valSelf ** valOther * np.log(valSelf)
            return 0

        gradOther = np.vectorize(gradOther, otypes=[float])(
            self.value, other.value, other._uncert)
        gradSelf = np.vectorize(gradSelf, otypes=[float])(
            self.value, other.value)

        # create a new variable
        var = variable(val, outputUnit)
        var._addDependent(self, gradSelf)
        var._addDependent(other, gradOther)
        var._calculateUncertanty()

        # converte self back if it was converted to the SI unit system
        if hasToBeScaledToSI:
            self.convert(selfUnit)

        return var

    def __rpow__(self, other):
        return variable(other, '1') ** self

    def __truediv__(self, other):
        if not isinstance(other, scalarVariable):
            return self / variable(other)

        # determine the value
        val = self.value / other.value

        # determine the output unit
        outputUnit = self._unitObject / other._unitObject

        # create a variable
        var = variable(val, outputUnit)
        var._addDependent(self, 1 / other.value)
        var._addDependent(other, -self.value / (other.value**2))
        var._calculateUncertanty()

        # if all units were cancled during the multiplication, then convert to 1
        # this will remove any remaining prefixes
        if var._unitObject.unitDictSI == {'1': {'': 1}} and var._unitObject.unitDict != {'1': {'': 1}}:
            var.convert('1')

        return var

    def __rtruediv__(self, other):
        if not isinstance(other, scalarVariable):
            return variable(other) / self

        val = other.value / self.value
        outputUnit = other._unitObject / self._unitObject

        var = variable(val, outputUnit)
        var._addDependent(self, -other.value / (self.value**2))
        var._addDependent(other, 1 / (self.value))
        var._calculateUncertanty()

        if var._unitObject.unitDictSI == {'1': {'': 1}} and var._unitObject.unitDict != {'1': {'': 1}}:
            var.convert('1')

        return var

    def __neg__(self):
        out = -1 * self
        out.convert(self.unit)
        return out

    def log(self):
        if self.unit != '1':
            raise ValueError(
                'You can only take the natural log of a variable if it has no unit')

        val = np.log(self.value)

        var = variable(val, '1')
        var._addDependent(self, 1 / self.value)
        var._calculateUncertanty()

        return var

    def log10(self):
        if self.unit != '1':
            raise ValueError(
                'You can only take the base 10 log of a variable if it has no unit')
        val = np.log10(self.value)

        var = variable(val, '1')
        var._addDependent(self, 1 / (self.value * np.log10(self.value)))
        var._calculateUncertanty()

        return var

    def exp(self):
        return np.e**self

    def sqrt(self):
        return self**(1 / 2)

    def sin(self):
        if self._unitObject.unitDictSI != {'rad': {'': 1}}:
            raise ValueError('You can only take sin of an angle')

        outputUnit = '1'
        if self.unit == 'rad':
            val = np.sin(self.value)
            grad = np.cos(self.value)
        else:
            val = np.sin(np.pi / 180 * self.value)
            grad = np.pi / 180 * np.cos(np.pi / 180 * self.value)

        var = variable(val, outputUnit)
        var._addDependent(self, grad)
        var._calculateUncertanty()

        return var

    def cos(self):
        if self._unitObject.unitDictSI != {'rad': {'': 1}}:
            raise ValueError('You can only take cos of an angle')

        outputUnit = '1'
        if self.unit == 'rad':
            val = np.cos(self.value)
            grad = -np.sin(self.value)
        else:
            val = np.cos(np.pi / 180 * self.value)
            grad = -np.pi / 180 * np.sin(np.pi / 180 * self.value)

        var = variable(val, outputUnit)
        var._addDependent(self, grad)
        var._calculateUncertanty()

        return var

    def tan(self):
        if self._unitObject.unitDictSI != {'rad': {'': 1}}:
            raise ValueError('You can only take tan of an angle')

        outputUnit = '1'
        if self.unit == 'rad':
            val = np.tan(self.value)
            grad = 2 / (np.cos(2 * self.value) + 1)
        else:
            val = np.tan(np.pi / 180 * self.value)
            grad = np.pi / (90 * (np.cos(np.pi / 90 * self.value) + 1))

        var = variable(val, outputUnit)
        var._addDependent(self, grad)
        var._calculateUncertanty()

        return var

    def __abs__(self):
        return variable(np.abs(self.value), self.unit, self.uncert)

    def __array_function__(self, func, types, args, kwargs):
        match func:
            case np.max:
                return self.max()
            case np.min:
                return self.min()
            case np.mean:
                return self.mean()
            case np.argmin:
                return self.argmin()
            case np.argmax:
                return self.argmax()
        raise NotImplementedError()

    def max(self):
        return self

    def min(self):
        return self

    def mean(self):
        return self

    def argmin(self):
        return 0

    def argmax(self):
        return 0

    @staticmethod
    def __comparer__(func):

        def wrap(*args):
            a = args[0]
            b = args[1]

            if not isinstance(b, scalarVariable):
                b = variable(b, a.unit)

            if a.unit == b.unit:
                return func(a, b)

            if not a._unitObject.unitDictSI == b._unitObject.unitDictSI:
                raise ValueError(
                    f'You cannot compare {a} and {b} as they do not have the same SI base unit')

            aUnit = a.unit
            bUnit = b.unit

            a.convert(a._unitObject.unitStrSI)
            b.convert(b._unitObject.unitStrSI)

            res = func(a, b)

            a.convert(aUnit)
            b.convert(bUnit)

            return res

        return wrap

    @__comparer__
    def __lt__(self, other):
        return self.value < other.value

    @__comparer__
    def __le__(self, other):
        return self.value <= other.value

    @__comparer__
    def __gt__(self, other):
        return self.value > other.value

    @__comparer__
    def __ge__(self, other):
        return self.value >= other.value

    @__comparer__
    def __eq__(self, other):
        if isinstance(self, arrayVariable) and isinstance(other, arrayVariable):
            if len(self) != len(other):
                raise ValueError(
                    f"operands could not be broadcast together with shapes {self.value.shape} {other.value.shape}")
        return self.value == other.value

    @__comparer__
    def __ne__(self, other):
        if isinstance(self, arrayVariable) and isinstance(other, arrayVariable):
            if len(self) != len(other):
                raise ValueError(
                    f"operands could not be broadcast together with shapes {self.value.shape} {other.value.shape}")
        return self.value != other.value

    def __hash__(self):
        return id(self)


class arrayVariable(scalarVariable):

    def __init__(self, value=None, unitStr=None, uncert=None, nDigits=None, scalarVariables=None) -> None:

        if not (value is None) and not (scalarVariables is None):
            raise ValueError(
                'You cannot supply both values and scalarVariables')

        if not value is None:
            self._nDigits = nDigits

            # create a unit object
            self._unitObject = unitStr if isinstance(
                unitStr, unit) else unit(unitStr)

            self.scalarVariables = []
            if not value is None:
                for v, u in zip(value, uncert):
                    self.scalarVariables.append(
                        scalarVariable(v, self._unitObject, u, nDigits))
        else:
            self.scalarVariables = scalarVariables
            self._unitObject = self.scalarVariables[0]._unitObject
            self.nDigits = self.scalarVariables[0].nDigits

    def __checkUnitOfScalarVariables(self):
        for var in self.scalarVariables:
            if var._unitObject != self._unitObject:
                raise ValueError(
                    f'Some of the scalarvariables in {self} did not have the unit [{self.unit}] as they should. This could happen if the user has converted a scalarVaraible instead of the arrayVaraible.')

    def __add__(self, other):
        self.__checkUnitOfScalarVariables()
        return scalarVariable.__add__(self, other)

    def __sub__(self, other):
        self.__checkUnitOfScalarVariables()
        return scalarVariable.__sub__(self, other)

    def __mul__(self, other):
        self.__checkUnitOfScalarVariables()
        return scalarVariable.__mul__(self, other)

    def __rmul__(self, other):
        self.__checkUnitOfScalarVariables()
        return scalarVariable.__rmul__(self, other)

    def __pow__(self, other):
        self.__checkUnitOfScalarVariables()
        return scalarVariable.__pow__(self, other)

    def __rpow__(self, other):
        self.__checkUnitOfScalarVariables()
        return scalarVariable.__rpow__(self, other)

    def __truediv__(self, other):
        self.__checkUnitOfScalarVariables()
        return scalarVariable.__truediv__(self, other)

    def __rtruediv__(self, other):
        self.__checkUnitOfScalarVariables()
        return scalarVariable.__rtruediv__(self, other)

    def __neg__(self):
        self.__checkUnitOfScalarVariables()
        return scalarVariable.__neg__(self)

    def _calculateUncertanty(self):
        for elem in self.scalarVariables:
            elem._calculateUncertanty()

    def addCovariance(self, var, grad: np.ndarray, unitStr: str):
        for elem, varElem, gradElem in zip(self.scalarVariables, var, grad):
            elem.addCovariance(varElem, gradElem, unitStr)

    def _addDependent(self, var, grad):
        isArrayVariable = isinstance(var, arrayVariable)
        isArrayGradient = isinstance(
            grad, list) or isinstance(grad, np.ndarray)

        if isArrayVariable:
            if not len(var) == len(self):
                var = var[0]
                isArrayVariable = False

        if isArrayGradient:
            if not len(grad) == len(self):
                grad = grad[0]
                isArrayGradient = False

        grad *= self._unitObject._converterToSI.scale / \
            var._unitObject._converterToSI.scale

        for i, elem in enumerate(self.scalarVariables):
            v = var[i] if isArrayVariable else var
            g = grad[i] if isArrayGradient else grad
            elem._scalarVariable__addDependent(v, g)

    def __len__(self):
        return len(self.scalarVariables)

    def __getitem__(self, index):

        if isinstance(index, int):
            return self.scalarVariables[index]
        elif isinstance(index, slice):
            return arrayVariable(scalarVariables=self.scalarVariables[index])
        if isinstance(index, list):
            return arrayVariable(scalarVariables=[self.scalarVariables[elem] for elem in index])
        else:
            raise NotImplementedError()

    @property
    def nDigits(self):
        return self._nDigits

    @nDigits.setter
    def nDigits(self, nDigits):
        self._nDigits = nDigits
        for elem in self.scalarVariables:
            elem.nDigits = nDigits

    @property
    def value(self):
        return np.array([elem.value for elem in self.scalarVariables])

    @property
    def uncert(self):
        return np.array([elem.uncert for elem in self.scalarVariables])

    def __setitem__(self, i, elem):
        if (type(elem) != scalarVariable):
            raise ValueError(
                f'You can only set an element with a scalar variable')
        if (elem.unit != self.unit):
            raise ValueError(
                f'You can not set an element of {self} with {elem} as they do not have the same unit')

        self.scalarVariables[i] = elem

    def append(self, elem):

        if (elem.unit != self.unit):
            raise ValueError(
                f'You can not set an element of {self} with {elem} as they do not have the same unit')

        if isinstance(elem, arrayVariable):
            elemsToAppend = [e for e in elem]
            for e in elemsToAppend:
                self.scalarVariables.append(e)
        else:
            self.scalarVariables.append(elem)

    def printUncertanty(self, value, uncert):
        # function to print number
        if uncert == 0 or uncert is None or np.isnan(uncert):
            return f'{value:.{self.nDigits}g}', 0

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

        # print array of values
        valStr = []
        uncStr = []
        for v, u in zip(self.value, self.uncert):
            v, u = self.printUncertanty(v, u)
            valStr.append(v)
            uncStr.append(u)

        if all(self.uncert == 0) or all(elem is None for elem in self.uncert):
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

    def __pow__(self, other):

        if not (isinstance(other, scalarVariable) or isinstance(other, arrayVariable)):
            return self ** variable(other)

        if isinstance(other, arrayVariable):
            if len(self) != len(other):
                raise ValueError(
                    f'operands could not be broadcast together with shapes {self.value.shape} {other.value.shape}')
            out = [a**b for a, b in zip(self, other)]
        else:
            out = [a**other for a in self]

        # if other is an arrayVariable, then this could imply that the unit of each element in out are not equal
        # if the units of the scalarVariables are equel, then collect them in an arrayVariable
        # else return the list of scalarVariables
        if all([out[0].unit == elem.unit for elem in out]):
            return arrayVariable(scalarVariables=out)

        return out

    def __rpow__(self, other):
        if not (isinstance(other, scalarVariable) or isinstance(other, arrayVariable)):
            return variable(other) ** self

        if isinstance(other, arrayVariable):
            if len(self) != len(other):
                raise ValueError(
                    f'operands could not be broadcast together with shapes {self.value.shape} {other.value.shape}')
            out = [a**b for a, b in zip(other, self)]
        else:
            out = [other**a for a in self]

        if all([out[0].unit == elem.unit for elem in out]):
            return variable([elem.value for elem in out], out[0].unit, [elem.uncert for elem in out])
        return out

    def __array_ufunc__(self, ufunc, *args, **kwargs):
        match ufunc:
            case np.log:
                return self.log()
            case np.log10:
                return self.log10()
            case np.exp:
                return self.exp()
            case np.sin:
                return self.sin()
            case np.cos:
                return self.cos()
            case np.tan:
                return self.tan()
            case np.sqrt:
                return self.sqrt()
        raise NotImplementedError()

    def min(self):
        index = np.argmin(self.value)
        return variable(self.value[index], self.unit, self.uncert[index])

    def max(self):
        index = np.argmax(self.value)
        return variable(self.value[index], self.unit, self.uncert[index])

    def mean(self):
        if not self._unitObject.isLogarithmicUnit():
            return sum(self / len(self))
        return logarithmicVariables.mean(self)

    def argmin(self):
        return np.argmin(self.value)

    def argmax(self):
        return np.argmax(self.value)

    def convert(self, newUnit):
        converter = self._unitObject.getConverter(newUnit)
        newUnit = unit(newUnit)
        for elem in self.scalarVariables:
            elem._value = converter.convert(
                elem._value, useOffset=not self._unitObject.isCombinationUnit())
            elem._uncert = converter.convert(elem._uncert, useOffset=False)
            elem._unitObject = newUnit
        self._unitObject = newUnit

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            out = self.scalarVariables[self.n]
            self.n += 1
            return out

        raise StopIteration

    def pop(self, index=-1):
        self.scalarVariables.pop(index)


def variable(value, unit='', uncert=None, nDigits=3):
    # store the value and the uncertaty
    def evaluateInput(input):
        if input is None:
            return input
        if isinstance(input, np.ndarray):
            return input
        else:
            if isinstance(input, list):
                return np.array(input, dtype=float)
            else:
                return float(input)

    value = evaluateInput(value)
    uncert = evaluateInput(uncert)

    if uncert is None:
        if isinstance(value, np.ndarray):
            uncert = np.zeros(len(value))
        else:
            uncert = 0
    else:
        if isinstance(value, np.ndarray):
            if not isinstance(uncert, np.ndarray):
                raise ValueError(
                    'The lenght of the value has to be equal to the lenght of the uncertanty')
            if len(uncert) != len(value):
                raise ValueError(
                    'The lenght of the value has to be equal to the lenght of the uncertanty')
        else:
            if isinstance(uncert, np.ndarray):
                raise ValueError(
                    'The lenght of the value has to be equal to the lenght of the uncertanty')

    if isinstance(value, np.ndarray):
        return arrayVariable(value=value, unitStr=unit, uncert=uncert, nDigits=nDigits)
    else:
        return scalarVariable(value, unit, uncert, nDigits)


if __name__ == "__main__":
    a = variable([1, 2], 'dB')
    b = np.mean(a)
    print(b.value)
