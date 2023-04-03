from copy import copy
import numpy as np
try:
    from unit import unit
    from unit import logrithmicUnits
except ImportError:
    from pyees.unit import unit
    from pyees.unit import logrithmicUnits
    
        
    
class logarithmicVariables:
    def __init__(self):
        pass
    
    @staticmethod
    def __add__(a,b):
        aUnit = str(a.unit)
        bUnit =str(b.unit)
        
        aUnitWithoutPrefix = a._unitObject.getUnitWithoutPrefix()
        a.convert(aUnitWithoutPrefix)

        aConverter = a._unitObject.getLogarithmicConverter()
        a._unitStr = '1'
        a._unitObject = unit('')
        aConverter.convertToSignal(a)
       
        bUnitWithoutPrefix = b._unitObject.getUnitWithoutPrefix()
        b.convert(bUnitWithoutPrefix)
        bConverter = b._unitObject.getLogarithmicConverter()
        b._unitStr = '1'
        b._unitObject = unit('')
        bConverter.convertToSignal(b)

        c = a + b
     
        aConverter.convertFromSignal(a)
        a._unitStr = aUnitWithoutPrefix
        a._unitObject = unit(aUnitWithoutPrefix)
        a.convert(aUnit)
        
        bConverter.convertFromSignal(b)
        b._unitStr = bUnitWithoutPrefix
        b._unitObject = unit(bUnitWithoutPrefix)
        b.convert(bUnit)
        
        
        aConverter.convertFromSignal(c)
        c._unitStr = aUnitWithoutPrefix
        c._unitObject = unit(aUnitWithoutPrefix)
        c.convert(aUnit)

        return c

    @staticmethod
    def __sub__(a,b):
        aUnit = str(a.unit)
        bUnit =str(b.unit)
        
        aUnitWithoutPrefix = a._unitObject.getUnitWithoutPrefix()
        a.convert(aUnitWithoutPrefix)

        aConverter = a._unitObject.getLogarithmicConverter()
        a._unitStr = '1'
        a._unitObject = unit('')
        aConverter.convertToSignal(a)
       
        bUnitWithoutPrefix = b._unitObject.getUnitWithoutPrefix()
        b.convert(bUnitWithoutPrefix)
        bConverter = b._unitObject.getLogarithmicConverter()
        b._unitStr = '1'
        b._unitObject = unit('')
        bConverter.convertToSignal(b)

        c = a - b
     
        aConverter.convertFromSignal(a)
        a._unitStr = aUnitWithoutPrefix
        a._unitObject = unit(aUnitWithoutPrefix)
        a.convert(aUnit)
        
        bConverter.convertFromSignal(b)
        b._unitStr = bUnitWithoutPrefix
        b._unitObject = unit(bUnitWithoutPrefix)
        b.convert(bUnit)
        
        
        aConverter.convertFromSignal(c)
        c._unitStr = aUnitWithoutPrefix
        c._unitObject = unit(aUnitWithoutPrefix)
        c.convert(aUnit)

        return c
   
class scalarVariable():
    def __init__(self, value, unitStr, uncert, nDigits) -> None:
        
        self._value = value
        self._uncert = uncert

        # create a unit object
        self._unitObject = copy(unitStr) if isinstance(unitStr, unit) else unit(unitStr)

        # number of digits to show
        self.nDigits = nDigits

        # value and unit in SI. This is used when determining the gradient in the uncertanty expression
        self._getConverterToSI()

        # uncertanty
        self.dependsOn = {}
        self.covariance = {}

    def _getConverterToSI(self):
        self._converterToSI = self._unitObject._converterToSI

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
        self._value = converter.convert(self._value, useOffset=not self._unitObject.isCombinationUnit())
        self._uncert = converter.convert(self._uncert, useOffset=False)
        self._unitObject.convert(newUnit)

        # update the converter to SI
        self._getConverterToSI()
 
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


        # print a single value
        value = self.value
        if self.uncert != 0:
            uncert = self.uncert

        value, uncert = self.printUncertanty(value, uncert)
        if uncert is None:
            return rf'{value}{space}{unitStr}'
        else:
                return rf'{value} {pm} {uncert}{space}{unitStr}'

    def _addDependent(self, var, grad):
        # scale the gradient to SI units. This is necessary if one of the variables are converted after the dependency has been noted
        grad *= self._converterToSI.convert(1, useOffset=False) / var._converterToSI.convert(1, useOffset=False)
        
        if var.dependsOn:
            # the variable depends on other variables
            # loop over the dependencies of the variables and add them to the dependencies of self.
            # this ensures that the product rule is used
            for vvar, dependency in var.dependsOn.items():
                if not vvar in self.dependsOn:
                    val = vvar._converterToSI.convert(vvar.value)
                    unc = vvar._converterToSI.convert(vvar.uncert, useOffset = False)
                    self.dependsOn[vvar] = [val ,unc, grad * dependency[2]]
                else:
                    self.dependsOn[vvar][2] += grad * dependency[2]
        else:    
            if not var in self.dependsOn:
                val = var._converterToSI.convert(var.value)
                unc = var._converterToSI.convert(var.uncert, useOffset = False)
                self.dependsOn[var] = [val, unc, grad]
            else:
                self.dependsOn[var][2] += grad
                 
   
    def addCovariance(self, var, covariance: float, unitStr: str):
        try:
            float(covariance)
        except TypeError:
            raise ValueError(f'You tried to set the covariance between {self} and {var} with a non scalar value')
        
        covUnit = unit(unitStr)
        selfVarUnit = unit(self._unitObject * var._unitObject)
        if not unit._assertEqualStatic(covUnit._SIBaseUnit, selfVarUnit._SIBaseUnit):
            raise ValueError(f'The covariance of {covariance} [{unitStr}] does not match the units of {self} and {var}')
        
        covariance = covUnit._converterToSI.convert(covariance, useOffset=False)
        
        self.covariance[var] = covariance        
        var.covariance[self] = covariance
        
    def _calculateUncertanty(self):
 
        # variance from each measurement
        variance = 0

        # loop over each different variable object in the dependency dict
        for dependency in self.dependsOn.values():
            
            ## loop over each "instance" of the variable
            ## an instance occurs when the variable has been changed using methods as __setitem__
            ## this affectively makes the variable a "new" variable in the eyes of other variables
    
            ## add the variance constribution to the variance
            unc, grad = dependency[1], dependency[2]
            variance += (unc * grad) ** 2
        
        ## scale the variance back in to the currenct unit
        ## the scaling is raised to a power of 2, as the above line should be
        ## variance += (scale * unc * grad) ** 2
        scale = 1 / self._converterToSI.convert(1, useOffset=False)
        variance *= scale ** 2
        
        # variance from the corralation between measurements
        ## covariance = list(vari, vari.currentValue, vari.currentUncert, varj, varj.currentValue, varj.currenctUncert, cov)
        ## from the above the gradients can be found from self.dependsOn
        
        for var1 in self.dependsOn.keys():
            for var2 in var1.covariance.keys():
                if var2 in self.dependsOn:
                    grad1 = self.dependsOn[var1][2]
                    grad2 = self.dependsOn[var2][2]
                    variance += scale**2 * grad1 * grad2 * var1.covariance[var2]

        self._uncert = np.sqrt(variance)
        
    def __add__(self, other):
        
        if not isinstance(other, scalarVariable):
            return self + variable(other, self.unit)

        # determine if the two variable can be added
        isLogarithmicUnit, outputUnit, scaleToSI, scaleSelf, scaleOther = self._unitObject + other._unitObject

        if isLogarithmicUnit:
            return logarithmicVariables.__add__(self, other)

        # convert self and other to the SI unit system
        selfUnit = copy(self.unit)
        otherUnit = copy(other.unit)
        
        if scaleToSI:
            self.convert(self._unitObject._SIBaseUnit)
            other.convert(other._unitObject._SIBaseUnit)
        else:
            if scaleSelf:
                a, _ = self._unitObject._removePrefixFromUnit(self.unit)
                self.convert(a)
            if scaleOther:
                a, _ = other._unitObject._removePrefixFromUnit(other.unit)
                other.convert(a)
        

        # determine the value and gradients
        val = self.value + other.value
        
        # create the new variable
        var = variable(val, outputUnit)
        var._addDependent(self, 1)
        var._addDependent(other, 1)
        var._calculateUncertanty()

        # convert all units back to their original units
        self.convert(selfUnit)
        other.convert(otherUnit)
        
        ## convert the variable in to the original unit if self and other has the same original unit
        ## otherwise keep the variable in the SI unit system
        if (selfUnit == otherUnit):
            var.convert(selfUnit)
        
        if outputUnit == 'K':
            SIBaseUnits = [self._unitObject._SIBaseUnit, other._unitObject._SIBaseUnit]
            if 'DELTAK' in SIBaseUnits and 'K' in SIBaseUnits:
                var.convert([selfUnit, otherUnit][SIBaseUnits.index('K')]) 
        return var

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if not isinstance(other, scalarVariable):
            return self - variable(other, self.unit)

        # determine if the variables can be subtracted
        isLogarithmicUnit, outputUnit, scaleToSI, scaleSelf, scaleOther = self._unitObject - other._unitObject

        if isLogarithmicUnit:
            return logarithmicVariables.__sub__(self, other)

        # convert self and other to the SI unit system
        selfUnit = copy(self.unit)
        otherUnit = copy(other.unit)
 
        if scaleToSI:
            self.convert(self._unitObject._SIBaseUnit)
            other.convert(other._unitObject._SIBaseUnit)
        else:
            if scaleSelf:
                a, _ = self._unitObject._removePrefixFromUnit(self.unit)
                self.convert(a)
            if scaleOther:
                a, _ = other._unitObject._removePrefixFromUnit(other.unit)
                other.convert(a)

        # determine the value and gradients
        val = self.value - other.value

        # create the new variable
        var = variable(val, outputUnit)
        var._addDependent(self, 1)
        var._addDependent(other, -1)
        var._calculateUncertanty()


        # convert self and other back
        self.convert(selfUnit)
        other.convert(otherUnit)

        if self.unit == other.unit and outputUnit == self.unit:
            var.convert(self.unit)
        if outputUnit == 'K':
            SIBaseUnits = [self._unitObject._SIBaseUnit, other._unitObject._SIBaseUnit]
            if 'DELTAK' in SIBaseUnits and 'K' in SIBaseUnits:
                var.convert([selfUnit, otherUnit][SIBaseUnits.index('K')]) 
        return var

    def __rsub__(self, other):
        return - self + other

    def __mul__(self, other):
    
        if not isinstance(other, scalarVariable):
            return self * variable(other)

        val = self.value * other.value
        outputUnit = self._unitObject * other._unitObject

        var = variable(val, outputUnit)
        var._addDependent(self, other.value)
        var._addDependent(other, self.value)
        var._calculateUncertanty()

        if var._unitObject._SIBaseUnit == '1' and var._unitObject != '1':
            var.convert('1')

        return var

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        if not isinstance(other, scalarVariable):
            return self ** variable(other, '')

        if str(other.unit) != '1':
            raise ValueError('The exponent can not have a unit')

        selfUnit = copy(self.unit)
        outputUnit, hasToBeScaledToSI = self._unitObject ** other.value

        if hasToBeScaledToSI:
            self.convert(self._unitObject._SIBaseUnit)
        
        val = self.value ** other.value

        def gradSelf(valSelf, valOTher):
            if valSelf != 0:
                return  valOTher * valSelf ** (valOTher - 1)
            return 0
        
        def gradOther(valSelf, valOther, uncertOther):
            if uncertOther != 0:
                return valSelf ** valOther * np.log(valSelf)
            return 0
        
        gradOther = np.vectorize(gradOther, otypes=[float])(self.value, other.value, other._uncert)
        gradSelf = np.vectorize(gradSelf, otypes=[float])(self.value, other.value)
        
        var = variable(val, outputUnit)
        var._addDependent(self, gradSelf)
        var._addDependent(other, gradOther)
        var._calculateUncertanty()
        
        if hasToBeScaledToSI:
            self.convert(selfUnit)
        
        return var

    def __rpow__(self, other):
        return variable(other, '1') ** self

    def __truediv__(self, other):
        if not isinstance(other, scalarVariable):
            return self / variable(other)

        val = self.value / other.value
        outputUnit = self._unitObject / other._unitObject

        var = variable(val, outputUnit)
        var._addDependent(self, 1 / other.value)
        var._addDependent(other, -self.value / (other.value**2))
        var._calculateUncertanty()

        if var._unitObject._SIBaseUnit == '1' and var._unitObject != '1':
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

        if var._unitObject._SIBaseUnit == '1' and var._unitObject != '1':
            var.convert('1')

        return var

    def __neg__(self):
        return -1 * self

    def log(self):
        if self.unit != '1':
            raise ValueError('You can only take the natural log of a variable if it has no unit')

        val = np.log(self.value)

        var = variable(val, '1')
        var._addDependent(self, 1 / self.value)
        var._calculateUncertanty()

        return var

    def log10(self):
        if self.unit != '1':
            raise ValueError('You can only take the base 10 log of a variable if it has no unit')
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
        if str(self._unitObject._SIBaseUnit) != 'rad':
            raise ValueError('You can only take sin of an angle')
        
        outputUnit = '1'
        if self._unitObject._assertEqual('rad'):
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
        if str(self._unitObject._SIBaseUnit) != 'rad':
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
        if str(self._unitObject._SIBaseUnit) != 'rad':
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
        raise NotImplementedError()
    
    def max(self):
        return self
    
    def min(self):
        return self
    
    def mean(self):
        return self

    def __lt__(self, other):
        if not isinstance(other, scalarVariable):
            return self < variable(other, self.unit)

        if not self._unitObject._SIBaseUnit == other._unitObject._SIBaseUnit:
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same SI base unit')
        
        selfUnit = copy(self.unit)
        otherUnit = copy(other.unit)
        
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(self._unitObject._SIBaseUnit)

        out = self.value < other.value
        
        self.convert(selfUnit)
        other.convert(otherUnit)
        return out

    def __le__(self, other):
        if not isinstance(other, scalarVariable):
            return self <= variable(other, self.unit)
        if not self._unitObject._SIBaseUnit == other._unitObject._SIBaseUnit:
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same SI base unit')
        
        selfUnit = copy(self.unit)
        otherUnit = copy(other.unit)
        
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(self._unitObject._SIBaseUnit)
        
        out = self.value <= other.value
        
        self.convert(selfUnit)
        other.convert(otherUnit)
        return out
    
    def __gt__(self, other):
        if not isinstance(other, scalarVariable):
            return self > variable(other, self.unit)

        if not self._unitObject._SIBaseUnit == other._unitObject._SIBaseUnit:
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same SI base unit')
        
        selfUnit = copy(self.unit)
        otherUnit = copy(other.unit)
        
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(self._unitObject._SIBaseUnit)

        out = self.value > other.value
           
        self.convert(selfUnit)
        other.convert(otherUnit)
        return out
   
    def __ge__(self, other):
        if not isinstance(other, scalarVariable):
            return self >= variable(other, self.unit)

        if not self._unitObject._SIBaseUnit == other._unitObject._SIBaseUnit:
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same SI base unit')
        
        selfUnit = copy(self.unit)
        otherUnit = copy(other.unit)
        
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(self._unitObject._SIBaseUnit)

        out = self.value >= other.value
        
        self.convert(selfUnit)
        other.convert(otherUnit)
        return out

    def __eq__(self, other):
        if not isinstance(other, scalarVariable):
            return self == variable(other, self.unit)

        if not self._unitObject._SIBaseUnit == other._unitObject._SIBaseUnit:
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same SI base unit')
        
        selfUnit = copy(self.unit)
        otherUnit = copy(other.unit)
        
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(self._unitObject._SIBaseUnit)
        
        if isinstance(self, arrayVariable) and isinstance(other, arrayVariable):
            if len(self) != len(other):
                raise ValueError(f"operands could not be broadcast together with shapes {self.value.shape} {other.value.shape}")    
        out = self.value == other.value
            
        self.convert(selfUnit)
        other.convert(otherUnit)
        return out

    def __ne__(self, other):
        if not isinstance(other, scalarVariable):
            return self != variable(other, self.unit)

        if not self._unitObject._SIBaseUnit == other._unitObject._SIBaseUnit:
            raise ValueError(f'You cannot compare {self} and {other} as they do not have the same SI base unit')
        
        selfUnit = copy(self.unit)
        otherUnit = copy(other.unit)
        
        self.convert(self._unitObject._SIBaseUnit)
        other.convert(self._unitObject._SIBaseUnit)

        if isinstance(self, arrayVariable) and isinstance(other, arrayVariable):
            if len(self) != len(other):
                raise ValueError(f"operands could not be broadcast together with shapes {self.value.shape} {other.value.shape}")    
        out = self.value != other.value
            
        self.convert(selfUnit)
        other.convert(otherUnit)
        return out

    def __hash__(self):
        return id(self)



## TODO arrayVariables are quite slow. This is due to the units being evaluated for all scalarelements in the arrayVariable

class arrayVariable(scalarVariable):
    
    def __init__(self, value, unitStr, uncert, nDigits) -> None:
        self.nDigits = nDigits
        
        # create a unit object
        self._unitObject = unitStr if isinstance(unitStr, unit) else unit(unitStr)

        # value and unit in SI. This is used when determining the gradient in the uncertanty expression
        self._getConverterToSI()
        
        
        self.scalarVariables = []
        if not value is None:
            for v, u in zip(value, uncert):
                self.scalarVariables.append(scalarVariable(v, self._unitObject, u, nDigits))
    
    def _calculateUncertanty(self):
        for elem in self.scalarVariables:
            elem._calculateUncertanty()
    
    def addCovariance(self, var, grad: np.ndarray, unitStr: str):
        for elem, varElem, gradElem in zip(self.scalarVariables, var, grad):
            elem.addCovariance(varElem, gradElem, unitStr)
    
    def _addDependent(self, var, grad):
        isArrayVariable = isinstance(var, arrayVariable)
        isArrayGradient = isinstance(grad, list) or isinstance(grad, np.ndarray)
        
        for i, elem in enumerate(self.scalarVariables):
            v = var[i] if isArrayVariable else var
            g = grad[i] if isArrayGradient else grad
            elem._addDependent(v, g)
                        
    def __len__(self):
        return len(self.scalarVariables)

    def __getitem__(self, index):
        
        if isinstance(index, int):
            return self.scalarVariables[index]
        elif isinstance(index, slice):
            vals = [elem.value for elem in self.scalarVariables[index]]
            unc = [elem.uncert for elem in self.scalarVariables[index]]
            return variable(vals, self._unitObject, unc, self.nDigits)
        else:
            raise NotImplementedError()
 
    @property
    def value(self):
        return np.array([elem.value for elem in self.scalarVariables])       
 
    @property
    def uncert(self):
        return np.array([elem.uncert for elem in self.scalarVariables])       
 
    def __setitem__(self, i, elem):
        if (type(elem) != scalarVariable):
            raise ValueError(f'You can only set an element with a scalar variable')
        if (elem.unit != self.unit):
            raise ValueError(f'You can not set an element of {self} with {elem} as they do not have the same unit')
        
        self.scalarVariables[i] = elem
   
    def append(self, elem):
        
        if (elem.unit != self.unit):
            raise ValueError(f'You can not set an element of {self} with {elem} as they do not have the same unit')

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
                raise ValueError(f'operands could not be broadcast together with shapes {self.value.shape} {other.value.shape}')
            out = [a**b for a,b in zip(self, other)]
        else:
            out = [a**other for a in self]

        if all([out[0].unit == elem.unit for elem in out]):
            oout = arrayVariable(None, out[0].unit, None, out[0].nDigits)
            for o in out:
                oout.append(o)
            return oout
        
        return out
     
    def __rpow__(self,other):
        if not (isinstance(other, scalarVariable) or isinstance(other, arrayVariable)):
            return variable(other) ** self
    
        if isinstance(other, arrayVariable):
            if len(self) != len(other):
                raise ValueError(f'operands could not be broadcast together with shapes {self.value.shape} {other.value.shape}')
            out = [a**b for a,b in zip(other, self)]
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
        return sum(self) / len(self)
    
    def convert(self, newUnit):
        self._unitObject.convert(newUnit)
        for elem in self.scalarVariables:
            elem.convert(newUnit)
        



def variable(value, unit = '', uncert = None, nDigits = 3):
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
        if isinstance(value,np.ndarray):
            uncert = np.zeros(len(value))
        else:
            uncert = 0
    else:
        if isinstance(value,np.ndarray):
            if not isinstance(uncert, np.ndarray):
                raise ValueError('The lenght of the value has to be equal to the lenght of the uncertanty')
            if len(uncert) != len(value):
                raise ValueError('The lenght of the value has to be equal to the lenght of the uncertanty')
        else:
            if isinstance(uncert, np.ndarray):
                raise ValueError('The lenght of the value has to be equal to the lenght of the uncertanty')      
    
    if isinstance(value, np.ndarray):
        return arrayVariable(value, unit, uncert, nDigits)
    else:
        return scalarVariable(value, unit, uncert, nDigits)



if __name__ == "__main__":
            
    a = variable([1])
    b = variable([2])
    
    c = a + b
    
    d = c**2 
    print(d)
    for dd in d:
        for key, item in dd.dependsOn.items():
            print(key, item) 
