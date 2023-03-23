from copy import deepcopy
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
        self._unitObject = unitStr if isinstance(unitStr, unit) else unit(unitStr)

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

    def _addDependents(self, vars, grads):
        grads = [deepcopy(elem) for elem in grads]
        selfScaleToSI = self._converterToSI.convert(1, useOffset=False)

        # loop over the variables and their gradients
        for var, grad in zip(vars, grads):
            
            # scale the gradient to SI units. This is necessary if one of the variables are converted after the dependency has been noted
            grad *= selfScaleToSI / var._converterToSI.convert(1, useOffset=False)
            
            if var.dependsOn:
                # the variable depends on other variables
                # loop over the dependencies of the variables and add them to the dependencies of self.
                # this ensures that the product rule is used
                for vvar, dependencies in var.dependsOn.items():
                    for dependency in dependencies.values():
                        self.__addDependency(vvar, dependency[0], dependency[1], dependency[2] * grad)
            else:    
                val = var._converterToSI.convert(var.value)
                unc = var._converterToSI.convert(var.uncert, useOffset = False)
                self.__addDependency(var, val, unc, grad)
                 
    def __addDependency(self, var, val, unc, grad):

        if not var in self.dependsOn:
            self.dependsOn[var] = {0 : [val, unc, grad]}
            return      
        
        ## self is already dependent on the variable
        indexes = [i  for i, elem in enumerate(self.dependsOn[var].values()) if np.array_equal(elem[0], val) and np.array_equal(elem[1], unc) ]
        if len(indexes) > 1:
            raise ValueError(f'Any error occured when adding {var} to the dependencies of {self}')
        if indexes:
            self.dependsOn[var][indexes[0]][2] += grad
        else:
            n = len(list(self.dependsOn[var].keys()))
            self.dependsOn[var][n] = [val, unc, grad]
    
    def addCovariance(self, var, covariance: float, unitStr: str = None):
        try:
            float(covariance)
        except TypeError:
            raise ValueError(f'You tried to set the covariance between {self} and {var} with a non scalar value')
        
        if not unitStr is None:
            covUnit = unit(unitStr)
            selfVarUnit = unit(self._unitObject * var._unitObject)
            if not unit._assertEqualStatic(covUnit._SIBaseUnit, selfVarUnit._SIBaseUnit):
                raise ValueError(f'The covariance of {covariance} [{unitStr}] does not match the units of {self} and {var}')
            
            covariance = covUnit._converterToSI.convert(covariance, useOffset=False)
        else:
            covariance = self._converterToSI.convert(covariance, useOffset=False)
            covariance = var._converterToSI.convert(covariance, useOffset=False)
            
        selfVal = self._converterToSI.convert(self.value, useOffset = False)
        selfUnc = self._converterToSI.convert(self.uncert, useOffset = False)
        otherVal = var._converterToSI.convert(var.value, useOffset = False)
        otherUnc = var._converterToSI.convert(var.uncert, useOffset = False)
        
        if not var in self.covariance:
            self.covariance[var] = [[self, selfVal, selfUnc, var, otherVal, otherUnc, covariance]]
        else:
            changed = False
            for i, listOfCovariance in enumerate(self.covariance[var]):
                if not id(listOfCovariance[0]) == id(self): continue
                if not id(listOfCovariance[3]) == id(var): continue
                if not np.array_equal(listOfCovariance[1], selfVal): continue
                if not np.array_equal(listOfCovariance[2], selfUnc): continue
                if not np.array_equal(listOfCovariance[4], otherVal): continue
                if not np.array_equal(listOfCovariance[5], otherUnc): continue
                
                self.covariance[var][i][6] = covariance
                changed = True
                break
             
            if not changed:
                self.covariance[var] = [[self, selfVal, selfUnc, var, otherVal, otherUnc, covariance]]
                
        if not self in var.covariance:
            var.covariance[self] = [[var, otherVal, otherUnc, self, selfVal, selfUnc, covariance]]
        else:
            changed = False
            for i, listOfCovariance in enumerate(var.covariance[self]):
                if not id(listOfCovariance[0]) == id(var): continue
                if not id(listOfCovariance[3]) == id(self): continue
                if not np.array_equal(listOfCovariance[1], otherVal): continue
                if not np.array_equal(listOfCovariance[2], otherUnc): continue
                if not np.array_equal(listOfCovariance[4], selfVal): continue
                if not np.array_equal(listOfCovariance[5], selfUnc): continue
                
                var.covariance[self][i][6] = covariance
                changed = True
                break
            if not changed:
                var.covariance[self] = [[var, otherVal, otherUnc, self, selfVal, selfUnc, covariance]] 

    def _calculateUncertanty(self):
 
        # variance from each measurement
        variance = 0

        # loop over each different variable object in the dependency dict
        for variableDependency in self.dependsOn.values():
            
            ## loop over each "instance" of the variable
            ## an instance occurs when the variable has been changed using methods as __setitem__
            ## this affectively makes the variable a "new" variable in the eyes of other variables
            for dependency in variableDependency.values():
                
                ## add the variance constribution to the variance
                unc, grad1 = dependency[1], dependency[2]
                variance += (unc * grad1) ** 2
        
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
                    for covariance in var1.covariance[var2]:
                        var1Val, var1Unc = covariance[1], covariance[2]
                        var2Val, var2Unc = covariance[4], covariance[5]
                        
                        found1 = False
                        for elem in self.dependsOn[var1].values():
                            if np.array_equal(elem[0], var1Val) and np.array_equal(elem[1], var1Unc):
                                grad1 = elem[2]
                                found1 = True
                                break
                        if not found1:
                            break
                        
                        found2 = False
                        for elem in self.dependsOn[var2].values():
                            if np.array_equal(elem[0], var2Val) and np.array_equal(elem[1], var2Unc):
                                grad2 = elem[2]
                                found2 = True
                                break
                        if not found2:
                            break

                    
                        variance += scale**2 * grad1 * grad2 * covariance[6]

        self._uncert = np.sqrt(variance)
        
    def __add__(self, other):
        
        if not isinstance(other, scalarVariable):
            return self + variable(other, self.unit)

        # determine if the two variable can be added
        isLogarithmicUnit, outputUnit, scaleToSI, scaleSelf, scaleOther = self._unitObject + other._unitObject

        if isLogarithmicUnit:
            return logarithmicVariables.__add__(self, other)

        # convert self and other to the SI unit system
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
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
        val = self._value + other._value
        grad = [1, 1]
        vars = [self, other]
        
        # create the new variable
        var = variable(val, outputUnit)
        var._addDependents(vars, grad)
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
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
 
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
        grad = [1, -1]
        vars = [self, other]
        # create the new variable
        var = variable(val, outputUnit)
        var._addDependents(vars, grad)
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
        if not isinstance(other, scalarVariable):
            return self ** variable(other, '')

        if str(other.unit) != '1':
            raise ValueError('The exponent can not have a unit')

        selfUnit = deepcopy(self.unit)
        outputUnit, hasToBeScaledToSI = self._unitObject ** other.value

        if hasToBeScaledToSI:
            self.convert(self._unitObject._SIBaseUnit)
        
        val = self._value ** other.value

        def gradSelf(valSelf, valOTher):
            if valSelf != 0:
                return  valOTher * valSelf ** (valOTher - 1)
            return 0
        
        def gradOther(valSelf, valOther, uncertOther):
            if uncertOther != 0:
                return valSelf ** valOther * np.log(valSelf)
            return 0
        
        gradOther = np.vectorize(gradOther, otypes=[float])(self._value, other._value, other._uncert)
        gradSelf = np.vectorize(gradSelf, otypes=[float])(self._value, other._value)
        grad = [gradSelf, gradOther]
        vars = [self, other]
                
        var = variable(val, outputUnit)
        var._addDependents(vars, grad)
        var._calculateUncertanty()
        
        if hasToBeScaledToSI:
            self.convert(selfUnit)
        
        return var

    def __rpow__(self, other):
        return variable(other, '1') ** self

    def __truediv__(self, other):
        if not isinstance(other, scalarVariable):
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
        if not isinstance(other, scalarVariable):
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
        
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
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
        
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
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
        
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
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
        
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
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
        
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
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
        
        selfUnit = deepcopy(self.unit)
        otherUnit = deepcopy(other.unit)
        
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

class arrayVariable(scalarVariable):
        
    def __len__(self):
        return len(self._value)

    def __getitem__(self, index):
        if isinstance(index, int) or isinstance(index, slice):
            if index >= len(self):
                raise IndexError('Index out of bounds')
            if len(self) == 1:
                if index != 0:
                    raise IndexError('Index out of bound')
                return variable(self.value, self._unitObject, self.uncert)
            return variable(self.value[index], self._unitObject, self.uncert[index])
        else:
            val = [self.value[elem]for elem in index]
            unc = [self.uncert[elem]for elem in index]
            return variable(val, self._unitObject, unc)
 
    def __setitem__(self, i, elem):
        if (type(elem) != scalarVariable):
            raise ValueError(f'You can only set an element with a scalar variable')
        if (elem.unit != self.unit):
            raise ValueError(f'You can not set an element of {self} with {elem} as they do not have the same unit')
        self._value[i] = elem.value
        self._uncert[i] = elem.uncert
   
    def append(self, elem):
        if (elem.unit != self.unit):
            raise ValueError(f'You can not set an element of {self} with {elem} as they do not have the same unit')
        
        elemValue = elem._converterToSI.convert(elem.value, useOffset = False)
        elemUncert = elem._converterToSI.convert(elem.uncert, useOffset = False)
        selfValue = self._converterToSI.convert(self.value, useOffset = False)
        selfUncert = self._converterToSI.convert(self.uncert, useOffset = False)
            
        ## update the value and uncertanty of self
        self._value = np.append(self._value, elem.value)
        self._uncert = np.append(self._uncert, elem.uncert)
        
        covariancesToAdd = []
        for key in elem.covariance.keys():      
            if key in self.covariance.keys():
                # both self and elem has covariance with key
                # the covariances has to be merged
                for i in range(len(self.covariance[key])):
                    selfCov = self.covariance[key][i][6]
                    for j in range(len(elem.covariance[key])):
                        elemCov = elem.covariance[key][j][6]    
                        if len(selfCov) != len(elemCov): continue
                        cov = np.array(selfCov) + np.array(elemCov)
                        covariancesToAdd.append([self, key, cov])
        
        variablesWichNeedNewCovariances = []
        for key in elem.covariance.keys():   
            if not key in self.covariance.keys():
                variablesWichNeedNewCovariances.append(key)
        
        for var in variablesWichNeedNewCovariances:
            for item in elem.covariance[var]:
                if np.array_equal(item[1], elemValue) and np.array_equal(item[2], elemUncert):
                    cov = item[6]
                    break
            if len(cov) != len(self):
                cov = np.append(0 * elemValue, cov)
            self._addCovariance(var, cov)    
            
        variablesWichNeedNewCovariances = []
        for key in self.covariance.keys():
            if not key in elem.covariance:
                variablesWichNeedNewCovariances.append(key)

        for var in variablesWichNeedNewCovariances:
            for item in self.covariance[var]:
                if np.array_equal(item[1], selfValue) and np.array_equal(item[2], selfUncert):
                    cov = item[6]
                    break
            if len(cov) != len(self):
                cov = np.append(cov, 0 * selfValue)

            self._addCovariance(var, cov)        
       
        for cov in covariancesToAdd:
            var1, var2, cov = cov[0], cov[1], cov[2]           
            var1._addCovariance(var2, cov)
       
        
            
       
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
            return variable([elem.value for elem in out], out[0].unit, [elem.uncert for elem in out])
        
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

    def addCovariance(self, var, covariance, unitStr : str = None):
        if len(var) != len(self) or len(covariance) != len(self):
            raise ValueError(f'The lengths of {self}, {var}, and {covariance} are not equal')

        self._addCovariance(var, covariance, unitStr)
    
    def _addCovariance(self, var, covariance, unitStr : str = None):
        if isinstance(covariance, list):
            covariance = np.array(covariance)
        
        if not unitStr is None:
            covUnit = unit(unitStr)
            selfVarUnit = unit(self._unitObject * var._unitObject)
            if not unit._assertEqualStatic(covUnit._SIBaseUnit, selfVarUnit._SIBaseUnit):
                raise ValueError(f'The covariance of {covariance} [{unitStr}] does not match the units of {self} and {var}')    
            covariance = covUnit._converterToSI.convert(covariance, useOffset=False)
        else:
            covariance = self._converterToSI.convert(covariance, useOffset=False)
            covariance = var._converterToSI.convert(covariance, useOffset=False)
            
        selfVal = self._converterToSI.convert(self.value, useOffset = False)
        selfUnc = self._converterToSI.convert(self.uncert, useOffset = False)
        otherVal = var._converterToSI.convert(var.value, useOffset = False)
        otherUnc = var._converterToSI.convert(var.uncert, useOffset = False)
        
        if not var in self.covariance:
            self.covariance[var] = [[self, selfVal, selfUnc, var, otherVal, otherUnc, covariance]]
        else:
            changed = False
            for i, listOfCovariance in enumerate(self.covariance[var]):
                if not id(listOfCovariance[0]) == id(self): continue
                if not id(listOfCovariance[3]) == id(var): continue
                if not np.array_equal(listOfCovariance[1], selfVal): continue
                if not np.array_equal(listOfCovariance[2], selfUnc): continue
                if not np.array_equal(listOfCovariance[4], otherVal): continue
                if not np.array_equal(listOfCovariance[5], otherUnc): continue
                
                self.covariance[var][i][6] = covariance
                changed = True
                break
             
            if not changed:
                self.covariance[var] = [[self, selfVal, selfUnc, var, otherVal, otherUnc, covariance]]
                
        if not self in var.covariance:
            var.covariance[self] = [[var, otherVal, otherUnc, self, selfVal, selfUnc, covariance]]
        else:
            changed = False
            for i, listOfCovariance in enumerate(var.covariance[self]):
                if not id(listOfCovariance[0]) == id(var): continue
                if not id(listOfCovariance[3]) == id(self): continue
                if not np.array_equal(listOfCovariance[1], otherVal): continue
                if not np.array_equal(listOfCovariance[2], otherUnc): continue
                if not np.array_equal(listOfCovariance[4], selfVal): continue
                if not np.array_equal(listOfCovariance[5], selfUnc): continue
                
                var.covariance[self][i][6] = covariance
                changed = True
                break
            if not changed:
                var.covariance[self] = [[var, otherVal, otherUnc, self, selfVal, selfUnc, covariance]]
            




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

