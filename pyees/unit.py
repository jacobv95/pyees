import numpy as np
from fractions import Fraction
from copy import copy

class _unitConversion():

    def __init__(self, scale, offset=0) -> None:
        self.scale = scale
        self.offset = offset

    def __mul__(self, other):
        if isinstance(other, _unitConversion):
            scale = self.scale * other.scale
            offset = self.scale * other.offset + self.offset
        else:
            scale = self.scale * other
            offset = self.offset
        return _unitConversion(scale, offset)

    def __pow__(self, other):
        scale = 1
        offset = 0
        
        for _ in range(other):
            scale *= self.scale
            offset*= self.scale
            offset += self.offset
            
        return _unitConversion(scale, offset)

    def __truediv__(self, other):
        if isinstance(other, _unitConversion):
            return _unitConversion(1 / other.scale, - other.offset / other.scale) * self
        return self * _unitConversion(1 / other)

    def convert(self, value, useOffset=True):
        if useOffset:
            return self.scale * value + self.offset
        else:
            return self.scale * value

class neperConversion():
    def __init__(self):
        self.scale = 1
        self.offset = 0
    @staticmethod
    def convertToSignal(var):
        var._uncert = 2*np.exp(2*var.value) * var.uncert
        var._value = np.exp(2*var.value)
    @staticmethod
    def convertFromSignal(var):
        var._uncert = 1 / (2*var.value) * var.uncert
        var._value = 1/2 * np.log(var.value)
    def __mul__(self, other):
        return 1 * other
    def __rmul__(self, other):
        return self * other
    def __truediv__(self, other):
        return 1 / other
    def __rtruediv__(self, other):
        return other / 1
    def __pow__ (self, other):
        return 1** other
    def __rpow__(self, other):
        return other ** 1

class bellConversion():
    def __init__(self):
        self.scale = 1
        self.offset = 0
    @staticmethod
    def convertToSignal(var):
        var._uncert = 10**var.value * np.log(10) * var.uncert
        var._value = 10**var.value
    @staticmethod
    def convertFromSignal(var):
        var._uncert = 1 / (var.value * np.log(10)) * var.uncert
        var._value = np.log10(var.value)
    def __mul__(self, other):
        return 1 * other
    def __rmul__(self, other):
        return self * other
    def __truediv__(self, other):
        return 1 / other
    def __rtruediv__(self, other):
        return other / 1
    def __pow__ (self, other):
        return 1** other
    def __rpow__(self, other):
        return other ** 1

class octaveConversion():
    def __init__(self):
        self.scale = 1
        self.offset = 0
    @staticmethod
    def convertToSignal(var):
        var._uncert = 2**var.value * np.log(2) * var.uncert
        var._value = 2**var.value
    @staticmethod
    def convertFromSignal(var):
        var._uncert = 1 / (var.value * np.log(2)) * var.uncert
        var._value = np.log2(var.value)
    def __mul__(self, other):
        return 1 * other
    def __rmul__(self, other):
        return self * other
    def __truediv__(self, other):
        return 1 / other
    def __rtruediv__(self, other):
        return other / 1
    def __pow__ (self, other):
        return 1** other
    def __rpow__(self, other):
        return other ** 1

    
baseUnit = {
    '1': _unitConversion(1),
    '': _unitConversion(1),
    '%': _unitConversion(1e-2)
}

force = {
    'N': _unitConversion(1)
}

mass = {
    'g': _unitConversion(1 / 1000)
}

energy = {
    'J': _unitConversion(1),
}

power = {
    'W': _unitConversion(1)
}

pressure = {
    'Pa': _unitConversion(1),
    'bar': _unitConversion(1e5)
}

temperature = {
    'K': _unitConversion(1),
    'C': _unitConversion(1, 273.15),
    'F': _unitConversion(5 / 9, 273.15 - 32 * 5 / 9)
}

temperatureDifference = {
    'DELTAK': _unitConversion(1),
    'DELTAC': _unitConversion(1),
    'DELTAF': _unitConversion(5 / 9)
}

time = {
    's': _unitConversion(1),
    'min': _unitConversion(60),
    'h': _unitConversion(60 * 60),
    'yr': _unitConversion(60 * 60 * 24 * 365)
}

volume = {
    'm3': _unitConversion(1),
    'L': _unitConversion(1 / 1000)
}

length = {
    'm': _unitConversion(1),
    'Å': _unitConversion(1e-10),
    'ly': _unitConversion(9460730472580800)
}

angle = {
    'rad': _unitConversion(1),
    'deg': _unitConversion(np.pi / 180)
}

current = {
    'A': _unitConversion(1)
}

voltage = {
    'V': _unitConversion(1)
}

frequency = {
    'Hz': _unitConversion(1)
}

resistance = {
    'ohm': _unitConversion(1)
}

kinematicViscosity = {
    'St': _unitConversion(1e-4)
}

logrithmicUnits = {
    'Np' : neperConversion(),
    'B': bellConversion(),
    'oct': octaveConversion(),
    'dec': bellConversion()
}

knownUnitsDict = {
    'kg-m/s2': force,
    'kg/m-s2': pressure,
    's': time,
    'K': temperature,
    'm3': volume,
    'm': length,
    'kg-m2/s2': energy,
    'kg-m2/s3': power,
    'kg': mass,
    'A': current,
    'kg-m2/s3-A': voltage,
    '1': baseUnit,
    '1/s': frequency,
    'rad': angle,
    'kg-m2/s3-A2' : resistance,
    'm2/s' : kinematicViscosity,
    'Np': logrithmicUnits,
    'DELTAK': temperatureDifference
}

knownPrefixes = {
    'T':10**12,
    'G':10**9,
    'M':10**6,
    'k':10**3,
    'h':10**2,
    'da':10**1,
    'd':10**-1,
    'c':10**-2,
    'm':10**-3,
    'mu':10**-6,
    'n':10**-9,
    'p':10**-12
}


knownUnits = {}
for key, d in knownUnitsDict.items():
    for item, _ in d.items():
        if item not in knownUnits:
            knownUnits[item] = [key, knownUnitsDict[key][item]]
        else:
            raise Warning(f'The unit {item} known in more than one unit system')



# determine the known characters within the unit system
knownCharacters = list(knownUnits.keys()) + list(knownPrefixes.keys())
knownCharacters = ''.join(knownCharacters)
knownCharacters += '-/ '
knownCharacters += '0123456789'
knownCharacters += '()'
knownCharacters = set(knownCharacters)

# check if all unit and prefix combinations can be distiguished
unitPrefixCombinations = []
for u in knownUnits:
    unitPrefixCombinations += [u]
    if u not in baseUnit or u == "%":
        unitPrefixCombinations += [p + u for p in knownPrefixes]

for elem in unitPrefixCombinations:
    count = sum([1 if u == elem else 0 for u in unitPrefixCombinations])
    if count > 1:
        prefix = elem[0:1]
        unit = elem[1:]

        unitType1 = ''
        for key, item in knownUnitsDict.items():
            if elem in item:
                unitType1 = [key for key, a in locals().items() if a == item][0]
        if unitType1 == '':
            raise ValueError(f'The unit {elem} was not found.')

        unitType2 = ''
        for key, item in knownUnitsDict.items():
            if unit in item:
                unitType2 = [key for key, a in locals().items() if a == item][0]
        if unitType2 == '':
            raise ValueError(f'The unit {unit} was not found.')

        raise ValueError(f'The unit {elem} can be interpreted as a {unitType1} or a {unitType2} with the prefix {prefix}. The cannot be distiguished.')


def addNewUnit(newUnit: str, scale: float, existingUnit: str, offset : float = 0):
        
    if newUnit in unitPrefixCombinations:
        raise ValueError(f'The unit {newUnit} is already known within the unit system')
    unitPrefixCombinations.append(newUnit)
    for p in knownPrefixes:
        if p+newUnit in unitPrefixCombinations:
            raise ValueError(f'The unit {p+newUnit} is already known within the unit system')
        unitPrefixCombinations.append(newUnit)

    ## create the conversion 
    conversion =  _unitConversion(scale, offset)


    existingUnitDict = unit._getUnitDict(existingUnit)
    existingUnitDictSI = unit._getUnitDictSI(existingUnitDict)
    SIBaseUnit = unit._getUnitStrFromDict(existingUnitDictSI)
        
    
    for key, item in existingUnitDict.items():
        for pre, exp in item.items():    
                    
            unitConversion = knownUnits[key][1]
            
            isUpper = exp > 0
            
            if not isUpper: exp *= -1
            
            unitConversion =  unitConversion ** exp

            if not pre is None: unitConversion *= knownPrefixes[pre]

            if isUpper:
                conversion = unitConversion * conversion
            else:
                conversion = _unitConversion(1) / unitConversion * conversion
    
    knownUnits[newUnit] = [SIBaseUnit, conversion]
        
    global knownCharacters
    for s in newUnit:
        knownCharacters.add(s)

    
hyphen = '-'
slash = '/'
integers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class unit():

    def __init__(self, unitStr = None, unitDict = None):
        if unitStr is None and unitDict is None:
            unitStr = ''
        
        if not unitStr is None:
            self.unitDict = self._getUnitDict(self._formatUnitStr(unitStr))
        else:
            self.unitDict = unitDict
            
        self.unitStr = self._getUnitStrFromDict(self.unitDict)
        self.unitDictSI = self._getUnitDictSI(self.unitDict)
        self.unitStrSI = self._getUnitStrFromDict(self.unitDictSI)
        
        # ## create a version of the self.unitDictSI that is formatted in the same way as a unitDict
        # unitDictSI = {}
        # for key, exp in self.unitDictSI.items():
        #     u = knownUnits[key][0]
        #     u,p,e = unit._splitUnitExponentAndPrefix(u)
        #     unitDictSI[u] = {p: e*exp}
        self._converterToSI = self.getConverterFromDict(self.unitDictSI)


    @staticmethod
    def _getUnitStrFromDict(unitDict):
        upper, lower = [], []
        for u, item in unitDict.items():    
            for p, exp in item.items():
                if p is None: p = ''

                up = exp > 0
                
                if not up:
                    exp *= -1
                
                if exp == 1:
                    exp = ''
                
                s = f'{p}{u}{exp}'
                if up:
                    upper.append(s)
                else:
                    lower.append(s)
                                
        upper = '-'.join(upper)
        
        if lower:
            upper += '/' + '-'.join(lower)
                
        return upper
      
    def getUnitWithoutPrefix(self):
        
        upper, lower = [],[]
        
        for key, item in self.unitDict.items():
            exp = sum(item.values())
            isUpper = exp > 0
            if not isUpper: exp *= -1
            if exp == 1: exp = ''
            s = f'{key}{exp}'
            if isUpper:
                upper.append(s)
            else:
                lower.append(s)
        
        out = '-'.join(upper)
        if lower:
            out += '/' + '-'.joing(lower)
            
        return out
        
    @ staticmethod
    def _splitCompositeUnit(compositeUnit):
        lower = []
        if not slash in compositeUnit:
            upper = compositeUnit.split(hyphen)
            return upper, lower
    
        compositeUnit = compositeUnit.split(slash)

        if len(compositeUnit) > 2:
            raise ValueError('A unit can only have a single slash (/)')

        upper = compositeUnit[0].split(hyphen)
        lower = compositeUnit[1].split(hyphen) if len(compositeUnit) > 1 else []

        return upper, lower

    @staticmethod
    def _splitUnitExponentAndPrefix(unitStr):
        prefixUnit, exponent = unit._removeExponentFromUnit(unitStr)
        u, prefix = unit._removePrefixFromUnit(prefixUnit)
        return u, prefix, exponent

    @staticmethod
    def _reduceDict(unitDict):
        
        ## loop over all units, and remove any prefixes, which has an exponenet of 0
        for key, item in unitDict.items():
            prefixesToRemove = []
            for pre, exp in item.items():
                if exp == 0:
                    prefixesToRemove.append(pre)
            for pre in prefixesToRemove:
                unitDict[key].pop(pre)
    
        ## remove the units, which has no items in their dictionary
        keysToRemove = []
        for key, item in unitDict.items():
            if not item:
                keysToRemove.append(key)
        for key in keysToRemove:
            unitDict.pop(key)
        
        ## remove the unit '1' above the fraction line, if there are any other units above the fraction line
        keys = list(unitDict.keys())
        if len(keys) > 1 and '1' in keys:
            otherUpper = False
            for key, item in unitDict.items():
                if key == '1':
                    continue
                for pre, exp in item.items():
                    if exp > 0:
                        otherUpper = True
                        break
                if otherUpper:
                    break
            if otherUpper:
                unitDict.pop('1')
                
        ## add the units '1' if there are not other units
        if not unitDict:
            unitDict = {'1': {None: 1}}
        
        ## set the exponent of the unit '1' to 1
        if '1' in unitDict:
            unitDict['1'][None] = 1
        
        ## make temperature units in to temperature differences, if there are any other units in the dict
        for temperatureUnit in temperature.keys():
            if temperatureUnit in unitDict:
                otherKeys = list(unitDict.keys())
                otherKeys.remove(temperatureUnit)
                if otherKeys:
                    unitDict['DELTA' + temperatureUnit] = unitDict[temperatureUnit]
                    unitDict.pop(temperatureUnit)
        
        
        
        return unitDict
    
    @staticmethod
    def _getUnitDictSI(unitDict):
        out = {}
        for key, item in unitDict.items():
            unitSI = knownUnits[key][0]
            upper,lower = unit._splitCompositeUnit(unitSI)
            nUpper = len(upper)
            for exp in item.values():
                for i, elem in enumerate(upper + lower):
                    u,p,e = unit._splitUnitExponentAndPrefix(elem)
                    if i > nUpper - 1:
                        e *= -1
                    if u in out:
                        if p in out[u]:
                            out[u][p] += e * exp
                        else:
                            out[u][p] = e * exp
                    else:
                        out[u] = {p: e * exp} 
        out = unit._reduceDict(out)                 
        return out

    @staticmethod
    def _getUnitDict(unitStr):
        upper, lower = unit._splitCompositeUnit(unitStr)

        out = {}
        
        nUpper = len(upper)
        for i, elem in enumerate(upper + lower):
            
            u, p, e = unit._splitUnitExponentAndPrefix(elem)

            if i > nUpper - 1:
                e *= -1
            
            if not u in out:
                out[u] = {p: e}
            else:
                if not p in out[u]:
                    out[u][p] = e
                else:
                    out[u][p] += e
        
        out = unit._reduceDict(out)
        
        return out
   
    @ staticmethod
    def _formatUnitStr(unitStr):
        
        ## remove spaces
        unitStr = unitStr.replace(' ', '')
        
        # check for any illegal symbols
        for s in unitStr:
            if s not in knownCharacters:
                raise ValueError(f'The character {s} is not used within the unitsystem')
        
        ## return the unity        
        if unitStr is None or unitStr == '':
            return '1'

        ## find start parenthesis is the unit string
        startParenIndexes = [i for i, s in enumerate(unitStr) if s == '(']
        stopParenIndexes = [i for i, s in enumerate(unitStr) if s == ')']
        
        if not startParenIndexes and not stopParenIndexes:
            unitStr = unitStr.split('-')
            unitStr = [elem for elem in unitStr if not elem == '']
            unitStr = '-'.join(unitStr) if unitStr else '1'
            return unitStr
        
        ## check that the number of start and stop parenthesis are equal
        if len(startParenIndexes) != len(stopParenIndexes):
            raise ValueError('The unit string has to have an equal number of open parenthesis and close parenthesis')
        
        if len(startParenIndexes) == 1 and startParenIndexes[0] == 0 and stopParenIndexes[0] == len(unitStr) -1:
            return unit._formatUnitStr(unitStr[startParenIndexes[0]+1:stopParenIndexes[0]])
        
        ## find all parenthesis pairs
        allIndexes = startParenIndexes + stopParenIndexes
        allIndexes.sort()
        isStartParen = [elem in startParenIndexes for elem in allIndexes]
        done, iter, maxIter, parenOrder  = False, 0, len(allIndexes), []
        while not done:
            for i in range(len(allIndexes)-1):
                if isStartParen[i] and (isStartParen[i] + isStartParen[i+1] == 1):
                    parenOrder.append([allIndexes[i], allIndexes[i+1]])
                    allIndexes.pop(i+1)
                    allIndexes.pop(i)
                    isStartParen.pop(i+1)
                    isStartParen.pop(i)
                    break

            if len(allIndexes) == 0: break
            
            iter += 1
            if iter > maxIter:
                raise ValueError('An order to evaluate the parenthesis could not be found')
        
        
        ## find any slashes outside of parenthesis
        slashOutsideParensFound = False
        index = -1
        parenLevel = 0
        for i, s in enumerate(unitStr):
            if s == '(':
                parenLevel += 1
            elif s== ')':
                parenLevel -= 1
            
            if parenLevel == 0 and s == '/':
                if not slashOutsideParensFound:
                    index = i
                    slashOutsideParensFound = True
                else:
                    raise ValueError("You can only have a signle slash ('/') outside of parenthesis")
        
        if slashOutsideParensFound:
            ## there was a slash outside of the parenthesis.
            ## split the unitStr at the slash and return the upper divided by the lower
            upper = unitStr[0:index]
            lower = unitStr[index+1:]
            upper = unit._getUnitDict(unit._formatUnitStr(upper))
            lower = unit._getUnitDict(unit._formatUnitStr(lower))
            return unit._getUnitStrFromDict(unit.staticTruediv(upper, lower))
            
        ## there were no slashes outside of the parenthesis
        ## append the entire unit as a "parenthesis"
        parenOrder.append([0,len(unitStr)])
        
        ## loop over the parenthesis from the inner parenthesis to the outer parenthesis
        for i in range(len(parenOrder)-1):
            currentParens = parenOrder[i]
            end = currentParens[1]
            nextParens = parenOrder[i+1]
            
            ## get the unitDict from the stringwithin the current parentheses
            _unitStr = unitStr[currentParens[0]+1:currentParens[1]]
            _unitStrLenOriginal = len(_unitStr)
            _unitStr = unit._formatUnitStr(_unitStr)
            _unitDict = unit._getUnitDict(_unitStr)

            ## determine if the nextparenthesis encompasses the current parenthesis
            if nextParens[0] <= currentParens[0] and nextParens[1]>=currentParens[1]:
                
                ## get the string from the end of the current parenthesis to the end of the next parenthesis
                nextBit = unitStr[currentParens[1]+1:nextParens[1]]
                
                ## A potential exponent of the current parenthis has to be
                # above a potential slash (/) and before any potential hyphens (-)
                exponent = nextBit.split('/')[0].split('-')[0]
                
                
                ## try to cast the exponent to an integer
                ## if this works, then raise the unitDict to the exponent
                try:
                    exponent = int(exponent)
                    _unitDict = unit.staticPow(_unitDict, None, _unitStr, exponent)
                    _unitStr = unit._getUnitStrFromDict(_unitDict)
                    end += len(str(exponent))
                except ValueError: pass
        

            ## update the next parenthesis based on the change of lengt of _unitStr
            dLen = len(_unitStr) - _unitStrLenOriginal
            for j in range(i+1, len(parenOrder)):
                if parenOrder[j][0] >= currentParens[1]: parenOrder[j][0] += dLen-1
                if parenOrder[j][1] >= currentParens[1]: parenOrder[j][1] += dLen-2
                
            ## update unitStr
            unitStr = unitStr[0:currentParens[0]] + _unitStr + unitStr[end+1:]

        return unitStr
 
    @ staticmethod
    def _removeExponentFromUnit(u):
        
        integerIndexes = [i for i, char in enumerate(u) if char in integers]
        
        if not integerIndexes:
            return u, 1

        for i in range(len(integerIndexes)-1):
            if integerIndexes[i+1] - integerIndexes[i] != 1:
                raise ValueError('All numbers in the unit has to be grouped together')

        # Determine if the last integer is placed at the end of the unit
        if integerIndexes[-1] != len(u) - 1:
            raise ValueError('Any number has to be placed at the end of the unit')

        u, exponent = u[:integerIndexes[0]], int(u[integerIndexes[0]:])


        # Ensure that the entire use was not removed by removing the integers
        if not u:
            # No symbols are left after removing the integers
            if exponent != 1:
                raise ValueError(f'The unit {u} was stripped of all integers which left no symbols in the unit. This is normally due to the integers removed being equal to 1, as the unit is THE unit. Howver, the intergers removed was not equal to 1. The unit is therefore not known.')
            u = '1'
        return u, exponent

    @staticmethod
    def _removePrefixFromUnit(unit):
        
        if unit in knownUnits:
            return unit, None

        # The unit was not found. This must be because the unit has a prefix
        found = False
        
        for p in knownPrefixes.keys():
            index = unit.find(p)
            if index != 0:
                continue
            u = unit[len(p):]
            if not u in knownUnits:
                continue
            found = True
            prefix, unit = p,u
            break
        

        if not found:
            raise ValueError(f'The unit ({unit}) was not found. Therefore it was interpreted as a prefix and a unit. However a combination of prefix and unit which matches {unit} was not found')
        
        if unit in baseUnit and unit != "%":
            unit = "1"
            raise ValueError(f'The unit ({prefix}) was not found. Therefore it was interpreted as a prefix and a unit. The prefix was identified as "{p}" and the unit was identified as "{unit}". However, the unit "1" cannot have a prefix')

        # look for the unit without the prefix
        if not unit in knownUnits:
            raise ValueError(f'The unit ({prefix}{unit}) was not found. Therefore it was interpreted as a prefix and a unit. However the unit ({unit}) was not found')
        return unit, prefix

    def __str__(self, pretty=False):
        
        if not pretty:
            return self.unitStr

        upper, lower = [],[]
        for u, item in self.unitDict.items():
            for pre, exp in item.items():
                isUpper = exp > 0
                if not isUpper: exp *= -1 
                if exp == 1: exp = ''
                if pre == None: pre = ''
                s = f'{pre}{u}{exp}'
                if isUpper:
                    upper.append(s)
                else:
                    lower.append(s)
        
        if not lower:
            # no fraction
            out = '\cdot'.join(upper)
            return out
        
        # a fraction is needed
        out = rf'\frac{{'
        out += '\cdot'.join(upper)
        out += rf'}}{{'
        out += '\cdot'.join(lower)
        out += rf'}}'
        return out

    def __eq__(self, other):
        return self.unitDict == other.unitDict

    def isLogarithmicUnit(self):
        upper, lower = unit._splitCompositeUnit(self.unitStr)
        if lower or len(upper) != 1: return False
        return unit._splitUnitExponentAndPrefix(upper[0])[0] in logrithmicUnits

    def __add__(self, other):
        ## determine the units of self without any prefixes and convert this to a string
        selfUnitDictWithoutPrefixes = {}
        for key, item in self.unitDict.items():
            selfUnitDictWithoutPrefixes[key] = {}
            selfUnitDictWithoutPrefixes[key][None] = 0
            for exp in item.values():
                selfUnitDictWithoutPrefixes[key][None] += exp
        
        ## determine if self is the same as other - then no conversions are necessary
        if self.unitDict == other.unitDict:
            return unit(unitDict=self.unitDict)  
        
        ## determine the units of other without any prefixes and convert this to a string
        otherUnitDictWithoutPrefixes = {}
        for key, item in other.unitDict.items():
            otherUnitDictWithoutPrefixes[key] = {}
            otherUnitDictWithoutPrefixes[key][None] = 0
            for exp in item.values():
                otherUnitDictWithoutPrefixes[key][None] += exp
        
        ## determine if the self and other are identical once any prefixes has been removed
        if selfUnitDictWithoutPrefixes == otherUnitDictWithoutPrefixes:
            return unit(unitDict=selfUnitDictWithoutPrefixes)
        
        # determine if the SI base units of self and other are equal
        if self.unitDictSI == other.unitDictSI:
            return unit(unitDict=copy(self.unitDictSI))
        
        ## determine if "DELTAK" and "K" are the SI Baseunits of self and other
        SIBaseUnits = [self.unitDictSI, other.unitDictSI]        
        if {'DELTAK':{None:1}} in SIBaseUnits and {'K':{None:1}} in SIBaseUnits:
            
            indexTemp = SIBaseUnits.index({'K':{None:1}})
            indexDiff = 0 if indexTemp == 1 else 1
            
            units = [list(selfUnitDictWithoutPrefixes.keys())[0], list(otherUnitDictWithoutPrefixes.keys())[0]]

            if units[indexTemp] == units[indexDiff][-1]:        
                return unit(unitDict=[selfUnitDictWithoutPrefixes, otherUnitDictWithoutPrefixes][indexTemp])
            
            return unit('K')

        raise ValueError(f'You tried to add a variable in [{self}] to a variable in [{other}], but the units do not have the same SI base unit')

    def __sub__(self, other):
        
        ## determine the units of self without any prefixes
        selfUnitDictWithoutPrefixes = {}
        for key, item in self.unitDict.items():
            selfUnitDictWithoutPrefixes[key] = {}
            selfUnitDictWithoutPrefixes[key][None] = 0
            for exp in item.values():
                selfUnitDictWithoutPrefixes[key][None] += exp
        
        
        ## determine the units of other without any prefixes and convert
        otherUnitDictWithoutPrefixes = {}
        for key, item in other.unitDict.items():
            otherUnitDictWithoutPrefixes[key] = {}
            otherUnitDictWithoutPrefixes[key][None] = 0
            for exp in item.values():
                otherUnitDictWithoutPrefixes[key][None] += exp
        
        
        ## determine if "DELTAK" and "K" are the SI Baseunits of self and other
        SIBaseUnits = [self.unitDictSI, other.unitDictSI]        
        if SIBaseUnits[0] == {'K':{None:1}} and SIBaseUnits[1] == {'K':{None:1}}:
            if self.unitDict == other.unitDict:
                return unit(unitDict = copy(self.unitDict))
            return unit('K')

        if {'DELTAK':{None:1}} in SIBaseUnits and {'K':{None:1}} in SIBaseUnits:
            indexTemp = SIBaseUnits.index({'K':{None:1}})
            if indexTemp != 0:
                raise ValueError('You tried to subtract a temperature from a temperature differnce. This is not possible.')      
            return unit(unitDict=[selfUnitDictWithoutPrefixes, otherUnitDictWithoutPrefixes][indexTemp])
        
                
        ## determine if self is the same as other - then no conversions are necessary
        if self.unitDict == other.unitDict:
            return unit(unitDict=self.unitDict)
        
        ## determine if the self and other are identical once any prefixes has been removed
        if selfUnitDictWithoutPrefixes == otherUnitDictWithoutPrefixes:
            return unit(unitDict=selfUnitDictWithoutPrefixes)
        
        # determine if the SI base units of self and other are equal
        if self.unitDictSI == other.unitDictSI:
            return unit(unitDict=self.unitDictSI)
        
        
        raise ValueError(f'You tried to subtract a variable in [{other}] from a variable in [{self}], but the units do not have the same SI base unit')

    def __mul__(self, other):
        out = {}
        
        for key, item in self.unitDict.items():
            out[key]= {}
            for pre, exp in item.items():
                out[key][pre] = exp
        
        for key, item in other.unitDict.items():
            if not key in out:
                out[key] = item
            else:
                for pre, exp in item.items():
                    if not pre in out[key]:
                        out[key][pre] = exp
                    else:
                        out[key][pre] += exp
        
        out = unit._reduceDict(out)
        return unit(unitDict = out)
    
    def __truediv__(self, other):
        return unit(unitDict = unit.staticTruediv(self.unitDict, other.unitDict))
    
    @staticmethod
    def staticTruediv(a, b):
        out = {}
        
        for key, item in a.items():
            out[key]= {}
            for pre, exp in item.items():
                out[key][pre] = exp
        
        for key, item in b.items():
            if not key in out:
                out[key] = {}
                for pre, exp in item.items():
                    out[key][pre] = -1 * exp
            else:
                for pre, exp in item.items():
                    if not pre in out[key]:
                        out[key][pre] = -exp
                    else:
                        out[key][pre] -= exp
        
        return unit._reduceDict(out)
    
    def __pow__(self, power):
        return unit(unitDict= unit.staticPow(self.unitDict, self.unitDictSI, self.unitStr, power))
    
    @staticmethod
    def staticPow(unitDict, unitDictSI, unitStr, power):
        
        if unitDict == {'1': {None: 1}}:
            return unitDict
        
        frac = Fraction(power).limit_denominator()
        num, den = frac._numerator, frac._denominator
        
        ## determine if it is possible to take the power of the unitDict
        isPossible = True
        out = {}
        for key, item in unitDict.items():
            for pre, exp in item.items():
                if not (exp * num) % den == 0:
                    isPossible = False
                    break
            if not isPossible:
                break
        
        ## if it is possible, then return a new unitDict
        if isPossible:
            for key, item in unitDict.items():
                out[key] = {}
                for pre, exp in unitDict[key].items():
                    out[key][pre] = int(num * exp / den)
            out = unit._reduceDict(out)
            return out
        
        ## determine if it is possible to take the power of the sibase unit
        isPossible = True
        out = {}
        for key, item in unitDictSI.items():
            for pre, exp in item.items():
                if not (exp * num) % den == 0:
                    isPossible = False
                    break
            
        ## if it is possible, then return a new unitDict
        if isPossible:
            for key, item in unitDictSI.items():
                out[key] = {}
                for pre, exp in item.items():
                    out[key][pre] = int(num * exp / den)
            out = unit._reduceDict(out)
            return out
        
        raise ValueError(f'You can not raise a variable with the unit {unitStr} to the power of {power}')

    def getConverter(self, newUnitStr):  
        newUnitStr =  unit._formatUnitStr(newUnitStr)
        newUnitDict = unit._getUnitDict(newUnitStr)
        newUnitDictSI = unit._getUnitDictSI(newUnitDict)
        if not (self.unitDictSI == newUnitDictSI):
            raise ValueError(f'You tried to convert from {self} to {newUnitStr}. But these do not have the same base units')
        return self.getConverterFromDict(newUnitDict)
    
    def getConverterFromDict(self, newUnitDict):    
        
        # initialize the scale and offset
        out = _unitConversion(1, 0)
        
        ## loop over self.unitDict
        for u, item in self.unitDict.items():
            for pre, exp in item.items():
                conv = knownUnits[u][1]
                if not pre is None: conv *= knownPrefixes[pre]
                isUpper = exp > 0
                if not isUpper: exp *= -1                    
                conv = conv ** exp
                out = out * conv if isUpper else out / conv

        ## loop over newUnitDict
        for u, item in newUnitDict.items():
            for pre, exp in item.items():
                conv = knownUnits[u][1]
                if not pre is None: conv *= knownPrefixes[pre]
                isUpper = exp > 0
                if not isUpper: exp *= -1
                conv = conv ** exp
                out = out * conv if not isUpper else out / conv
        
        return out

    def isCombinationUnit(self):
        return len(list(self.unitDict.keys())) > 1

    def getLogarithmicConverter(self, unitStr = None):
        if unitStr is None:
            unitStr = self.unitStr
        u, _ = self._removePrefixFromUnit(unitStr)
        return knownUnits[u][1]

    
if __name__ == "__main__":
    a = unit('K')
    converter = a.getConverter('C')
    print(converter.convert(300), 26.85)
    