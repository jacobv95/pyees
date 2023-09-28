import numpy as np
from fractions import Fraction

class _unitConversion():
    def __init__(self, scale, offset=0) -> None:
        self.scale, self.offset = scale, offset
     
    @staticmethod
    def staticMul(selfScale, selfOffset, otherScale, otherOffset = 0):
        scale = selfScale * otherScale
        offset = selfScale * otherOffset + selfOffset
        return scale, offset
    
    @staticmethod
    def staticPow(scale, offset, pow):
        if pow == 1: return scale, offset
        
        ## invert the scale and the offset if the power is negative
        positivePower = pow > 0
        if not positivePower:
            scale, offset = _unitConversion.staticTruediv(1,0,scale, offset) * (not positivePower)
            pow =  - pow 
        
        ## raise the scale and the offset to the power
        offset = offset * sum([scale ** (i) for i in range(pow)])
        scale = scale ** pow
        return scale, offset
       
    @staticmethod
    def staticTruediv(selfScale, selfOffset, otherScale, otherOffset = 0):
        return _unitConversion.staticMul(1 / otherScale, - otherOffset / otherScale, selfScale, selfOffset)

    def convert(self, value, useOffset=True):
        return self.scale * value + useOffset * self.offset

class _neperConversion():
            
    @staticmethod
    def convertToSignal(var):
        var._uncert = 2*np.exp(2*var.value) * var.uncert
        var._uncertSI = 2*np.exp(2*var.value) * var._uncertSI
        var._value = np.exp(2*var.value)
        
    @staticmethod
    def convertFromSignal(var):
        var._uncert = 1 / (2*var.value) * var.uncert
        var._uncertSI = 1 / (2*var.value) * var._uncertSI
        var._value = 1/2 * np.log(var.value)

class _bellConversion():
        
    @staticmethod
    def convertToSignal(var):
        var._uncert = 10**var.value * np.log(10) * var.uncert
        var._uncertSI = 10**var.value * np.log(10) * var._uncertSI
        var._value = 10**var.value
        
    @staticmethod
    def convertFromSignal(var):
        var._uncert = 1 / (var.value * np.log(10)) * var.uncert
        var._uncertSI = 1 / (var.value * np.log(10)) * var._uncertSI
        var._value = np.log10(var.value)

class _octaveConversion():
        
    @staticmethod
    def convertToSignal(var):
        var._uncert = 2**var.value * np.log(2) * var.uncert
        var._uncertSI = 2**var.value * np.log(2) * var._uncertSI
        var._value = 2**var.value
        
    @staticmethod
    def convertFromSignal(var):
        var._uncert = 1 / (var.value * np.log(2)) * var.uncert
        var._uncertSI = 1 / (var.value * np.log(2)) * var._uncertSI
        var._value = np.log2(var.value)

    
    
_baseUnit = {
    '1': (1,0),
    '': (1,0),
    '%': (1e-2,0)
}

_force = {
    'N': (1,0)
}

_mass = {
    'g': (1 / 1000,0)
}

_energy = {
    'J': (1,0),
}

_power = {
    'W': (1,0)
}

_pressure = {
    'Pa': (1,0),
    'bar': (1e5,0)
}

_temperature = {
    'K': (1,0),
    'C': (1, 273.15),
    'F': (5 / 9, 273.15 - 32 * 5 / 9)
}

_temperatureDifference = {
    'DELTAK': (1,0),
    'DELTAC': (1,0),
    'DELTAF': (5 / 9,0)
}

_time = {
    's': (1,0),
    'min': (60,0),
    'h': (60 * 60,0),
    'yr': (60 * 60 * 24 * 365,0)
}

_volume = {
    'm3': (1,0),
    'L': (1 / 1000,0)
}

_length = {
    'm': (1,0),
    'Å': (1e-10,0),
    'ly': (9460730472580800,0)
}

_angle = {
    'rad': (1,0),
    'deg': (np.pi / 180,0)
}

_current = {
    'A': (1,0)
}

_voltage = {
    'V': (1,0)
}

_frequency = {
    'Hz': (1,0)
}

_resistance = {
    'ohm': (1,0)
}

_kinematicViscosity = {
    'St': (1e-4,0)
}

_logrithmicUnits = {
    'Np' : (1,0),
    'B': (1,0),
    'oct': (1,0),
    'dec': (1,0)
}

_conductance = {
    'S' : (1,0)
}

## Create a dictionary of all the si base units of each different type of measurement
## the value of this dictionary is the dictionary representation of the si units
_SIBaseUnits = {
    'Force'                     : {'g':{'k':1}, 'm':{'':1}, 's':{'':-2}}, 
    'Pressure'                  : {'g':{'k':1}, 'm':{'':-1}, 's':{'':-2}},
    'Time'                      : {'s':{'':1}},
    'Temperature'               : {'K':{'':1}},
    'Volume'                    : {'m':{'':3}},
    'Length'                    : {'m':{'':1}},
    'Energy'                    : {'g':{'k':1}, 'm':{'':2}, 's':{'':-2}},
    'Power'                     : {'g':{'k':1}, 'm':{'':2}, 's':{'':-3}},
    'Mass'                      : {'g':{'k':1}},
    'Current'                   : {'A':{'':1}},
    'Voltage'                   : {'g':{'k':1}, 'm':{'':2}, 's':{'':-3}, 'A':{'':-1}},
    'BaseUnit'                  : {'1':{'':1}},
    'Freqeuncy'                 : {'s':{'':-1}},
    'Angle'                     : {'rad':{'':1}},
    'Resistance'                : {'g':{'k':1}, 'm':{'':2}, 's':{'':-3}, 'A':{'':-2}},
    'KinematicViscosity'        : {'m':{'':2},'s':{'':-1}},
    'LogarithmicUnit'           : {'Np':{'':1}},
    'TemperatureDifference'     : {'DELTAK':{'':1}},
    'Conductance'              : {'g':{'k':-1}, 'm':{'':-2}, 's':{'':3}, 'A':{'':2}},
}

## create a dictionary of all units for each type of measurement.
## the values of this dictionary is itself a dictionary of (string, conversion) pairs
_knownUnitsDict = {
    'Force':                        _force,
    'Pressure':                     _pressure,
    'Time':                         _time,
    'Temperature':                  _temperature,
    'Volume':                       _volume,
    'Length':                       _length,
    'Energy' :                      _energy,
    'Power' :                       _power,
    'Mass':                         _mass,
    'Current':                      _current,
    'Voltage' :                     _voltage,
    'BaseUnit':                     _baseUnit,
    'Freqeuncy':                    _frequency,
    'Angle':                        _angle,
    'Resistance':                   _resistance,
    'KinematicViscosity':           _kinematicViscosity,
    'LogarithmicUnit':              _logrithmicUnits,
    'TemperatureDifference':        _temperatureDifference,
    'Conductance':                  _conductance
}

## create a dictionary of the known prefixes
_knownPrefixes = {
    'T':10**12,
    'G':10**9,
    'M':10**6,
    'k':10**3,
    'h':10**2,
    'da':10**1,
    '': 1,
    'd':10**-1,
    'c':10**-2,
    'm':10**-3,
    'mu':10**-6,
    'n':10**-9,
    'p':10**-12
}

## create a dictionary of all known units
## the keys of this dictionary is the stringrepresenation of the unit
## the values is a list of two elements:
## the first element is the dictionary representation of the corresponding si unit
## the second element is the conversion from the unit to the corresponding si unit
_knownUnits = {}
for key in _SIBaseUnits.keys():
    sibaseunit = _SIBaseUnits[key]
    for kkey, conversion in _knownUnitsDict[key].items():
        if kkey not in _knownUnits:
            _knownUnits[kkey] = [sibaseunit, conversion]
        else:
            raise Warning(f'The unit {kkey} is known in more than one unit system')
    


# determine the known characters within the unit system
_knownCharacters = list(_knownUnits.keys()) + list(_knownPrefixes.keys())
_knownCharacters = ''.join(_knownCharacters)
_knownCharacters += '-/ '
_knownCharacters += '0123456789'
_knownCharacters += '()'
_knownCharacters = set(_knownCharacters)

# check if all unit and prefix combinations can be distiguished
_unitPrefixCombinations = []
for u in _knownUnits:
    _unitPrefixCombinations += [u]
    if u not in _baseUnit or u == "%":
        for p in _knownPrefixes:
            if p == '':continue
            _unitPrefixCombinations.append(p+u)




def _checkForAmbiguityInTheUnits():
    for elem in _unitPrefixCombinations:
        count = sum([1 if u == elem else 0 for u in _unitPrefixCombinations])
        if count > 1:
            
            def splitPrefixAndUnit(unit):
                
                # The unit was not found. This must be because the unit has a prefix
                found = False
                
                for p in _knownPrefixes:
                    if p != unit[0:len(p)]: continue
                    u = unit[len(p):]
                    if not u in _knownUnits: continue
                    found = True
                    prefix, unit = p,u
                    break 

                if not found:
                    raise ValueError(f'The unit ({unit}) was not found. Therefore it was interpreted as a prefix and a unit. However a combination of prefix and unit which matches {unit} was not found')
                
                if unit in _baseUnit and unit != "%":
                    unit = "1"
                    raise ValueError(f'The unit ({prefix}) was not found. Therefore it was interpreted as a prefix and a unit. The prefix was identified as "{p}" and the unit was identified as "{unit}". However, the unit "1" cannot have a prefix')

                # look for the unit without the prefix
                if not unit in _knownUnits:
                    raise ValueError(f'The unit ({prefix}{unit}) was not found. Therefore it was interpreted as a prefix and a unit. However the unit ({unit}) was not found')
                return unit, prefix

            unit, prefix = splitPrefixAndUnit(elem)

            unitType1 = ''
            for key, item in _knownUnitsDict.items():
                if elem in item:
                    unitType1 = key
            if unitType1 == '':
                raise ValueError(f'The unit {elem} was not found.')

            unitType2 = ''
            for key, item in _knownUnitsDict.items():
                if unit in item:
                    unitType2 = key
            if unitType2 == '':
                raise ValueError(f'The unit {unit} was not found.')

            raise ValueError(f'The unit {elem} can be interpreted as a {unitType1} ({elem}) or a {unitType2} in the unit "{unit}" with the prefix "{prefix}". This cannot be distiguished.')



def addNewUnit(newUnit: str, scale: float, existingUnitStr: str, offset : float = 0):
    
    ## add the newUnit to the unitPrefix combinations
    _unitPrefixCombinations.append(newUnit)
    for p in _knownPrefixes:
        if p == '': continue
        _unitPrefixCombinations.append(p+newUnit)

    ## create a unitObject from the existing unit string
    existingUnit = unit(existingUnitStr)
    
    ## the SI unit of the existing unit must be the same as that of the new unit
    newUnitDictSI = existingUnit.unitDictSI
    
    ## determine the scale and the offset from the new unit to the si unit using the scale and the offset from the existing unit
    scale, offset = _unitConversion.staticMul(existingUnit._converterToSI.scale, existingUnit._converterToSI.offset, scale, offset)
   
    ## add the new unit to the knownUnits
    _knownUnits[newUnit] = [newUnitDictSI, (scale, offset)]
    
    ## add the newunit to the values of knownUnitsDict. This is used to check for ambiguity in the unit system
    for item in _knownUnitsDict.values():
        if existingUnitStr in item:
            item[newUnit] = (scale,offset)
    
    ## add the characters of the new unit to the knownCharacters
    for s in newUnit:
        _knownCharacters.add(s)
        
    ## check for ambiguity in the unit system
    _checkForAmbiguityInTheUnits()

    

hyphen = '-'
slash = '/'
integers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



class unit():

    def __init__(self, unitStr = None, unitDict = None, unitDictSI = None, selfUnitStr = None, selfUnitStrSI = None, converterToSI = None):
        if unitStr is None:
            unitStr = ''
        
        if unitDict is None:
            self.unitDict = self._getUnitDict(self._formatUnitStr(unitStr))
        else:
            self.unitDict = unitDict
        
        if unitDictSI is None:
            self.unitDictSI = self._getUnitDictSI(self.unitDict)
        else:
            self.unitDictSI = unitDictSI    
        
        if selfUnitStr is None:
            self.unitStr = self._getUnitStrFromDict(self.unitDict)
        else:
            self.unitStr = selfUnitStr
        
        if selfUnitStrSI is None:
            self.unitStrSI = self._getUnitStrFromDict(self.unitDictSI)
        else:
            self.unitStrSI = selfUnitStrSI
        
        if converterToSI is None:
            self.getConverterToSI()
        else:
            self._converterToSI = converterToSI

    @staticmethod
    def _getUnitStrFromDict(unitDict):
        upper, lower = '',''
        
        for u, item in unitDict.items():    
            for p, exp in item.items():
                isUpper = exp > 0
                if not isUpper: exp = - exp
                exp = str(exp) if exp != 1 else ''
                s = p + u + exp + '-'
                if isUpper: upper += s
                else: lower += s
                
        upper = upper[:-1]
        if lower: upper += '/' + lower[:-1]
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
        if not slash in compositeUnit:
            upper = compositeUnit.split(hyphen)
            return upper, []
    
        compositeUnit = compositeUnit.split(slash)

        if len(compositeUnit) > 2:
            raise ValueError(f'A unit can only have a single slash ({slash})')

        upper = compositeUnit[0].split(hyphen)
        lower = compositeUnit[1].split(hyphen) if len(compositeUnit) == 2 else []

        return upper, lower

    @staticmethod
    def _splitUnitExponentAndPrefix(unitStr):
        prefixUnit, exponent = unit._removeExponentFromUnit(unitStr)
        u, prefix = unit._removePrefixFromUnit(prefixUnit)
        return u, prefix, exponent
              
    @staticmethod
    def _reduceDict(unitDict):
        
        ## check if 1 is in the unit
        n = len(unitDict)
        if '1' in unitDict:
            if n > 1:     
                otherUpper = False
                for key, item in unitDict.items():
                    if key == '1': continue
                    for exp in item.values():
                        if exp > 0:
                            otherUpper = True
                            break
                    if otherUpper: break

                    
                ## if there are any other upper units, then remove 1
                ## else set the exponent of the unit '1' to 1
                if otherUpper:
                    unitDict.pop('1')
                    n -= 1
                else: unitDict['1'][''] = 1
            else: unitDict['1'][''] = 1
                
        ## make temperature units in to temperature differences, if there are any other units in the dict
        if n > 1:
            keysToChange = []
            for key in _temperature.keys():
                if key in unitDict: keysToChange.append(key)
            for key in keysToChange:
                if 'DELTA' + key in unitDict:
                    for pre, exp in unitDict[key].items():
                        if pre in unitDict['DELTA' + key]:
                            unitDict['DELTA' + key][pre] += exp
                        else:
                            unitDict['DELTA' + key][pre] = exp
                else:
                    unitDict['DELTA' + key] = unitDict[key]
                unitDict.pop(key)
        
       
        ## loop over all units, and remove any prefixes, which has an exponenet of 0
        keysToRemove = []
        for key, item in unitDict.items():
            prefixesToRemove = []
            for pre, exp in item.items():
                if exp == 0:  prefixesToRemove.append(pre)
            for pre in prefixesToRemove: item.pop(pre)
            if not item: keysToRemove.append(key)
        n -= len(keysToRemove)        
        
        ## if all the keys in unitDict has to be removed, then return the unit '1'
        if n == 0:
            ## return '1' if there are not other units
            return {'1': {'': 1}}
        
        ## remove the keys
        for key in keysToRemove: unitDict.pop(key)
        
        return unitDict
    
    @staticmethod
    def _getUnitDictSI(unitDict):
        out = {}
        for key, item in unitDict.items():
            exp = sum(item.values())
            unitSI = _knownUnits[key][0]
            for kkey, iitem in unitSI.items():
                for p, e in iitem.items():
                    e = e * exp
                    if kkey in out: e = e + out[kkey][p]
                    out[kkey] = {p: e}
                    
                    
        out = unit._reduceDict(out)                 
        return out

    @staticmethod
    def _getUnitDict(unitStr):
        upper, lower = unit._splitCompositeUnit(unitStr)

        out = {}
        nUpper = len(upper)
        for i, elem in enumerate(upper + lower):
            u, p, e = unit._splitUnitExponentAndPrefix(elem)
            if i > nUpper - 1: e = - e
            if not u in out:
                out[u] = {p: e}
            else:
                if p in out[u]: e += out[u][p]
                out[u][p] = e
        
        return unit._reduceDict(out)
    
   
    @ staticmethod
    def _formatUnitStr(unitStr):
        
        ## remove spaces
        unitStr = unitStr.replace(' ', '')
        
        # check for any illegal symbols
        for s in unitStr:
            if s not in _knownCharacters:
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
        
        if unit in _knownUnits:
            return unit, ''

        # The unit was not found. This must be because the unit has a prefix
        found = False
        
        for p in _knownPrefixes:
            if p != unit[0:len(p)]: continue
            u = unit[len(p):]
            if not u in _knownUnits: continue
            found = True
            prefix, unit = p,u
            break 

        if not found:
            raise ValueError(f'The unit ({unit}) was not found. Therefore it was interpreted as a prefix and a unit. However a combination of prefix and unit which matches {unit} was not found')
        
        if unit in _baseUnit and unit != "%":
            unit = "1"
            raise ValueError(f'The unit ({prefix}) was not found. Therefore it was interpreted as a prefix and a unit. The prefix was identified as "{p}" and the unit was identified as "{unit}". However, the unit "1" cannot have a prefix')

        # look for the unit without the prefix
        if not unit in _knownUnits:
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
        out += '\cdot '.join(upper)
        out += rf'}}{{'
        out += '\cdot '.join(lower)
        out += rf'}}'
        return out
    
    def __eq__(self, other):
        return self.unitDict == other.unitDict

    def isLogarithmicUnit(self):
        upper, lower = unit._splitCompositeUnit(self.unitStr)
        if lower or len(upper) != 1: return False
        return unit._splitUnitExponentAndPrefix(upper[0])[0] in _logrithmicUnits

    def __add__(self, other):
        ## determine the units of self without any prefixes and convert this to a string
        selfUnitDictWithoutPrefixes = {}
        for key, item in self.unitDict.items():
            selfUnitDictWithoutPrefixes[key] = {}
            selfUnitDictWithoutPrefixes[key][''] = 0
            for exp in item.values():
                selfUnitDictWithoutPrefixes[key][''] += exp
        
        ## determine if self is the same as other - then no conversions are necessary
        if self.unitDict == other.unitDict:
            return self
            
        ## determine the units of other without any prefixes and convert this to a string
        otherUnitDictWithoutPrefixes = {}
        for key, item in other.unitDict.items():
            otherUnitDictWithoutPrefixes[key] = {}
            otherUnitDictWithoutPrefixes[key][''] = 0
            for exp in item.values():
                otherUnitDictWithoutPrefixes[key][''] += exp
        
        ## determine if the self and other are identical once any prefixes has been removed
        if selfUnitDictWithoutPrefixes == otherUnitDictWithoutPrefixes:
            return unit(unitDict=selfUnitDictWithoutPrefixes, unitDictSI=self.unitDictSI, selfUnitStrSI=self.unitStrSI)
        
        # determine if the SI base units of self and other are equal
        if self.unitDictSI == other.unitDictSI:
            return unit(unitDict=self.unitDictSI, unitDictSI=self.unitDictSI, selfUnitStr=self.unitStrSI, selfUnitStrSI = self.unitStrSI, converterToSI=_unitConversion(1,0))
        
        ## determine if "DELTAK" and "K" are the SI Baseunits of self and other
        SIBaseUnits = [self.unitDictSI, other.unitDictSI]        
        if {'DELTAK':{'':1}} in SIBaseUnits and {'K':{'':1}} in SIBaseUnits:
            
            indexTemp = SIBaseUnits.index({'K':{'':1}})
            indexDiff = 0 if indexTemp == 1 else 1
            
            units = [self.unitStr, other.unitStr]

            ## check to see if the temperature differnce has the same unit as the temperature
            if units[indexTemp] == units[indexDiff][-1]:
                return [self, other][indexTemp]        
                
            return unit(unitDict={'K':{'':1}}, unitDictSI={'K':{'':1}}, selfUnitStr = 'K', selfUnitStrSI = 'K', converterToSI=_unitConversion(1,0))

        raise ValueError(f'You tried to add a variable in [{self}] to a variable in [{other}], but the units do not have the same SI base unit')

    def __sub__(self, other):
        
        ## determine the units of self without any prefixes
        selfUnitDictWithoutPrefixes = {}
        for key, item in self.unitDict.items():
            selfUnitDictWithoutPrefixes[key] = {}
            selfUnitDictWithoutPrefixes[key][''] = 0
            for exp in item.values():
                selfUnitDictWithoutPrefixes[key][''] += exp
        
        
        ## determine the units of other without any prefixes and convert
        otherUnitDictWithoutPrefixes = {}
        for key, item in other.unitDict.items():
            otherUnitDictWithoutPrefixes[key] = {}
            otherUnitDictWithoutPrefixes[key][''] = 0
            for exp in item.values():
                otherUnitDictWithoutPrefixes[key][''] += exp
        
        
        ## determine if "DELTAK" and "K" are the SI Baseunits of self and other
        SIBaseUnits = [self.unitDictSI, other.unitDictSI]        
        if SIBaseUnits[0] == {'K':{'':1}} and SIBaseUnits[1] == {'K':{'':1}}:
            if self.unitDict == other.unitDict:
                return self
            return unit(unitDict={'K':{'':1}}, unitDictSI={'K':{'':1}}, selfUnitStr='K', selfUnitStrSI = 'K', converterToSI=_unitConversion(1,0))

        if {'DELTAK':{'':1}} in SIBaseUnits and {'K':{'':1}} in SIBaseUnits:
            indexTemp = SIBaseUnits.index({'K':{'':1}})
            if indexTemp != 0:
                raise ValueError('You tried to subtract a temperature from a temperature differnce. This is not possible.')      
            return [self, other][indexTemp]
                
        ## determine if self is the same as other - then no conversions are necessary
        if self.unitDict == other.unitDict:
            return self
        
        ## determine if the self and other are identical once any prefixes has been removed
        if selfUnitDictWithoutPrefixes == otherUnitDictWithoutPrefixes:
            return unit(unitDict=selfUnitDictWithoutPrefixes, unitDictSI=self.unitDictSI, selfUnitStrSI=self.unitStrSI)
        
        # determine if the SI base units of self and other are equal
        if self.unitDictSI == other.unitDictSI:
            return unit(unitDict=self.unitDictSI, unitDictSI = self.unitDictSI, selfUnitStr = self.unitStrSI, selfUnitStrSI=self.unitStrSI, converterToSI=_unitConversion(1,0))
        
        
        raise ValueError(f'You tried to subtract a variable in [{other}] from a variable in [{self}], but the units do not have the same SI base unit')

    def __mul__(self, other):
        unitDict = {}
        for key, item in self.unitDict.items():
            unitDict[key]= {}
            for pre, exp in item.items():
                unitDict[key][pre] = exp
                
        for key, item in other.unitDict.items():
            if not key in unitDict:
                unitDict[key] = {}
                for pre, exp in item.items():
                    unitDict[key][pre] = exp
            else:
                for pre, exp in item.items():
                    if not pre in unitDict[key]:
                        unitDict[key][pre] = exp
                    else:
                        unitDict[key][pre] += exp
                        
        unitDict = unit._reduceDict(unitDict)
                
        converterToSI = _unitConversion(*_unitConversion.staticMul(self._converterToSI.scale, self._converterToSI.offset, other._converterToSI.scale, other._converterToSI.offset))
        
        return unit(unitDict = unitDict, converterToSI = converterToSI)
    
    def __truediv__(self, other):
        unitDict = unit.staticTruediv(self.unitDict, other.unitDict)
        converterToSI = _unitConversion(*_unitConversion.staticTruediv(self._converterToSI.scale, self._converterToSI.offset, other._converterToSI.scale, other._converterToSI.offset))
        return unit(unitDict = unitDict, converterToSI = converterToSI)
    
    @staticmethod
    def staticTruediv(a, b):
       
        unitDict = {}
        for key, item in a.items():
            unitDict[key]= {}
            for pre, exp in item.items():
                unitDict[key][pre] = exp
                
        for key, item in b.items():
            if not key in unitDict:
                unitDict[key] = {}
                for pre, exp in item.items():
                    unitDict[key][pre] = -exp
            else:
                for pre, exp in item.items():
                    if not pre in unitDict[key]:
                        unitDict[key][pre] = -exp
                    else:
                        unitDict[key][pre] -= exp
    
        return unit._reduceDict(unitDict)
    
    def __pow__(self, power):
        unitDict = unit.staticPow(self.unitDict, self.unitDictSI, self.unitStr, power)
        return unit(unitDict= unitDict)
    
    @staticmethod
    def staticPow(unitDict, unitDictSI, unitStr, power):
        
        if unitDict == {'1': {'': 1}}:
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

    def getConverterToSI(self):
        # initialize the scale and offset
        outScale, outOffset = 1,0
        
        ## loop over self.unitDict
        for u, item in self.unitDict.items():
            for pre, exp in item.items():
                convScale, convOffset = _knownUnits[u][1]
                convScale = convScale * _knownPrefixes[pre]
                if convScale == 1 and convOffset == 0: continue
                convScale, convOffset = _unitConversion.staticPow(convScale, convOffset, exp)
                outScale, outOffset = _unitConversion.staticMul(outScale, outOffset, convScale, convOffset)
     
        self._converterToSI = _unitConversion(outScale, outOffset)

    def getConverter(self, newUnitStr):  
        newUnitStr =  unit._formatUnitStr(newUnitStr)
        newUnitDict = unit._getUnitDict(newUnitStr)
        if newUnitDict == self.unitDictSI:
            return self._converterToSI
        
        newUnitDictSI = unit._getUnitDictSI(newUnitDict)
        if not (self.unitDictSI == newUnitDictSI):
            raise ValueError(f'You tried to convert from {self} to {newUnitStr}. But these do not have the same base units')
        
        outScale, outOffset = self._converterToSI.scale, self._converterToSI.offset
        
        ## loop over newUnitDict
        for u, item in newUnitDict.items():
            for pre, exp in item.items():
                convScale, convOffset = _knownUnits[u][1]
                convScale = convScale * _knownPrefixes[pre]
                if convScale == 1 and convOffset == 0: continue
                convScale, convOffset = _unitConversion.staticPow(convScale, convOffset, exp)
                outScale, outOffset = _unitConversion.staticTruediv(outScale, outOffset, convScale, convOffset)      
            
        return _unitConversion(outScale, outOffset)

    def isCombinationUnit(self):
        return len(list(self.unitDict.keys())) > 1

    def getLogarithmicConverter(self):
        u, _ = self._removePrefixFromUnit(self.unitStr)
        
        if u == 'B':
            return _bellConversion()
        if u == 'Np':
            return _neperConversion()
        if u == 'oct':
            return _octaveConversion()
        return _bellConversion()

