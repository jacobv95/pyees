import numpy as np
from fractions import Fraction


class _unitConversion():
    def __init__(self, scale, offset=0) -> None:
        self.scale, self.offset = scale, offset

    @staticmethod
    def staticMul(selfScale, selfOffset, otherScale, otherOffset=0):
        scale = selfScale * otherScale
        offset = selfScale * otherOffset + selfOffset
        return scale, offset

    @staticmethod
    def staticPow(scale, offset, pow, num = None, den = None):
        if pow == 1:
            return scale, offset
        
        if num is None and den is None:
            frac = Fraction(pow).limit_denominator()
            num, den = frac._numerator, frac._denominator
        
        if num != 1:
            offset = offset * sum([scale ** (i) for i in range(num)])
            scale = scale ** pow
        
        if den != 1:
            scale, offset = _unitConversion.staticTruediv(1, 0, scale, offset)
            offset = offset * sum([scale ** (i) for i in range(num)])
            scale = scale ** (-pow)
        
        return scale, offset

    @staticmethod
    def staticTruediv(selfScale, selfOffset, otherScale, otherOffset=0):
        return _unitConversion.staticMul(1 / otherScale, - otherOffset / otherScale, selfScale, selfOffset)

    def convert(self, var, useOffset=True):
        value = var.value
        uncert = var.uncert
        
        val = self.scale * value + useOffset * self.offset
        unc = self.scale * uncert
        
        var._value = val
        var._uncert = unc


class _neperConversion():
    def __init__(self):
        self.linearConverter = _unitConversion(1, 0).convert

    def converterToSignal(self, var, useOffset=True):
        self.linearConverter(var, useOffset=useOffset)
        
        scale = 2*np.exp(2*var._value)
        
        var._uncert = scale * var._uncert
        var._uncertSI = scale * var._uncertSI
        var._value = np.exp(2*var._value)
        
        ## scale the uncertSI of the dependencies
        for dependency in var.dependsOn.values():
            dependency[1] *= scale    

    def converterFromSignal(self, var, useOffset=True):

        scale = 1 / (2*var._value)
        
        var._uncert = scale * var._uncert
        var._uncertSI = scale * var._uncertSI
        var._value = 1/2 * np.log(var._value)
        
        ## scale the uncertSI of the dependencies
        for dependency in var.dependsOn.values():
            dependency[1] *= scale 
        
        self.linearConverter(var, useOffset=useOffset)


class _bellConversion():
    def __init__(self):
        self.linearConverter = _unitConversion(1, 0).convert

    def converterToSignal(self, var, useOffset=True):
        self.linearConverter(var, useOffset=useOffset)
        
        scale = 10**var.value * np.log(10)
        
        var._uncert = scale * var._uncert
        var._uncertSI = scale * var._uncertSI
        var._value = 10**var.value
        
        ## scale the uncertSI of the dependencies
        for dependency in var.dependsOn.values():
            dependency[1] *= scale 




    def converterFromSignal(self, var, useOffset=True):

        scale = 1 / (var._value * np.log(10))
        
        var._uncert = scale *  var._uncert
        var._uncertSI = scale * var._uncertSI
        var._value = np.log10(var._value)
        
        ## scale the uncertSI of the dependencies
        for dependency in var.dependsOn.values():
            dependency[1] *= scale 
                
        self.linearConverter(var, useOffset=useOffset)


class _octaveConversion():
    def __init__(self):
        self.linearConverter = _unitConversion(1, 0).convert

    def converterToSignal(self, var, useOffset=True):
        self.linearConverter(var, useOffset=useOffset)
        
        scale = 2**var._value * np.log(2)
        
        var._uncert = scale * var._uncert
        var._uncertSI = scale * var._uncertSI
        var._value = 2**var._value
        
        ## scale the uncertSI of the dependencies
        for dependency in var.dependsOn.values():
            dependency[1] *= scale 

    def converterFromSignal(self, var, useOffset=True):

        scale = 1 / (var._value * np.log(2))
        
        var._uncert = scale * var._uncert
        var._uncertSI = scale * var._uncertSI
        var._value = np.log2(var._value)
        
        ## scale the uncertSI of the dependencies
        for dependency in var.dependsOn.values():
            dependency[1] *= scale 
        
        self.linearConverter(var, useOffset=useOffset)


_baseUnit = {
    '1': (1, 0),
    '': (1, 0),
    '%': (1e-2, 0)
}

_force = {
    'N': (1, 0)
}

_mass = {
    'g': (1 / 1000, 0)
}

_energy = {
    'J': (1, 0),
}

_power = {
    'W': (1, 0)
}

_pressure = {
    'Pa': (1, 0),
    'bar': (1e5, 0)
}

_temperature = {
    'K': (1, 0),
    'C': (1, 273.15),
    'F': (5 / 9, 273.15 - 32 * 5 / 9)
}

_temperatureDifference = {
    'DELTAK': (1, 0),
    'DELTAC': (1, 0),
    'DELTAF': (5 / 9, 0)
}

_time = {
    's': (1, 0),
    'min': (60, 0),
    'h': (60 * 60, 0),
    'yr': (60 * 60 * 24 * 365, 0)
}

_volume = {
    'm3': (1, 0),
    'L': (1 / 1000, 0)
}

_length = {
    'm': (1, 0),
    'Å': (1e-10, 0),
    'ly': (9460730472580800, 0)
}

_angle = {
    'rad': (1, 0),
    'deg': (np.pi / 180, 0)
}

_current = {
    'A': (1, 0)
}

_voltage = {
    'V': (1, 0)
}

_frequency = {
    'Hz': (1, 0)
}

_resistance = {
    'ohm': (1, 0)
}

_kinematicViscosity = {
    'St': (1e-4, 0)
}

_logrithmicUnits = {
    'Np': (1, 0),
    'B': (1, 0),
    'oct': (1, 0),
    'dec': (1, 0)
}

_conductance = {
    'S': (1, 0)
}

# Create a dictionary of all the si base units of each different type of measurement
# the value of this dictionary is the dictionary representation of the si units
_SIBaseUnits = {
    'Force': {('k','g') : 1 , ('','m') : 1, ('','s') : -2},
    'Pressure': {('k','g') : 1 , ('','m') : -1, ('','s') : -2},
    'Time': {('','s') : 1},
    'Temperature': {('','K') : 1},
    'Volume': {('','m') : 3},
    'Length': {('','m') : 1},
    'Energy': {('k','g') : 1 , ('','m') : 2, ('','s') : -2},
    'Power': {('k','g') : 1 , ('','m') : 2, ('','s') : -3},
    'Mass': {('k','g') : 1 },
    'Current': {('','A') : 1},
    'Voltage': {('k','g') : 1 , ('','m') : 2, ('','s') : -3, ('','A') : -1},
    'BaseUnit': {('','1') : 1},
    'Freqeuncy': {('','s') : -1},
    'Angle': {('','1') : 1},
    'Resistance': {('k','g') : 1 , ('','m') : 2, ('','s') : -3, ('','A') : -2},
    'KinematicViscosity': {('','m') : 2, ('','s') : -1},
    'LogarithmicUnit': {('','Np') : 1},
    'TemperatureDifference': {('','DELTAK') :1},
    'Conductance': {('k','g') : 1 , ('','m') : -2, ('','s') : 3, ('','A') : 2}
}

# create a dictionary of all units for each type of measurement.
# the values of this dictionary is itself a dictionary of (string, conversion) pairs
_knownUnitsDict = {
    'Force':                        _force,
    'Pressure':                     _pressure,
    'Time':                         _time,
    'Temperature':                  _temperature,
    'Volume':                       _volume,
    'Length':                       _length,
    'Energy':                      _energy,
    'Power':                       _power,
    'Mass':                         _mass,
    'Current':                      _current,
    'Voltage':                     _voltage,
    'BaseUnit':                     _baseUnit,
    'Freqeuncy':                    _frequency,
    'Angle':                        _angle,
    'Resistance':                   _resistance,
    'KinematicViscosity':           _kinematicViscosity,
    'LogarithmicUnit':              _logrithmicUnits,
    'TemperatureDifference':        _temperatureDifference,
    'Conductance':                  _conductance
}

# create a dictionary of the known prefixes
_knownPrefixes = {
    'T': 10**12,
    'G': 10**9,
    'M': 10**6,
    'k': 10**3,
    'h': 10**2,
    'da': 10**1,
    '': 1,
    'd': 10**-1,
    'c': 10**-2,
    'm': 10**-3,
    'mu': 10**-6,
    'n': 10**-9,
    'p': 10**-12,
    'µ': 10**-6
}

# create a dictionary of all known units
# the keys of this dictionary is the stringrepresenation of the unit
# the values is a list of two elements:
# the first element is the dictionary representation of the corresponding si unit
# the second element is the conversion from the unit to the corresponding si unit
_knownUnits = {}
for key in _SIBaseUnits.keys():
    sibaseunit = _SIBaseUnits[key]
    for kkey, conversion in _knownUnitsDict[key].items():
        if kkey not in _knownUnits:
            _knownUnits[kkey] = [sibaseunit, conversion]
        else:
            raise Warning(
                f'The unit {kkey} is known in more than one unit system')


# determine the known characters within the unit system
_knownCharacters = list(_knownUnits.keys()) + list(_knownPrefixes.keys())
_knownCharacters = ''.join(_knownCharacters)
_knownCharacters += '-/ 0123456789.()'
_knownCharacters = set(_knownCharacters)

# check if all unit and prefix combinations can be distiguished
_unitPrefixCombinations = []
for u in _knownUnits:
    _unitPrefixCombinations += [u]
    if u not in _baseUnit or u == "%":
        for p in _knownPrefixes:
            if p == '':
                continue
            _unitPrefixCombinations.append(p+u)


def _checkForAmbiguityInTheUnits():
    for elem in _unitPrefixCombinations:
        count = sum([1 if u == elem else 0 for u in _unitPrefixCombinations])
        if count > 1:

            def splitPrefixAndUnit(unit):

                # The unit was not found. This must be because the unit has a prefix
                found = False

                for p in _knownPrefixes:
                    if p != unit[0:len(p)]:
                        continue
                    u = unit[len(p):]
                    if not u in _knownUnits:
                        continue
                    found = True
                    prefix, unit = p, u
                    break

                if not found:
                    raise ValueError(
                        f'The unit ({unit}) was not found. Therefore it was interpreted as a prefix and a unit. However a combination of prefix and unit which matches {unit} was not found')

                if unit in _baseUnit and unit != "%":
                    unit = "1"
                    raise ValueError(
                        f'The unit ({prefix}) was not found. Therefore it was interpreted as a prefix and a unit. The prefix was identified as "{p}" and the unit was identified as "{unit}". However, the unit "1" cannot have a prefix')

                # look for the unit without the prefix
                if not unit in _knownUnits:
                    raise ValueError(
                        f'The unit ({prefix}{unit}) was not found. Therefore it was interpreted as a prefix and a unit. However the unit ({unit}) was not found')
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

            raise ValueError(
                f'The unit {elem} can be interpreted as a {unitType1} ({elem}) or a {unitType2} in the unit "{unit}" with the prefix "{prefix}". This cannot be distiguished.')


def addNewUnit(newUnit: str, scale: float, existingUnitStr: str, offset: float = 0):

    # add the newUnit to the unitPrefix combinations
    _unitPrefixCombinations.append(newUnit)
    for p in _knownPrefixes:
        if p == '':
            continue
        _unitPrefixCombinations.append(p+newUnit)

    # create a unitObject from the existing unit string
    existingUnit = unit(existingUnitStr)

    # the SI unit of the existing unit must be the same as that of the new unit
    newUnitDictSI = existingUnit.unitDictSI

    # determine the scale and the offset from the new unit to the si unit using the scale and the offset from the existing unit
    scale, offset = _unitConversion.staticMul(
        existingUnit._converterToSI.scale, existingUnit._converterToSI.offset, scale, offset)

    # add the new unit to the knownUnits
    _knownUnits[newUnit] = [newUnitDictSI, (scale, offset)]

    # add the newunit to the values of knownUnitsDict. This is used to check for ambiguity in the unit system
    for item in _knownUnitsDict.values():
        if existingUnitStr in item:
            item[newUnit] = (scale, offset)

    # add the characters of the new unit to the knownCharacters
    for s in newUnit:
        _knownCharacters.add(s)

    # check for ambiguity in the unit system
    _checkForAmbiguityInTheUnits()


hyphen = '-'
slash = '/'
integers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class unit():

    def __init__(self, unitStr=None, unitDict=None, unitDictSI=None, unitStrSI=None, converterToSI=None, unitStrPretty=None):
       
        if unitStr is None:
            unitStr = ''

        if unitDict is None:
            
            # remove spaces
            unitStr = unitStr.replace(' ', '')
            
            # check for any illegal symbols
            for s in unitStr:
                if s not in _knownCharacters:
                    raise ValueError(
                        f'The character {s} is not used within the unitsystem')

            if unitStrPretty is None:
                self.unitStr, unitStrPretty = self._formatUnitStrAndUnitStrPretty(unitStr, unitStr)
            else:
                self.unitStr = self._formatUnitStr(unitStr)
            self.unitDict = self._getUnitDict(self.unitStr)
        else:
            self.unitDict = unitDict

        if unitDictSI is None:
            self.unitDictSI = self._getUnitDictSI(self.unitDict)
        else:
            self.unitDictSI = unitDictSI

        if not hasattr(self, 'unitStr'):
            self.unitStr = self._getUnitStrFromDict(self.unitDict)

        if unitStrPretty is None:
            self.unitStrPretty = self._formatUnitStrPretty(unitStr)
        else:
            self.unitStrPretty = unitStrPretty

        if unitStrSI is None:
            self.unitStrSI = self._getUnitStrFromDict(self.unitDictSI)
        else:
            self.unitStrSI = unitStrSI

        if converterToSI is None:
            self.getConverterToSI()
        else:
            self._converterToSI = converterToSI

    @staticmethod
    def _getUnitStrFromDict(unitDict):
        upper, lower = '', ''

        for (p,u), exp in unitDict.items():
            isUpper = exp > 0
            if not isUpper:
                exp = - exp
            exp = str(exp) if exp != 1 else ''
            s = p + u + exp + '-'
            if isUpper:
                upper += s
            else:
                lower += s

        upper = upper[:-1]
        if lower:
            upper += '/' + lower[:-1]
        return upper

    def getUnitWithoutPrefix(self):

        upper, lower = [], []

        for (_,u), exp in self.unitDict.items():
            isUpper = exp > 0
            if not isUpper:
                exp *= -1
            if exp == 1:
                exp = ''
            s = f'{u}{exp}'
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
            upper = [elem for elem in compositeUnit.split(hyphen) if not elem == '']
            return upper, []

        compositeUnit = [elem for elem in compositeUnit.split(slash) if not elem == '']

        if len(compositeUnit) > 2:
            raise ValueError(f'A unit can only have a single slash ({slash})')

        upper = [elem for elem in compositeUnit[0].split(hyphen) if not elem == '']
        lower = [elem for elem in compositeUnit[1].split(hyphen) if not elem == ''] if len(compositeUnit) == 2 else []

        return upper, lower

    @staticmethod
    def _splitUnitExponentAndPrefix(unitStr):
        prefixUnit, exponent = unit._removeExponentFromUnit(unitStr)
        u, prefix = unit._removePrefixFromUnit(prefixUnit)
        return u, prefix, exponent

    @staticmethod
    def _reduceDict(unitDict):

        # check if 1 is in the unit
        n = len(unitDict)
        if ('','1') in unitDict:
            unitDict[('','1')] = 1
            if n > 1:
                otherUpper = False                
                for key, exp in unitDict.items():
                    if key == ('','1'): continue
                    if exp > 0:
                        otherUpper = True
                        break

                # if there are any other upper units, then remove 1
                # else set the exponent of the unit '1' to 1
                if otherUpper:
                    unitDict.pop(('','1'))
                    n -= 1

        # make temperature units in to temperature differences, if there are any other units in the dict
        if n > 1:
            keysToChange = []
            for key in unitDict:
                if key[1] in _temperature:
                    keysToChange.append(key)
                    
            for (p,u) in keysToChange:
                
                oldKey = (p, u)
                newKey = (p, 'DELTA' + u)
                if newKey in unitDict:
                    unitDict[newKey] += unitDict[oldKey]
                else:
                    unitDict[newKey] = unitDict[oldKey] 
                unitDict.pop(oldKey)

        # loop over all units, and remove any prefixes, which has an exponenet of 0
        keysToRemove = [key for key, exp in unitDict.items() if exp == 0]
        n -= len(keysToRemove)

        # if all the keys in unitDict has to be removed, then return the unit '1'
        if n == 0:
            # return '1' if there are not other units
            return {('','1'):1}

        # remove the keys
        for key in keysToRemove:
            unitDict.pop(key)

        if not ('','1') in unitDict:
            hasUpper = False
            for exp in unitDict.values():
                if exp > 0:
                    hasUpper = True
                    break
            if not hasUpper:
                unitDict[('','1')] = 1

        return unitDict

    @staticmethod
    def _getUnitDictSI(unitDict):
        out = {}
        for key, exp in unitDict.items():
            unitSI = _knownUnits[key[1]][0]
            for kkey, e in unitSI.items():
                e = e * exp
                if kkey in out:
                    out[kkey] += e
                else:
                    out[kkey] = e
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
                e = - e
            key = (p,u)
            if not key in out:
                out[key] = e
            else:
                out[key] += e
        return unit._reduceDict(out)

    @staticmethod
    def _formatUnitStrAndUnitStrPretty(unitStr, unitStrPretty):

        # return the unity
        if unitStr is None or unitStr == '':
            return '1', '1'

        # find start parenthesis is the unit string
        startParenIndexes = [i for i, s in enumerate(unitStr) if s == '(']
        stopParenIndexes = [i for i, s in enumerate(unitStr) if s == ')']
        
        if not startParenIndexes and not stopParenIndexes:
            
            changeUnitStrPretty = unitStrPretty == unitStr
            
            upper,lower = unit._splitCompositeUnit(unitStr)
            if len(upper) + len(lower) > 1:
                out = []
                for elem in upper:
                    e, _ = unit._removeExponentFromUnit(elem)
                    if e in _temperature:
                        elem = 'DELTA' + elem
                    out.append(elem)

                upper = out
                
                out = []
                for elem in lower:
                    e, _ = unit._removeExponentFromUnit(elem)
                    if e in _temperature:
                        elem = 'DELTA' + elem
                    out.append(elem)

                lower = out
                
            unitStr = '-'.join(upper)
            if lower:
                unitStr += '/' + '-'.join(lower)
                
            if not unitStr: unitStr = '1'
            
            
            if changeUnitStrPretty:
                unitStrPrettyUpper = []
                for up in upper:
                    up, exp = unit._removeExponentFromUnit(up)
                    up = up.replace('DELTA', r'\Delta ')
                    up = up.replace('%', r'\%')
                    if exp != 1:
                        unitStrPrettyUpper.append(rf'{up}^{{{exp}}}')
                    else:
                        unitStrPrettyUpper.append(up)
                
                
                unitStrPrettyLower = []
                for low in lower:
                    low, exp = unit._removeExponentFromUnit(low)
                    low = low.replace('DELTA', r'\Delta ')
                    low = low.replace('%', r'\%')
                    if exp != 1:
                        unitStrPrettyLower.append(rf'{low}^{{{exp}}}')
                    else:
                        unitStrPrettyLower.append(low)
                
                unitStrPretty = r'\cdot '.join(unitStrPrettyUpper)           
                if unitStrPrettyLower:
                    unitStrPrettyLower = r'\cdot '.join(unitStrPrettyLower)
                    unitStrPretty = rf'\frac{{{unitStrPretty}}}{{{unitStrPrettyLower}}}'
            
            return unitStr, unitStrPretty
        

        # check that the number of start and stop parenthesis are equal
        if len(startParenIndexes) != len(stopParenIndexes):
            raise ValueError(
                'The unit string has to have an equal number of open parenthesis and close parenthesis')

        # chekc if the entire unit is encapsuled by a single parenthesis
        ## this only works, if there is only one set of parenthesis
        if len(startParenIndexes) == 1 and startParenIndexes[0] == 0 and stopParenIndexes[0] == len(unitStr) - 1:
            return unit._formatUnitStrAndUnitStrPretty(
                unitStr[startParenIndexes[0]+1:stopParenIndexes[0]],
                unitStrPretty[startParenIndexes[0]+1:stopParenIndexes[0]]
                )

        # find all parenthesis pairs
        allIndexes = startParenIndexes + stopParenIndexes
        allIndexes.sort()
        isStartParen = [elem in startParenIndexes for elem in allIndexes]
        done, iter, maxIter, parenOrder = False, 0, len(allIndexes), []
        while not done:
            for i in range(len(allIndexes)-1):
                if isStartParen[i] and (isStartParen[i] + isStartParen[i+1] == 1):
                    parenOrder.append([allIndexes[i], allIndexes[i+1]])
                    allIndexes.pop(i+1)
                    allIndexes.pop(i)
                    isStartParen.pop(i+1)
                    isStartParen.pop(i)
                    break

            if len(allIndexes) == 0:
                break

            iter += 1
            if iter > maxIter:
                raise ValueError(
                    'An order to evaluate the parenthesis could not be found')

        ## check if the last parenthesis pair encapusles the entire unit
        if (parenOrder[-1][0] == 0 and parenOrder[-1][1] == len(unitStr) - 1):
            return unit._formatUnitStrAndUnitStrPretty(
                unitStr[parenOrder[-1][0]+1:parenOrder[-1][1]],
                unitStrPretty[parenOrder[-1][0]+1:parenOrder[-1][1]],
                )


        ## determine if there is a hyphen or a slash outside of any parenthesis.
        parenLevel = 0    
        for i, s in enumerate(unitStr):
            if s == '(': parenLevel +=1
            elif s == ')': parenLevel -=1
            
            if parenLevel == 0:
                if s in ['/', '-']:
                    a = unit._getUnitDict(unit._formatUnitStr(unitStr[0:i]))
                    b = unit._getUnitDict(unit._formatUnitStr(unitStr[i+1:]))
                    
                    aPretty = unit._formatUnitStrPretty(unitStrPretty[0:i])
                    bPretty = unit._formatUnitStrPretty(unitStrPretty[i+1:])
                    
                    if s == '/':
                        out = unit.staticTruediv(a, b)     
                        unitStrPretty = rf'\frac{{{aPretty}}}{{{bPretty}}}'    
                    else:
                        out = unit.staticMul(a,b)
                        unitStrPretty = rf'{aPretty} \cdot {bPretty}'
                    
                    unitStr = unit._getUnitStrFromDict(out)
                    
                    return unitStr, unitStrPretty
                
        # # there were no slashes outside of the parenthesis
        ## select the outer most parenthesis to work with
        ## select the enture unit and the next parenthesis
        ## this allows to find exponents of the outer parenthesis
        currentParens = parenOrder[-1]
        nextParens = [0, len(unitStr)]

        # get the unitDict from the stringwithin the current parentheses
        _unitStr = unitStr[currentParens[0]+1:currentParens[1]]
        _unitStr = unit._formatUnitStr(_unitStr)
        _unitDict = unit._getUnitDict(_unitStr)
        _unitStrPretty = unitStrPretty[currentParens[0]+1 : currentParens[1]]
        _unitStrPretty = unit._formatUnitStrPretty(_unitStrPretty)

        # determine if the nextparenthesis encompasses the current parenthesis
        if nextParens[0] <= currentParens[0] and nextParens[1] >= currentParens[1]:

            # get the string from the end of the current parenthesis to the end of the next parenthesis
            nextBit = unitStr[currentParens[1]+1:nextParens[1]]

            # A potential exponent of the current parenthis has to be
            # above a potential slash (/) and before any potential hyphens (-)
            exponent = nextBit.split('/')[0].split('-')[0]

            # try to cast the exponent to an integer
            # if this works, then raise the unitDict to the exponent
            try:
                exponent = float(exponent)
                if exponent.is_integer():
                    exponent = int(exponent)
                    
                _unitDict = unit.staticPow(_unitDict, None, exponent)
                _unitStr = unit._getUnitStrFromDict(_unitDict) 
                
                if exponent != 1:
                    _unitStrPretty = rf'\left( {_unitStrPretty} \right)^{{{exponent}}}'
                               
                currentParens[1] += len(str(exponent))
            except ValueError:
                pass
        
        return unit._formatUnitStrAndUnitStrPretty(
            unitStr[0:currentParens[0]] + _unitStr + unitStr[currentParens[1]+1:],
            unitStr[0:currentParens[0]] + _unitStrPretty + unitStr[currentParens[1]+1:]
            )

    @staticmethod
    def _formatUnitStr(unitStr):

        # return the unity
        if unitStr is None or unitStr == '':
            return '1'

        # find start parenthesis is the unit string
        startParenIndexes = [i for i, s in enumerate(unitStr) if s == '(']
        stopParenIndexes = [i for i, s in enumerate(unitStr) if s == ')']
        
        if not startParenIndexes and not stopParenIndexes:

            upper,lower = unit._splitCompositeUnit(unitStr)
            if len(upper) + len(lower) > 1:
                out = []
                for elem in upper:
                    e, _ = unit._removeExponentFromUnit(elem)
                    if e in _temperature:
                        out.append('DELTA' + elem)
                    else:
                        out.append(elem)
                upper = out
                
                out = []
                for elem in lower:
                    e, _ = unit._removeExponentFromUnit(elem)
                    if e in _temperature:
                        out.append('DELTA' + elem)
                    else:
                        out.append(elem)
                lower = out
                
            unitStr = '-'.join(upper)
            if lower:
                unitStr += '/' + '-'.join(lower)
            if not unitStr: return '1'
            return unitStr
        

        # check that the number of start and stop parenthesis are equal
        if len(startParenIndexes) != len(stopParenIndexes):
            raise ValueError(
                'The unit string has to have an equal number of open parenthesis and close parenthesis')

        # chekc if the entire unit is encapsuled by a single parenthesis
        ## this only works, if there is only one set of parenthesis
        if len(startParenIndexes) == 1 and startParenIndexes[0] == 0 and stopParenIndexes[0] == len(unitStr) - 1:
            return unit._formatUnitStr(unitStr[startParenIndexes[0]+1:stopParenIndexes[0]])

        # find all parenthesis pairs
        allIndexes = startParenIndexes + stopParenIndexes
        allIndexes.sort()
        isStartParen = [elem in startParenIndexes for elem in allIndexes]
        done, iter, maxIter, parenOrder = False, 0, len(allIndexes), []
        while not done:
            for i in range(len(allIndexes)-1):
                if isStartParen[i] and (isStartParen[i] + isStartParen[i+1] == 1):
                    parenOrder.append([allIndexes[i], allIndexes[i+1]])
                    allIndexes.pop(i+1)
                    allIndexes.pop(i)
                    isStartParen.pop(i+1)
                    isStartParen.pop(i)
                    break

            if len(allIndexes) == 0:
                break

            iter += 1
            if iter > maxIter:
                raise ValueError(
                    'An order to evaluate the parenthesis could not be found')

        ## check if the last parenthesis pair encapusles the entire unit
        if (parenOrder[-1][0] == 0 and parenOrder[-1][1] == len(unitStr) - 1):
            return unit._formatUnitStr(unitStr[parenOrder[-1][0]+1:parenOrder[-1][1]])


        ## determine if there is a hyphen or a slash outside of any parenthesis.
        parenLevel = 0    
        for i, s in enumerate(unitStr):
            if s == '(': parenLevel +=1
            elif s == ')': parenLevel -=1
            
            if parenLevel == 0:
                if s in ['/', '-']:
                    a = unit._formatUnitStr(unitStr[0:i])
                    b = unit._formatUnitStr(unitStr[i+1:])
                    
                    a = unit._getUnitDict(a)
                    b = unit._getUnitDict(b)
                    
                    if s == '/':
                        out = unit.staticTruediv(a, b)     
                    else:
                        out = unit.staticMul(a,b)
                    
                    out = unit._getUnitStrFromDict(out)
                        
                    return out                  
                
        # # there were no slashes outside of the parenthesis
        ## select the outer most parenthesis to work with
        ## select the enture unit and the next parenthesis
        ## this allows to find exponents of the outer parenthesis
        currentParens = parenOrder[-1]
        nextParens = [0, len(unitStr)]

        # get the unitDict from the stringwithin the current parentheses
        _unitStr = unitStr[currentParens[0]+1:currentParens[1]]
        _unitStr = unit._formatUnitStr(_unitStr)
        _unitDict = unit._getUnitDict(_unitStr)

        # determine if the nextparenthesis encompasses the current parenthesis
        if nextParens[0] <= currentParens[0] and nextParens[1] >= currentParens[1]:

            # get the string from the end of the current parenthesis to the end of the next parenthesis
            nextBit = unitStr[currentParens[1]+1:nextParens[1]]

            # A potential exponent of the current parenthis has to be
            # above a potential slash (/) and before any potential hyphens (-)
            exponent = nextBit.split('/')[0].split('-')[0]

            # try to cast the exponent to an integer
            # if this works, then raise the unitDict to the exponent
            try:
                exponent = float(exponent)
                if exponent.is_integer():
                    exponent = int(exponent)
                _unitDict = unit.staticPow(
                    _unitDict, None, exponent)
                _unitStr = unit._getUnitStrFromDict(_unitDict)                
                currentParens[1] += len(str(exponent))
            except ValueError:
                pass
        
        # update unitStr
        unitStr = unitStr[0:currentParens[0]] + _unitStr + unitStr[currentParens[1]+1:]
        return unit._formatUnitStr(unitStr)

    @staticmethod
    def _formatUnitStrPretty(unitStrPretty):

        # return the unity
        if unitStrPretty is None or unitStrPretty == '':
            return '1'

        # find start parenthesis is the unit string
        startParenIndexesPretty = [i for i, s in enumerate(unitStrPretty) if s == '(']
        stopParenIndexesPretty = [i for i, s in enumerate(unitStrPretty) if s == ')']

        
        if not startParenIndexesPretty and not stopParenIndexesPretty:
                    
            upper,lower = unit._splitCompositeUnit(unitStrPretty)
            
            if len(upper) + len(lower) > 1:
                out = []
                for elem in upper:
                    e, _ = unit._removeExponentFromUnit(elem)
                    if e in _temperature:
                        elem = 'DELTA' + elem
                    out.append(elem)

                upper = out
                
                out = []
                for elem in lower:
                    e, _ = unit._removeExponentFromUnit(elem)
                    if e in _temperature:
                        elem = 'DELTA' + elem
                    out.append(elem)

                lower = out
              
            
            out = []
            for up in upper:
                up, exp = unit._removeExponentFromUnit(up)
                up = up.replace('DELTA', r'\Delta ')
                up = up.replace('%', r'\%')
                if exp != 1:
                    out.append(rf'{up}^{{{exp}}}')
                else:
                    out.append(up)
            
            upper = r'\cdot '.join(out)
            
            out = []
            for low in lower:
                low, exp = unit._removeExponentFromUnit(low)
                low = low.replace('DELTA', r'\Delta ')
                low = low.replace('%', r'\%')
                if exp != 1:
                    out.append(rf'{low}^{{{exp}}}')
                else:
                    out.append(low)
            
            lower = r'\cdot '.join(out)
            
            if lower:
                unitStrPretty = rf'\frac{{{upper}}}{{{lower}}}'
            else:
                unitStrPretty = upper
            
            return unitStrPretty
        

        # check that the number of start and stop parenthesis are equal
        if len(startParenIndexesPretty) != len(stopParenIndexesPretty):
            raise ValueError(
                'The unit string has to have an equal number of open parenthesis and close parenthesis')

        # chekc if the entire unit is encapsuled by a single parenthesis
        ## this only works, if there is only one set of parenthesis
        if len(startParenIndexesPretty) == 1 and startParenIndexesPretty[0] == 0 and stopParenIndexesPretty[0] == len(unitStrPretty) - 1:
            return unit._formatUnitStrPretty(unitStrPretty[startParenIndexesPretty[0]+1 : stopParenIndexesPretty[0]])

        # find all parenthesis pairs
        allIndexesPretty = startParenIndexesPretty + stopParenIndexesPretty
        allIndexesPretty.sort()
        isStartParen = [elem in startParenIndexesPretty for elem in allIndexesPretty]
        done, iter, maxIter, parenOrderPretty = False, 0, len(allIndexesPretty), []
        while not done:
            for i in range(len(allIndexesPretty)-1):
                if isStartParen[i] and (isStartParen[i] + isStartParen[i+1] == 1):
                    parenOrderPretty.append([allIndexesPretty[i], allIndexesPretty[i+1]])
                    isStartParen.pop(i+1)
                    isStartParen.pop(i)
                    allIndexesPretty.pop(i+1)
                    allIndexesPretty.pop(i)
                    break

            if len(allIndexesPretty) == 0:
                break

            iter += 1
            if iter > maxIter:
                raise ValueError(
                    'An order to evaluate the parenthesis could not be found')

        ## check if the last parenthesis pair encapusles the entire unit
        if (parenOrderPretty[-1][0] == 0 and parenOrderPretty[-1][1] == len(unitStrPretty) - 1):
            return unit._formatUnitStrPretty(unitStrPretty[parenOrderPretty[-1][0]+1 : parenOrderPretty[-1][1]])


        ## determine if there is a hyphen or a slash outside of any parenthesis.
        parenLevel = 0    
        for i, s in enumerate(unitStrPretty):
            if s == '(': parenLevel +=1
            elif s == ')': parenLevel -=1
            
            if parenLevel == 0:
                if s in ['/', '-']:
                    aPretty = unit._formatUnitStrPretty(unitStrPretty[0:i])
                    bPretty = unit._formatUnitStrPretty(unitStrPretty[i+1:])
                    
                    if s == '/':
                        outPretty = rf'\frac{{{aPretty}}}{{{bPretty}}}'    
                    else:
                        outPretty = rf'{aPretty} \cdot {bPretty}'
                        
                    return outPretty                  
                
        # # there were no slashes outside of the parenthesis
        ## select the outer most parenthesis to work with
        ## select the enture unit and the next parenthesis
        ## this allows to find exponents of the outer parenthesis
        currentParensPretty = parenOrderPretty[-1]
        nextParensPretty = [0, len(unitStrPretty)]

        # get the unitDict from the stringwithin the current parentheses
        _unitStrPretty = unitStrPretty[currentParensPretty[0]+1 : currentParensPretty[1]]
        _unitStrPretty = unit._formatUnitStrPretty(_unitStrPretty)

        # determine if the nextparenthesis encompasses the current parenthesis
        if nextParensPretty[0] <= currentParensPretty[0] and nextParensPretty[1] >= currentParensPretty[1]:

            # get the string from the end of the current parenthesis to the end of the next parenthesis
            nextBit = unitStrPretty[currentParensPretty[1]+1:nextParensPretty[1]]

            # A potential exponent of the current parenthis has to be
            # above a potential slash (/) and before any potential hyphens (-)
            exponent = nextBit.split('/')[0].split('-')[0]

            # try to cast the exponent to an integer
            # if this works, then raise the unitDict to the exponent
            try:
                exponent = float(exponent)
                if exponent.is_integer():
                    exponent = int(exponent)
                    
                if exponent != 1:
                    _unitStrPretty = rf'\left( {_unitStrPretty} \right)^{{{exponent}}}'
                
                currentParensPretty[1] += len(str(exponent))
            except ValueError:
                pass
            except UnboundLocalError:
                pass
        
        # update unitStr
        return unitStrPretty[0:currentParensPretty[0]] + _unitStrPretty + unitStrPretty[currentParensPretty[1]+1:]
        # return unit._formatUnitStrPretty(unitStrPretty)

    @ staticmethod
    def _removeExponentFromUnit(u):

        integerIndexes = [i for i, char in enumerate(u) if char in integers or char == '.']

        if not integerIndexes:
            return u, 1

        for i in range(len(integerIndexes)-1):
            if integerIndexes[i+1] - integerIndexes[i] != 1:
                raise ValueError(
                    'All numbers in the unit has to be grouped together')

        # Determine if the last integer is placed at the end of the unit
        if integerIndexes[-1] != len(u) - 1:
            raise ValueError(
                'Any number has to be placed at the end of the unit')

        exponent = u[integerIndexes[0]:]
        exponent = float(exponent) if '.' in exponent else int(exponent)
        u = u[:integerIndexes[0]]
        
        
        # Ensure that the entire use was not removed by removing the integers
        if not u:
            # No symbols are left after removing the integers
            if exponent != 1:
                raise ValueError(
                    f'The unit {u} was stripped of all integers which left no symbols in the unit. This is normally due to the integers removed being equal to 1, as the unit is THE unit. Howver, the intergers removed was not equal to 1. The unit is therefore not known.')
            u = '1'
        
        return u, exponent

    @staticmethod
    def _removePrefixFromUnit(unit):

        if unit in _knownUnits:
            return unit, ''

        # The unit was not found. This must be because the unit has a prefix
        found = False

        for p in _knownPrefixes:
            if p != unit[0:len(p)]:
                continue
            u = unit[len(p):]
            if not u in _knownUnits:
                continue
            found = True
            prefix, unit = p, u
            break

        if not found:
            raise ValueError(
                f'The unit ({unit}) was not found. Therefore it was interpreted as a prefix and a unit. However a combination of prefix and unit which matches {unit} was not found')

        if unit in _baseUnit and unit != "%":
            unit = "1"
            raise ValueError(
                f'The unit ({prefix}) was not found. Therefore it was interpreted as a prefix and a unit. The prefix was identified as "{p}" and the unit was identified as "{unit}". However, the unit "1" cannot have a prefix')

        # look for the unit without the prefix
        if not unit in _knownUnits:
            raise ValueError(
                f'The unit ({prefix}{unit}) was not found. Therefore it was interpreted as a prefix and a unit. However the unit ({unit}) was not found')
        return unit, prefix

    def __str__(self, pretty=False):

        if not pretty:
            return self.unitStr
        
        return self.unitStrPretty
        
    def __eq__(self, other):
        return self.unitDict == other.unitDict

    def isLogarithmicUnit(self):
        upper, lower = unit._splitCompositeUnit(self.unitStr)
        if lower or len(upper) != 1:
            return False
        return unit._splitUnitExponentAndPrefix(upper[0])[0] in _logrithmicUnits

    def __add__(self, other):
        
        # determine if self is the same as other - then no conversions are necessary
        if self.unitDict == other.unitDict:
            return self.unitStr, other.unitStr, self
        
        # determine the units of self without any prefixes
        selfUnitDictWithoutPrefixes = {}
        for (_,u), exp in self.unitDict.items():
            key = ('',u)
            if u in selfUnitDictWithoutPrefixes:
                selfUnitDictWithoutPrefixes[key] += exp
            else:
                selfUnitDictWithoutPrefixes[key] = exp

        # determine the units of other without any prefixes and convert
        otherUnitDictWithoutPrefixes = {}
        for (_,u), exp in other.unitDict.items():
            key = ('',u)
            if u in otherUnitDictWithoutPrefixes:
                otherUnitDictWithoutPrefixes[key] += exp
            else:
                otherUnitDictWithoutPrefixes[key] = exp
          
        # determine if the self and other are identical once any prefixes has been removed
        if selfUnitDictWithoutPrefixes == otherUnitDictWithoutPrefixes:
            out = unit(unitDict=selfUnitDictWithoutPrefixes, unitDictSI=self.unitDictSI, unitStrSI=self.unitStrSI)
            return out.unitStr, out.unitStr, out

        # determine if the SI base units of self and other are equal
        if self.unitDictSI == other.unitDictSI:
            return self.unitStrSI, self.unitStrSI, unit(unitDict=self.unitDictSI, unitDictSI=self.unitDictSI, unitStr=self.unitStrSI, unitStrSI=self.unitStrSI, converterToSI=_unitConversion(1, 0))

        # determine if "DELTAK" and "K" are the SI Baseunits of self and other
        SIBaseUnits = [self.unitDictSI, other.unitDictSI]
        if {('','DELTAK') : 1} in SIBaseUnits and {('','K') : 1} in SIBaseUnits:

            indexTemp = SIBaseUnits.index({('','K') : 1})
            indexDiff = 0 if indexTemp == 1 else 1
            
            units = [self.unitStr, other.unitStr]
                        
            # check to see if the temperature differnce has the same unit as the temperature
            if units[indexTemp] == units[indexDiff][-1]:
                return self.unitStr, other.unitStr, [self, other][indexTemp]

            out = unit(unitDict={('','K') : 1}, unitDictSI={('','K') : 1}, unitStr='K', unitStrSI='K', converterToSI=_unitConversion(1, 0))
            if indexTemp == 0:
                return 'K', 'DELTAK', out
            return 'DELTAK', 'K', out

        raise ValueError(
            f'You tried to add a variable in [{self}] to a variable in [{other}], but the units do not have the same SI base unit')

    def __sub__(self, other):

        # determine if "K" and "K" are the SI Baseunits of self and other
        SIBaseUnits = [self.unitDictSI, other.unitDictSI]
        if SIBaseUnits[0] == {('','K') : 1} and SIBaseUnits[1] == {('','K') : 1}:
            if self.unitDict == other.unitDict:
                return self.unitStr, other.unitStr, unit(unitStr = 'DELTA' + self.unitStr, unitStrPretty=rf'\Delta {self.unitStr}')
            return 'K', 'K', unit(unitDict={('','DELTAK') : 1}, unitDictSI={('','DELTAK') : 1}, unitStr='DELTAK', unitStrSI='DELTAK', converterToSI=_unitConversion(1, 0))

        if {('','DELTAK') : 1} in SIBaseUnits and {('','K') : 1} in SIBaseUnits:
            indexTemp = SIBaseUnits.index({('','K') : 1})
            if indexTemp != 0:
                raise ValueError('You tried to subtract a temperature from a temperature differnce. This is not possible.')
            
            temp = list(self.unitDict.keys())[0][1]
            tempDiff = list(other.unitDict.keys())[0][1]
            
            if temp == tempDiff[5:]:
                return self.unitStr, other.unitStr, self
            
            return 'K', 'DELTAK', unit(unitDict={('','K') : 1}, unitDictSI={('','K') : 1}, unitStr='K', unitStrSI='K', converterToSI=_unitConversion(1, 0))

        # determine if self is the same as other - then no conversions are necessary
        if self.unitDict == other.unitDict:
            return self.unitStr, other.unitStr, self

        # determine the units of self without any prefixes
        selfUnitDictWithoutPrefixes = {}
        for (_,u), exp in self.unitDict.items():
            key = ('',u)
            if u in selfUnitDictWithoutPrefixes:
                selfUnitDictWithoutPrefixes[key] += exp
            else:
                selfUnitDictWithoutPrefixes[key] = exp

        # determine the units of other without any prefixes and convert
        otherUnitDictWithoutPrefixes = {}
        for (_,u), exp in other.unitDict.items():
            key = ('',u)
            if u in otherUnitDictWithoutPrefixes:
                otherUnitDictWithoutPrefixes[key] += exp
            else:
                otherUnitDictWithoutPrefixes[key] = exp
            
        # determine if the self and other are identical once any prefixes has been removed
        if selfUnitDictWithoutPrefixes == otherUnitDictWithoutPrefixes:
            out = unit(unitDict=selfUnitDictWithoutPrefixes, unitDictSI=self.unitDictSI, unitStrSI=self.unitStrSI)
            return out.unitStr, out.unitStr, out

        # determine if the SI base units of self and other are equal
        if self.unitDictSI == other.unitDictSI:
            return self.unitStrSI, self.unitStrSI, unit(unitStr=self.unitStrSI, unitDict=self.unitDictSI, unitDictSI=self.unitDictSI, unitStrSI=self.unitStrSI, converterToSI=_unitConversion(1, 0))

        raise ValueError(
            f'You tried to subtract a variable in [{other}] from a variable in [{self}], but the units do not have the same SI base unit')

    def __mul__(self, other):
        unitDict = unit.staticMul(self.unitDict, other.unitDict)
        converterToSI = _unitConversion(*_unitConversion.staticMul(self._converterToSI.scale,
                                        self._converterToSI.offset, other._converterToSI.scale, other._converterToSI.offset))
        unitStrPretty = rf'{{{self.unitStrPretty}}} \cdot {{{other.unitStrPretty}}}'
        unitStr=f'({self.unitStr})-({other.unitStr})'
        return unit(unitStr=unitStr, unitDict=unitDict, converterToSI=converterToSI, unitStrPretty = unitStrPretty)
        
    @staticmethod
    def staticMul(a,b):
        unitDict = {}
        for key, exp in a.items():
            unitDict[key] = exp

        for key, exp in b.items():
            if key in unitDict:
                unitDict[key] += exp
            else:
                unitDict[key] = exp
            
        unitDict = unit._reduceDict(unitDict)
        
        return unitDict

    def isConvertableToDimensionless(self):
        unitDict = unit.__staticConvertToDimensionless(self.unitDict)
        if unitDict != self.unitDict:
            return True, unit._getUnitStrFromDict(unitDict)
        return False, None
            
    
    @staticmethod
    def __staticConvertToDimensionless(unitDict):

        unitDictWithoutPrefix = {}
        for (_,key), exp in unitDict.items():
            key = _knownUnits[key][0]
            for (_,k), e in key.items():

                if k in unitDictWithoutPrefix:
                    unitDictWithoutPrefix[k] += exp * e
                else:
                    unitDictWithoutPrefix[k] = exp * e
        
        keys = [key for key, exp in unitDictWithoutPrefix.items() if exp == 0]
        if not keys:
            return unitDict
        
        keysToRemove = []
        for key in unitDictWithoutPrefix.keys():
            if key in keys:
                keysToRemove.append(key)
        
        for key in keysToRemove:
            unitDictWithoutPrefix.pop(key)

        if len(unitDictWithoutPrefix) == 0:
            return {('','1'):1}
        

    
        return unitDict

    def __rtruediv__(self, other):
        
        other = unit(str(other))
        return other / self
        
    def __truediv__(self, other):
        unitDict = unit.staticTruediv(self.unitDict, other.unitDict)
        converterToSI = _unitConversion(*_unitConversion.staticTruediv(self._converterToSI.scale,
                                        self._converterToSI.offset, other._converterToSI.scale, other._converterToSI.offset))
        unitStrPretty = rf'\frac{{{self.unitStrPretty}}}{{{other.unitStrPretty}}}'
        unitStr=f'({self.unitStr})/({other.unitStr})'
        return unit(unitStr = unitStr, unitDict=unitDict, converterToSI=converterToSI, unitStrPretty=unitStrPretty)

    @staticmethod
    def staticTruediv(a, b):
        unitDict = {}
        for key, exp in a.items():
            unitDict[key] = exp

        for key, exp in b.items():
            if key in unitDict:
                unitDict[key] -= exp
            else:
                unitDict[key] = -exp
            
        unitDict = unit._reduceDict(unitDict)
        return unitDict

    def __pow__(self, power):

        frac = Fraction(power).limit_denominator()
        num, den = frac._numerator, frac._denominator
        
        unitDict = unit.staticPow(self.unitDict, self.unitDictSI, power)
        
        if power != 1:
            if den == 1:
                unitStrPretty = rf'\left( {self.unitStrPretty} \right)^{{{num}}}'
            else:
                if num == 1:
                    if den == 2:
                        unitStrPretty = rf'\sqrt{{{self.unitStrPretty}}}'
                    else:
                        unitStrPretty = rf'\sqrt[{den}]{{{self.unitStrPretty}}}'
                else:
                    unitStrPretty = rf'\left( {self.unitStrPretty} \right)^{{\frac{{{num}}}{{{den}}}}}'
        else:
            unitStrPretty = self.unitStrPretty
        unitStr = f'({self.unitStr}){power}'
        converterToSI = _unitConversion(*_unitConversion.staticPow(self._converterToSI.scale, self._converterToSI.offset, power, num, den))
        return unit(
            unitStr=unitStr,
            unitStrPretty=unitStrPretty, 
            unitDict=unitDict,
            converterToSI=converterToSI
            )
    

    @staticmethod
    def staticPow(unitDict, unitDictSI, power):

        if unitDict == {('','1') : 1}:
            return unitDict
        
        out = {}
        for key, exp in unitDict.items():
            newExp = power * exp
            if hasattr(newExp, 'is_integer') and newExp.is_integer():
                newExp = int(newExp)
            out[key] = newExp
        out = unit._reduceDict(out)
        return out


    def getConverterToSI(self):
        # initialize the scale and offset
        outScale, outOffset = 1, 0

        # loop over self.unitDict
        for (pre, u), exp in self.unitDict.items():
            convScale, convOffset = _knownUnits[u][1]
            convScale = convScale * _knownPrefixes[pre]
            if convScale == 1 and convOffset == 0:
                continue
            convScale, convOffset = _unitConversion.staticPow(
                convScale, convOffset, exp)
            outScale, outOffset = _unitConversion.staticMul(
                outScale, outOffset, convScale, convOffset)

        self._converterToSI = _unitConversion(outScale, outOffset)

    def getConverter(self, newUnitStr):
         # remove spaces
        newUnitStr = newUnitStr.replace(' ', '')
            
        # check for any illegal symbols
        for s in newUnitStr:
            if s not in _knownCharacters:
                raise ValueError(
                    f'The character {s} is not used within the unitsystem')
                
        newUnitStr = self._formatUnitStr(newUnitStr)
        newUnitDict = unit._getUnitDict(newUnitStr)
        if newUnitDict == self.unitDictSI:
            return self._converterToSI.convert, {'unitStr': newUnitStr, 'unitDict': newUnitDict, 'unitStrSI': self.unitStrSI, 'unitDictSI': self.unitDictSI}

        newUnitDictSI = unit._getUnitDictSI(newUnitDict)
        if self.unitDictSI == newUnitDictSI:

            outScale, outOffset = self._converterToSI.scale, self._converterToSI.offset

            # loop over newUnitDict
            for (pre, u), exp in newUnitDict.items():
                
                convScale, convOffset = _knownUnits[u][1]
                convScale = convScale * _knownPrefixes[pre]
                if convScale == 1 and convOffset == 0:
                    continue
                convScale, convOffset = _unitConversion.staticPow(
                    convScale, convOffset, exp)
                outScale, outOffset = _unitConversion.staticTruediv(
                    outScale, outOffset, convScale, convOffset)

            return _unitConversion(outScale, outOffset).convert, {'unitStr': newUnitStr, 'unitDict': newUnitDict, 'unitDictSI': newUnitDictSI, 'unitStrSI': self.unitStrSI}

        if (self.unitDictSI == {('','Np') : 1} and newUnitDictSI == {('','1') : 1}):
            # convert from logarithmic unit to signal
            converter = self.getLogarithmicConverter(
                self._removePrefixFromUnit(self.unitStr)[0])
            converter.linearConverter, _ = self.getConverter(
                self.getUnitWithoutPrefix())
            return converter.converterToSignal, {'unitStr': newUnitStr}

        if (self.unitDictSI == {('','1') : 1} and newUnitDictSI == {('','Np') : 1}):
            # convert from signal to logarithmic unit
            newUnit = unit(newUnitStr)
            newUnitWithoutPrefix = unit(newUnit.getUnitWithoutPrefix())
            converter = self.getLogarithmicConverter(newUnit.getUnitWithoutPrefix())
            converter.linearConverter, _ = newUnitWithoutPrefix.getConverter(
                newUnitStr)
            return converter.converterFromSignal, {'unitStr': newUnitStr}

        raise ValueError(
            f'You tried to convert from {self} to {newUnitStr}. But these do not have the same base units')

    def isCombinationUnit(self):
        return len(list(self.unitDict.keys())) > 1

    @staticmethod
    def getLogarithmicConverter(u):

        if u == 'B':
            return _bellConversion()
        if u == 'Np':
            return _neperConversion()
        if u == 'oct':
            return _octaveConversion()
        if u == 'dec':
            return _bellConversion()

        raise ValueError(f'The logarithmic conversion of {u} is not knwon')


