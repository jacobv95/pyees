import numpy as np
from fractions import Fraction
from copy import copy

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
        scale = scale ** pow
        offset = offset * sum([scale ** (pow - i) for i in range(pow)])
        return scale, offset
       
    @staticmethod
    def staticTruediv(selfScale, selfOffset, otherScale, otherOffset = 0):
        return _unitConversion.staticMul(1 / otherScale, - otherOffset / otherScale, selfScale, selfOffset)

    def convert(self, value, useOffset=True):
        return self.scale * value + useOffset * self.offset

class neperConversion():
            
    @staticmethod
    def convertToSignal(var):
        var._uncert = 2*np.exp(2*var.value) * var.uncert
        var._value = np.exp(2*var.value)
        
    @staticmethod
    def convertFromSignal(var):
        var._uncert = 1 / (2*var.value) * var.uncert
        var._value = 1/2 * np.log(var.value)

class bellConversion():
        
    @staticmethod
    def convertToSignal(var):
        var._uncert = 10**var.value * np.log(10) * var.uncert
        var._value = 10**var.value
        
    @staticmethod
    def convertFromSignal(var):
        var._uncert = 1 / (var.value * np.log(10)) * var.uncert
        var._value = np.log10(var.value)

class octaveConversion():
        
    @staticmethod
    def convertToSignal(var):
        var._uncert = 2**var.value * np.log(2) * var.uncert
        var._value = 2**var.value
        
    @staticmethod
    def convertFromSignal(var):
        var._uncert = 1 / (var.value * np.log(2)) * var.uncert
        var._value = np.log2(var.value)

    
    
baseUnit = {
    '1': (1,0),
    '': (1,0),
    '%': (1e-2,0)
}

force = {
    'N': (1,0)
}

mass = {
    'g': (1 / 1000,0)
}

energy = {
    'J': (1,0),
}

power = {
    'W': (1,0)
}

pressure = {
    'Pa': (1,0),
    'bar': (1e5,0)
}

temperature = {
    'K': (1,0),
    'C': (1, 273.15),
    'F': (5 / 9, 273.15 - 32 * 5 / 9)
}

temperatureDifference = {
    'DELTAK': (1,0),
    'DELTAC': (1,0),
    'DELTAF': (5 / 9,0)
}

time = {
    's': (1,0),
    'min': (60,0),
    'h': (60 * 60,0),
    'yr': (60 * 60 * 24 * 365,0)
}

volume = {
    'm3': (1,0),
    'L': (1 / 1000,0)
}

length = {
    'm': (1,0),
    'Å': (1e-10,0),
    'ly': (9460730472580800,0)
}

angle = {
    'rad': (1,0),
    'deg': (np.pi / 180,0)
}

current = {
    'A': (1,0)
}

voltage = {
    'V': (1,0)
}

frequency = {
    'Hz': (1,0)
}

resistance = {
    'ohm': (1,0)
}

kinematicViscosity = {
    'St': (1e-4,0)
}

logrithmicUnits = {
    'Np' : (1,0),
    'B': (1,0),
    'oct': (1,0),
    'dec': (1,0)
}

def siUnitForce():                      return {'g':{'k':1}, 'm':{'':1}, 's':{'':-2}} 
def siUnitPressure():                   return {'g':{'k':1}, 'm':{'':-1}, 's':{'':-2}}
def siUnitTime():                       return {'s':{'':1}}
def siUnitTemperature():                return {'K':{'':1}}
def siUnitVolume():                     return {'m':{'':3}}
def siUnitLength():                     return {'m':{'':1}}
def siUnitEnergy():                     return {'g':{'k':1}, 'm':{'':2}, 's':{'':-2}}
def siUnitPower():                      return {'g':{'k':1}, 'm':{'':2}, 's':{'':-3}}
def siUnitMass():                       return {'g':{'k':1}}
def siUnitCurrent():                    return {'A':{'':1}}
def siUnitVoltage():                    return {'g':{'k':1}, 'm':{'':2}, 's':{'':-3}, 'A':{'':-1}}
def siUnitBaseUnit():                   return {'1':{'':1}}
def siUnitFreqeuncy():                  return {'s':{'':-1}}
def siUnitAngle():                      return {'rad':{'':1}}
def siUnitResistance():                 return {'g':{'k':1}, 'm':{'':2}, 's':{'':-3}, 'A':{'':-2}}
def siUnitKinematicViscosity():         return {'m':{'':2},'s':{'':-1}}
def siUnitLogarithmicUnit():            return {'Np':{'':1}}
def siUnitTemperatureDifference():      return {'DELTAK':{'':1}}

knownUnitsDict = {
    siUnitForce:                        force,
    siUnitPressure:                     pressure,
    siUnitTime:                         time,
    siUnitTemperature:                  temperature,
    siUnitVolume:                       volume,
    siUnitLength:                       length,
    siUnitEnergy :                      energy,
    siUnitPower :                       power,
    siUnitMass:                         mass,
    siUnitCurrent:                      current,
    siUnitVoltage :                     voltage,
    siUnitBaseUnit:                     baseUnit,
    siUnitFreqeuncy:                    frequency,
    siUnitAngle:                        angle,
    siUnitResistance:                   resistance,
    siUnitKinematicViscosity:           kinematicViscosity,
    siUnitLogarithmicUnit:              logrithmicUnits,
    siUnitTemperatureDifference:        temperatureDifference
}

knownPrefixes = {
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
        for p in knownPrefixes:
            if p == '':continue
            
            unitPrefixCombinations.append(p+u)

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
        if p == '': continue
        if p+newUnit in unitPrefixCombinations:
            raise ValueError(f'The unit {p+newUnit} is already known within the unit system')
        unitPrefixCombinations.append(newUnit)

    
    existingUnitDict = unit._getUnitDict(existingUnit)
    existingUnitDictSI = unit._getUnitDictSI(existingUnitDict)
    def existingUnitDictSIMethod(): return existingUnitDictSI

    
    for u, item in existingUnitDict.items():
        for pre, exp in item.items():
            convScale, convOffset = knownUnits[u][1]
            convScale = convScale * knownPrefixes[pre]
            isUpper = exp > 0
            if not isUpper: exp = -exp 
            convScale, convOffset = _unitConversion.staticPow(convScale, convOffset, exp)
            if isUpper:
                scale, offset = _unitConversion.staticMul(convScale, convOffset, scale, offset)
            else:
                scale, offset = _unitConversion.staticTruediv(scale, offset, convScale, convOffset)      
    
    knownUnits[newUnit] = [existingUnitDictSIMethod, (scale, offset)]
        
    global knownCharacters
    for s in newUnit:
        knownCharacters.add(s)

    
hyphen = '-'
slash = '/'
integers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class unit():

    def __init__(self, unitStr = None, unitDict = None, unitDictSI = None):
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
            
        self.unitStr = self._getUnitStrFromDict(self.unitDict)
        self.unitStrSI = self._getUnitStrFromDict(self.unitDictSI)
        self.getConverterToSI()

    @staticmethod
    def _getUnitStrFromDict(unitDict):
        upper, lower = [], []
        for u, item in unitDict.items():    
            for p, exp in item.items():
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
                if exp == 0:  prefixesToRemove.append(pre)
            for pre in prefixesToRemove: unitDict[key].pop(pre)
    
        ## remove the units, which has no items in their dictionary
        keysToRemove = []
        for key, item in unitDict.items():
            if not item: keysToRemove.append(key)
        for key in keysToRemove: unitDict.pop(key)
        
        ## remove the unit '1' above the fraction line, if there are any other units above the fraction line
        if len(unitDict) > 1 and '1' in unitDict:
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
                
        ## set the exponent of the unit '1' to 1
        if '1' in unitDict:
            unitDict['1'] = {'':1}
            
        ## add the units '1' if there are not other units
        if not unitDict:
            unitDict = {'1': {'': 1}}
        
        ## make temperature units in to temperature differences, if there are any other units in the dict
        if len(unitDict) > 1:
            keysToChange = []
            for key in unitDict.keys():
                if key in temperature: keysToChange.append(key)
            for key in keysToChange: 
                unitDict['DELTA' + key] = unitDict[key]
                unitDict.pop(key)       
            
        return unitDict
    
    @staticmethod
    def _getUnitDictSI(unitDict):
        out = {}
        for key, item in unitDict.items():
            exp = sum(item.values())
            unitSI = knownUnits[key][0]()
            for kkey, iitem in unitSI.items():
                if kkey in out:
                    for p,e in iitem.items():
                        e = e * exp
                        if p in out[kkey]:
                            out[kkey][p] += e
                        else:
                            out[kkey][p] = e
                else:
                    out[kkey] = {}
                    for p,e in iitem.items():
                        out[kkey][p] = e * exp         
                    
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
            return unit, ''

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
            selfUnitDictWithoutPrefixes[key][''] = 0
            for exp in item.values():
                selfUnitDictWithoutPrefixes[key][''] += exp
        
        ## determine if self is the same as other - then no conversions are necessary
        if self.unitDict == other.unitDict:
            return unit(unitDict=self.unitDict, unitDictSI=self.unitDictSI)  
        
        ## determine the units of other without any prefixes and convert this to a string
        otherUnitDictWithoutPrefixes = {}
        for key, item in other.unitDict.items():
            otherUnitDictWithoutPrefixes[key] = {}
            otherUnitDictWithoutPrefixes[key][''] = 0
            for exp in item.values():
                otherUnitDictWithoutPrefixes[key][''] += exp
        
        ## determine if the self and other are identical once any prefixes has been removed
        if selfUnitDictWithoutPrefixes == otherUnitDictWithoutPrefixes:
            return unit(unitDict=selfUnitDictWithoutPrefixes, unitDictSI=self.unitDictSI)
        
        # determine if the SI base units of self and other are equal
        if self.unitDictSI == other.unitDictSI:
            return unit(unitDict=self.unitDictSI, unitDictSI=self.unitDictSI)
        
        ## determine if "DELTAK" and "K" are the SI Baseunits of self and other
        SIBaseUnits = [self.unitDictSI, other.unitDictSI]        
        if {'DELTAK':{'':1}} in SIBaseUnits and {'K':{'':1}} in SIBaseUnits:
            
            indexTemp = SIBaseUnits.index({'K':{'':1}})
            indexDiff = 0 if indexTemp == 1 else 1
            
            units = [list(selfUnitDictWithoutPrefixes.keys())[0], list(otherUnitDictWithoutPrefixes.keys())[0]]

            if units[indexTemp] == units[indexDiff][-1]:        
                return unit(unitDict=[selfUnitDictWithoutPrefixes, otherUnitDictWithoutPrefixes][indexTemp], unitDictSI = SIBaseUnits[indexTemp])
            
            return unit(unitDict={'K':{'':1}}, unitDictSI={'K':{'':1}})

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
                return unit(unitDict = self.unitDict, unitDictSI=self.unitDictSI)
            return unit(unitDict={'K':{'':1}}, unitDictSI={'K':{'':1}})

        if {'DELTAK':{'':1}} in SIBaseUnits and {'K':{'':1}} in SIBaseUnits:
            indexTemp = SIBaseUnits.index({'K':{'':1}})
            if indexTemp != 0:
                raise ValueError('You tried to subtract a temperature from a temperature differnce. This is not possible.')      
            return unit(unitDict=[selfUnitDictWithoutPrefixes, otherUnitDictWithoutPrefixes][indexTemp], unitDictSI=SIBaseUnits[indexTemp])
        
                
        ## determine if self is the same as other - then no conversions are necessary
        if self.unitDict == other.unitDict:
            return unit(unitDict=self.unitDict, unitDictSI = self.unitDictSI)
        
        ## determine if the self and other are identical once any prefixes has been removed
        if selfUnitDictWithoutPrefixes == otherUnitDictWithoutPrefixes:
            return unit(unitDict=selfUnitDictWithoutPrefixes, unitDictSI=self.unitDictSI)
        
        # determine if the SI base units of self and other are equal
        if self.unitDictSI == other.unitDictSI:
            return unit(unitDict=self.unitDictSI, unitDictSI = self.unitDictSI)
        
        
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
                convScale, convOffset = knownUnits[u][1]
                isUpper = exp > 0
                if not isUpper: exp = -exp  
                convScale, convOffset = _unitConversion.staticPow(convScale * knownPrefixes[pre], convOffset, exp)
                if isUpper:
                    outScale, outOffset = _unitConversion.staticMul(outScale, outOffset, convScale, convOffset)
                else:
                    outScale, outOffset = _unitConversion.staticTruediv(outScale, outOffset, convScale, convOffset)      
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
                convScale, convOffset = knownUnits[u][1]
                isUpper = exp > 0
                if not isUpper: exp = -exp
                convScale, convOffset = _unitConversion.staticPow(convScale * knownPrefixes[pre], convOffset, exp)
                if not isUpper:
                    outScale, outOffset = _unitConversion.staticMul(outScale, outOffset, convScale, convOffset)
                else:
                    outScale, outOffset = _unitConversion.staticTruediv(outScale, outOffset, convScale, convOffset)      
            
        return _unitConversion(outScale, outOffset)

    def isCombinationUnit(self):
        return len(list(self.unitDict.keys())) > 1

    def getLogarithmicConverter(self):
        u, _ = self._removePrefixFromUnit(self.unitStr)
        
        if u == 'B':
            return bellConversion()
        if u == 'Np':
            return neperConversion()
        if u == 'oct':
            return octaveConversion()
        return bellConversion()

if __name__ == '__main__':

    addNewUnit('gnA', 9.81, 'm/s2')
    a = unit('gnA')
    converter = a.getConverter('m/s2')
    print(converter.convert(1), 9.81)
    # converter = unit('Rø').getConverter('C')
    # print(converter.convert(100), 176.190476190476190476)