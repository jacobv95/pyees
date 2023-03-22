import numpy as np


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
    'T': 1e12,
    'G': 1e9,
    'M': 1e6,
    'k': 1e3,
    'h': 1e2,
    'da': 1e1,
    'd': 1e-1,
    'c': 1e-2,
    'm': 1e-3,
    'mu': 1e-6,
    'n': 1e-9,
    'p': 1e-12
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
knownCharacters += '-/'
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


    upper, upperPrefix,upperExp, lower, lowerPrefix, lowerExp = unit._getLists(existingUnit)
    nUpper = len(upper)
    units = upper + lower
    exponents = upperExp + lowerExp
    prefixes = upperPrefix + lowerPrefix
    SIBaseUnitLists = [[],[]]
    
    for i, (u, exponent, prefix) in enumerate(zip(units, exponents, prefixes)):
                
        _SIBaseUnit, unitConversion = knownUnits[u]
        unitConversion =  unitConversion ** exponent

        if exponent != 1: _SIBaseUnit += str(exponent)
        
        if not prefix is None: unitConversion *= knownPrefixes[prefix]

        if i <= nUpper-1:
            conversion = unitConversion * conversion
            SIBaseUnitLists[0].append(_SIBaseUnit)
        else:
            conversion = _unitConversion(1) / unitConversion * conversion
            SIBaseUnitLists[1].append(_SIBaseUnit)
            
    SIBaseUnit = '-'.join(SIBaseUnitLists[0])
    if SIBaseUnitLists[1]: SIBaseUnit += '/' + '-'.join(SIBaseUnitLists[1])
    
    knownUnits[newUnit] = [SIBaseUnit, conversion]
        
    global knownCharacters
    for s in newUnit:
        knownCharacters.add(s)

    
hyphen = '-'
slash = '/'
integers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class unit():

    def __init__(self, unitStr) -> None:
        if unitStr == '':
            unitStr = '1'
        
        # remove any unknown characters
        unitStr = self._formatUnit(unitStr)

        # split the unit in upper and lower
        self.upper, self.upperPrefix, self.upperExp, self.lower, self.lowerPrefix, self.lowerExp = self._getLists(unitStr)

        # create the unit string
        self.unitStr = self._createUnitString()

        self._SIBaseUnit = self._getSIBaseUnit(self.upper, self.upperExp, self.lower, self.lowerExp)
        otherUpper, otherUpperPrefix, otherUpperExp, otherLower, otherLowerPrefix, otherLowerExp = self._getLists(self._SIBaseUnit)
        self._converterToSI = self._getConverter(otherUpper, otherUpperPrefix, otherUpperExp, otherLower, otherLowerPrefix, otherLowerExp)

    def _createUnitString(self):
        return self._combineUpperAndLower(self.upper, self.upperPrefix, self.upperExp, self.lower, self.lowerPrefix, self.lowerExp)

    @staticmethod
    def _cancleUnits(upper, upperPrefix, upperExp, lower, lowerPrefix, lowerExp):
        # cancle the units
        for indexUpper, up in enumerate(upper):
            if up in lower:
                indexLower = lower.index(up)

                # only cancle units if they have the same prefix
                if upperPrefix[indexUpper] == lowerPrefix[indexLower]:
                    expUpper = upperExp[indexUpper]
                    expLower = lowerExp[indexLower]

                    # set the unit to '1'
                    if expUpper == expLower:
                        upper[indexUpper] = '1'
                        lower[indexLower] = '1'
                    elif expUpper < expLower:
                        upper[indexUpper] = '1'
                    else:
                        lower[indexLower] = '1'

                    # reduce the exponent
                    minExp = np.min([expUpper, expLower])
                    lowerExp[indexLower] -= minExp
                    upperExp[indexUpper] -= minExp

        # remove '1' if the upper or lower is longer than 1
        if len(upper) > 1:
            indexesToRemove = [i for i, elem in enumerate(upper) if elem == '1']
            upper = [elem for i, elem in enumerate(upper) if i not in indexesToRemove]
            upperPrefix = [elem for i, elem in enumerate(upperPrefix) if i not in indexesToRemove]
            upperExp = [elem for i, elem in enumerate(upperExp) if i not in indexesToRemove]
        if len(lower) > 1:
            indexesToRemove = [i for i, elem in enumerate(lower) if elem == '1']
            lower = [elem for i, elem in enumerate(lower) if i not in indexesToRemove]
            lowerPrefix = [elem for i, elem in enumerate(lowerPrefix) if i not in indexesToRemove]
            lowerExp = [elem for i, elem in enumerate(lowerExp) if i not in indexesToRemove]

        # return the list ['1'] if there are no more units
        if not upper:
            upper = ['1']
            upperExp = ['1']
        if not lower:
            lower = ['1']
            lowerExp = ['1']
        return upper, upperPrefix, upperExp, lower, lowerPrefix, lowerExp

    @staticmethod
    def _combineUpperAndLower(upper, upperPrefix, upperExp, lower, lowerPrefix, lowerExp):

        upperPrefix = [elem if not elem is None else "" for elem in upperPrefix]
        lowerPrefix = [elem if not elem is None else "" for elem in lowerPrefix]
        upperExp = [str(elem) if elem != 1 else "" for elem in upperExp]
        lowerExp = [str(elem) if elem != 1 else "" for elem in lowerExp]
        upper = [pre + up + exp for pre, up, exp in zip(upperPrefix, upper, upperExp) if up != "1"]
        lower = [pre + low + exp for pre, low, exp in zip(lowerPrefix, lower, lowerExp) if low != "1"]

        # create a unit string
        u = hyphen.join(upper) if upper else "1"
        if lower:
            lower = hyphen.join(lower)
            u = u + '/' + lower

        return u

    @staticmethod
    def _multiply(a, b):

        aUpper, aUpperPrefix, aUpperExp, aLower, aLowerPrefix, aLowerExp = unit._getLists(a)
        bUpper, bUpperPrefix, bUpperExp, bLower, bLowerPrefix, bLowerExp = unit._getLists(b)

        upper = aUpper + bUpper
        upperPrefix = [elem if not elem is None else '' for elem in aUpperPrefix + bUpperPrefix]
        upperExp = aUpperExp + bUpperExp
        lower = aLower + bLower
        lowerPrefix = [elem if not elem is None else '' for elem in aLowerPrefix + bLowerPrefix]
        lowerExp = aLowerExp + bLowerExp

        def reduceLists(units, unitPrefixes, unitExponents):
            # combine the prefix and the unit
            units = [pre + u for u, pre in zip(units, unitPrefixes)]

            # loop over all unique combinations of prefix and units
            setUnits = set(units)
            tmpUnits = [''] * len(setUnits)
            for i, u in enumerate(setUnits):
                indexes = [_ for _, elem in enumerate(units) if elem == u]
                exponent = sum([unitExponents[elem] for elem in indexes])
                tmpUnits[i] = u

                # add the exponent
                if exponent != 1 and u != '1':
                    tmpUnits[i] += str(exponent)

            # split in to lists againt
            tmpUnits = [unit._removeExponentFromUnit(elem) for elem in tmpUnits]
            exponents = [elem[1] for elem in tmpUnits]
            tmpUnits = [elem[0]for elem in tmpUnits]
            tmpUnits = [unit._removePrefixFromUnit(elem) for elem in tmpUnits]
            prefixes = [elem[1] for elem in tmpUnits]
            units = [elem[0]for elem in tmpUnits]
            return units, prefixes, exponents

        upper, upperPrefix, upperExp = reduceLists(upper, upperPrefix, upperExp)
        lower, lowerPrefix, lowerExp = reduceLists(lower, lowerPrefix, lowerExp)

        upper, upperPrefix, upperExp, lower, lowerPrefix, lowerExp = unit._cancleUnits(
            upper,
            upperPrefix,
            upperExp,
            lower,
            lowerPrefix,
            lowerExp
        )

        out = unit._combineUpperAndLower(upper, upperPrefix, upperExp, lower, lowerPrefix, lowerExp)

        return out

    @staticmethod
    def _splitUnitExponentAndPrefix(listOfUnitStrings):
        n = len(listOfUnitStrings)
        uList = [None] * n
        prefixList = [None] * n
        expList = [None] * n
        for i in range(n):
            prefixUnit, exponent = unit._removeExponentFromUnit(listOfUnitStrings[i])
            u, prefix = unit._removePrefixFromUnit(prefixUnit)
            uList[i] = u
            prefixList[i] = prefix
            expList[i] = exponent
        return uList, prefixList, expList

    @staticmethod
    def _getLists(unitStr):
        upper, lower = unit._splitCompositeUnit(unitStr)

        upper, upperPrefix, upperExp = unit._splitUnitExponentAndPrefix(upper)
        lower, lowerPrefix, lowerExp = unit._splitUnitExponentAndPrefix(lower)

        if lower:
            upper = ['DELTA' + elem if elem in temperature else elem for elem in upper]
            lower = ['DELTA' + elem if elem in temperature else elem for elem in lower]

        return upper, upperPrefix, upperExp, lower, lowerPrefix, lowerExp

    @ staticmethod
    def _formatUnit(unitStr):
        # Removing any spaces
        unitStr = unitStr.replace(' ', '')

        # Removing any illegal symbols
        for s in unitStr:
            if s not in knownCharacters:
                raise ValueError(f'The character {s} is not used within the unitsystem')

        ## find start parenthesis is the unit string
        startParenIndexes = [i for i, s in enumerate(unitStr) if s == '(']
        stopParenIndexes = [i for i, s in enumerate(unitStr) if s == ')']

        ## return if no parenthesis where found
        if not startParenIndexes and not stopParenIndexes:
            return unitStr
      
        ## check that the number of start and stop parenthesis are equal
        if len(startParenIndexes) != len(stopParenIndexes):
            raise ValueError('The unit string has to have an equal number of open parenthesis and clsoe parenthesis')
        
        ## find all parenthesis pairs
        allIndexes = startParenIndexes + stopParenIndexes
        allIndexes.sort()
        isStartParen = [elem in startParenIndexes for elem in allIndexes]
        done, iter, maxIter, order  = False, 0, len(allIndexes), []
        while not done:
            for i in range(len(allIndexes)-1):
                if isStartParen[i] and (isStartParen[i] + isStartParen[i+1] == 1):
                    order.append([allIndexes[i], allIndexes[i+1]])
                    allIndexes.pop(i+1)
                    allIndexes.pop(i)
                    isStartParen.pop(i+1)
                    isStartParen.pop(i)
                    break

            if len(allIndexes) == 0: break
            
            iter += 1
            if iter > maxIter:
                raise ValueError('An order to evaluate the parenthesis could not be found')
        

        ## find the outer most parenthesis        
        done, maxIter, iter = False, len(order), 0
        while not done:
            changesMade = False
            for i in range(len(order)):
                iLow = order[i][0]
                iHigh = order[i][1]
                for j in range(len(order)):
                    jLow = order[j][0]
                    jHigh = order[j][1]
                    if iLow < jLow and iHigh > jHigh:
                        order.pop(j)
                        changesMade = True
                        break
                if changesMade:
                    break
            if not changesMade:
                break
            iter += 1
            if iter > maxIter:
                raise ValueError('An order to evaluate the parenthesis could not be found')
        
        ## find all slashes outside the outer most parenthesis
        slashes = [i for i, s in enumerate(unitStr) if s == slash]

        iter, maxIter, done = 0, len(slashes), False
        while not done:
            done = True
            for s in slashes:    
                for part in order:
                    if part[0] < s < part[1]:
                        done = False
                        slashes.remove(s)
                        break                
            if done: break
            iter += 1
            if iter > maxIter:
                raise ValueError('An order to evaluate the parenthesis could not be found')

        if len(slashes) > 1:
            raise ValueError('You have more than one slash ("/") outside the outer most parenthesis. This makes the unit ambiguis')
        
        if slashes:
            splits = [-1] + slashes
            parts = [unitStr[i+1:j] for i,j in zip(splits, splits[1:]+[None])]
            parts = [unit._formatUnit(part) for part in parts]
            if isinstance(parts, list):
                parts = unit(parts[0]) / unit(parts[1])
            return parts
        
        parts = unit._formatUnit(unitStr[order[0][0]+1: order[0][1]])
        if order[0][1] != len(unitStr) - 1:
            remaining = unitStr[order[0][1]+1:]
            parts = parts.split('/')
            if len(parts) > 1:
                upper, lower = parts
                lower = lower.split('-')
            else:
                upper, lower = parts[0], []
            upper = upper.split('-')
            try: 
                exponent = int(remaining)
                for i, up in enumerate(upper):
                    u, e = unit._removeExponentFromUnit(up)
                    upper[i] = u + str(e * exponent)
                for i, low in enumerate(lower):
                    u, e = unit._removeExponentFromUnit(low)
                    lower[i] = u + str(e * exponent)
                out = '-'.join(upper)
                if lower:
                    out += '/' + '-'.join(lower)
                return out
            except TypeError:
                raise TypeError()
        return parts
    
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

    @ staticmethod
    def _removeExponentFromUnit(u):
        integerIndexes = [i for i, char in enumerate(u) if char in integers]
        exponent = 1
        if not integerIndexes:
            return u, exponent

        # override the exponent if there are any integerindexes
        if integerIndexes:
            # determine if all integers are consectutive together
            # sum(a, a+1, ... b-1, b) = (b * (b-1) - a * (a-1)) / 2
            minIndex, maxIndex = integerIndexes[0] - 1, integerIndexes[-1]
            if sum(integerIndexes) != (maxIndex * (maxIndex + 1) - minIndex * (minIndex + 1)) / 2:
                raise ValueError('All numbers in the unit has to be grouped together')

            # Determine if the last integer is placed at the end of the unit
            if integerIndexes[-1] != len(u) - 1:
                raise ValueError('Any number has to be placed at the end of the unit')

            exponent = int(u[minIndex+1:])
            u = u[:minIndex+1]


        # Ensure that the entire use was not removed by removing the integers
        if not u:
            # No symbols are left after removing the integers
            if exponent != 1:
                raise ValueError(f'The unit {u} was stripped of all integers which left no symbols in the unit. This is normally due to the integers removed being equal to 1, as the unit is THE unit. Howver, the intergers removed was not equal to 1. The unit is therefore not known.')
            u = '1'
        return u, exponent

    @staticmethod
    def _assertEqualStatic(a, b):

        aUpper, aLower = unit._splitCompositeUnit(a)
        bUpper, bLower = unit._splitCompositeUnit(b)
               
        for elem in aUpper:
            if not elem in bUpper:
                return False
            bUpper.remove(elem)
        if bUpper:
            return False
        
        for elem in aLower:
            if not elem in bLower:
                return False
            bLower.remove(elem)
        if bLower:
            return False
        
        return True

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

    @staticmethod
    def _getSIBaseUnit(upper, upperExp, lower, lowerExp):       
        nUpper = len(upper)
        upperOut = []
        lowerOut = []
        upperExpOut = []
        lowerExpOut = []
        for i, (elem, exp) in enumerate(zip(upper + lower, upperExp + lowerExp)):
            up,low = unit._splitCompositeUnit(knownUnits[elem][0])
            if (i >= nUpper):
                up, low = low, up
            for elem in up:
                elem, elemExp = unit._removeExponentFromUnit(elem)
                if not elem in upperOut:
                    upperOut.append(elem)
                    upperExpOut.append(elemExp * exp)
                else:
                    index = upperOut.index(elem)
                    upperExpOut[index] += elemExp * exp
            for elem in low:
                elem, elemExp = unit._removeExponentFromUnit(elem)
                if not elem in lowerOut:
                    lowerOut.append(elem)
                    lowerExpOut.append(elemExp * exp)
                else:
                    index = lowerOut.index(elem)
                    lowerExpOut[index] += elemExp * exp

        upperPrefix = [None] * len(upperOut)
        lowerPrefix = [None] * len(lowerOut)
        
        upper, upperPrefix, upperExp, lower, lowerPrefix, lowerExp = unit._cancleUnits(
            upperOut, upperPrefix, upperExpOut, lowerOut, lowerPrefix, lowerExpOut
        )

        return unit._combineUpperAndLower(upper, upperPrefix, upperExp, lower, lowerPrefix, lowerExp)

    def isCombinationUnit(self):
        if len(self.upper) > 1:
            return True
        if self.lower:
            return True
        return False

    def __str__(self, pretty=False):
        if not pretty:
            return self.unitStr
        else:
            if self.lower:
                # a fraction is needed
                out = rf'\frac{{'
                for i, (up, prefix, exp) in enumerate(zip(self.upper, self.upperPrefix, self.upperExp)):
                    if exp > 1:
                        up = rf'{up}^{exp}'
                    if prefix is None:
                        prefix = ''
                    out += rf'{prefix}{up}'
                    if i != len(self.upper) - 1:
                        out += rf' \cdot '
                out += rf'}}{{'
                for i, (low, prefix, exp) in enumerate(zip(self.lower, self.lowerPrefix, self.lowerExp)):
                    if exp > 1:
                        low = rf'{low}^{exp}'
                    if prefix is None:
                        prefix = ''
                    out += rf'{prefix}{low}'
                    if i != len(self.lower) - 1:
                        out += rf' \cdot '
                out += rf'}}'
            else:
                # no fraction
                out = r''
                for i, (up, prefix, exp) in enumerate(zip(self.upper, self.upperPrefix, self.upperExp)):
                    if exp > 1:
                        up = rf'{up}^{exp}'
                    if prefix is None:
                        prefix = ''
                    out += rf'{prefix}{up}'
                    if i != len(self.upper) - 1:
                        out += rf' \cdot '
            return out

    def _assertEqual(self, other):
        if isinstance(other, unit):
            other = other.unitStr
        return self._assertEqualStatic(self.unitStr, other)

    def __add__(self, other):
        
        ## output 1: are the units a logarithmic unit
        ## output 2: outputUnit
        ## output 3: scaleToSI. If true, then scale both self and other to SI and neglect output 2 and 3
        ## output 4: scaleSelf. If true then scale self to remove the prefix
        ## output 5: scaleOther. If true then scale other to remove the prefix
        
        
        selfWithoutPrefixes = unit(unit._combineUpperAndLower(self.upper, [None] * len(self.upperPrefix), self.upperExp, self.lower, [None] * len(self.lowerPrefix), self.lowerExp))
        otherWithoutPrefixes = unit(unit._combineUpperAndLower(other.upper, [None] * len(other.upperPrefix), other.upperExp, other.lower, [None] * len(other.lowerPrefix), other.lowerExp))
        aStr = str(selfWithoutPrefixes)
        bStr = str(otherWithoutPrefixes)
        
        isLogarithmicUnit = aStr in logrithmicUnits.keys()
                
        scaleSelf = str(self) != aStr
        scaleOther = str(other) != bStr
        
        if unit._assertEqualStatic(aStr, bStr):
            return isLogarithmicUnit, aStr, False, scaleSelf, scaleOther
        
        # test if the SI base units are identical
        if self._SIBaseUnit == other._SIBaseUnit:
            return isLogarithmicUnit, self._SIBaseUnit, True, False, False
        
        
        SIBaseUnits = [selfWithoutPrefixes._SIBaseUnit,  otherWithoutPrefixes._SIBaseUnit]        
        if 'DELTAK' in SIBaseUnits and 'K' in SIBaseUnits:
            
            indexTemp = SIBaseUnits.index('K')
            indexDiff = 0 if indexTemp == 1 else 1
            
            units = [aStr, bStr]

            if units[indexTemp] == units[indexDiff][-1]:        
                return isLogarithmicUnit, units[indexTemp], False, scaleSelf, scaleOther
            
            return isLogarithmicUnit, 'K', True, False, False

        raise ValueError(f'You tried to add a variable in [{self}] to a variable in [{other}], but the units do not have the same SI base unit')

    def __sub__(self, other):
        ## output 1: are the units a logarithmic unit
        ## output 2: outputUnit
        ## output 3: scaleToSI. If true, then scale both self and other to SI and neglect output 2 and 3
        ## output 4: scaleSelf. If true then scale self to remove the prefix
        ## output 5: scaleOther. If true then scale other to remove the prefix
        
        
        selfWithoutPrefixes = unit(unit._combineUpperAndLower(self.upper, [None] * len(self.upperPrefix), self.upperExp, self.lower, [None] * len(self.lowerPrefix), self.lowerExp))
        otherWithoutPrefixes = unit(unit._combineUpperAndLower(other.upper, [None] * len(other.upperPrefix), other.upperExp, other.lower, [None] * len(other.lowerPrefix), other.lowerExp))
        aStr = str(selfWithoutPrefixes)
        bStr = str(otherWithoutPrefixes)
        
        isLogarithmicUnit = aStr in logrithmicUnits.keys()
                
        scaleSelf = str(self) != aStr
        scaleOther = str(other) != bStr

        SIBaseUnits = [self._SIBaseUnit, other._SIBaseUnit]
        if SIBaseUnits[0] == 'K' and SIBaseUnits[1] == 'K':
            if aStr == bStr:
                return isLogarithmicUnit, 'DELTA' + str(self), False, scaleSelf, scaleOther
            return isLogarithmicUnit,'DELTAK', True, False, False

        if 'DELTAK' in SIBaseUnits and 'K' in SIBaseUnits:
            
            indexTemp = SIBaseUnits.index('K')
            if indexTemp != 0:
                raise ValueError('You tried to subtract a temperature from a temperature differnce. This is not possible.')
            indexDiff = 0 if indexTemp == 1 else 1
            
            units = [aStr, bStr]


            if units[indexTemp] == units[indexDiff][-1]:        
                return isLogarithmicUnit, units[indexTemp], False, scaleSelf, scaleOther
            
            return isLogarithmicUnit, 'K', True, False, False
        
        if unit._assertEqualStatic(aStr, bStr):
            return isLogarithmicUnit, aStr, False, scaleSelf, scaleOther
        
        # test if the SI base units are identical
        if self._SIBaseUnit == other._SIBaseUnit:
            return isLogarithmicUnit, self._SIBaseUnit, True, False, False
        
        
        
        raise ValueError(f'You tried to subtract a variable in [{other}] from a variable in [{self}], but the units do not have the same SI base unit')


    def __mul__(self, other):
        return unit._multiply(self.unitStr, other.unitStr)

    def __truediv__(self, other):
        
        a = self.unitStr
        b = other.unitStr

        bUpper, bUpperPrefix, bUpperExp, bLower, bLowerPrefix, bLowerExp = unit._getLists(b)

        b = self._combineUpperAndLower(
            upper=bLower,
            upperPrefix=bLowerPrefix,
            upperExp=bLowerExp,
            lower=bUpper,
            lowerPrefix=bUpperPrefix,
            lowerExp=bUpperExp
        )

        return unit._multiply(a, b)

    def __pow__(self, power):
        if power == 0:
            return '1', False

        elif power > 1:

            if self.unitStr == '1':
                # self is '1'. Therefore the power does not matter
                return self.unitStr, False

            else:
                # self is not '1'. Therefore all exponents are multiplied by the power

                if not (isinstance(power, int) or power.is_integer()):
                    raise ValueError('The power has to be an integer')

                upperExp = [int(elem * power) for elem in self.upperExp]
                lowerExp = [int(elem * power) for elem in self.lowerExp]

                return self._combineUpperAndLower(self.upper, self.upperPrefix, upperExp, self.lower, self.lowerPrefix, lowerExp) , False

        else:
            # the power is smaller than 1.
            # Therefore it is necessary to determine if all exponents are divisible by the recibricol of the power

            if self.unitStr == '1':
                # self is '1'. Therefore the power does not matter
                return self.unitStr, False
            else:
                # self is not '1'.
                # Therefore it is necessary to determine if all exponents are divisible by the recibricol of the power

                def isCloseToInteger(a, rel_tol=1e-9, abs_tol=0.0):
                    b = np.around(a)
                    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
                
                # Test if the exponent of all units is divisible by the power
                try:
                    for exp in self.upperExp + self.lowerExp:
                        if not isCloseToInteger(exp * power):
                            raise ValueError(f'You can not raise a variable with the unit {self.unitStr} to the power of {power}')

                    upperExp = [int(elem * power) for elem in self.upperExp]
                    lowerExp = [int(elem * power) for elem in self.lowerExp]

                    return self._combineUpperAndLower(self.upper, self.upperPrefix, upperExp, self.lower, self.lowerPrefix, lowerExp), False
                
                except ValueError:                
                    ## the exponents of the unit was not divisible by the power
                    ## test if the exponents of the SIBaseunit is divisible by the power 
                    
                    siUpper, siUpperPrefix, siUpperExp, siLower, siLowerPrefix, siLowerExp = self._getLists(self._SIBaseUnit)
                    for exp in siUpperExp + siLowerExp:
                        if not isCloseToInteger(exp * power):
                            raise ValueError(f'You can not raise a variable with the unit {self.unitStr} to the power of {power}')

                    siUpperExp = [int(elem * power) for elem in siUpperExp]
                    siLowerExp = [int(elem * power) for elem in siLowerExp]

                    return self._combineUpperAndLower(siUpper, siUpperPrefix, siUpperExp, siLower, siLowerPrefix, siLowerExp), True


    def getConverter(self, newUnit):
        newUnit = unit._formatUnit(newUnit)

        # get the upper, upperExp, lower and lowerExp of the newUnit without creating a unit
        otherUpper, otherUpperPrefix, otherUpperExp, otherLower, otherLowerPrefix, otherLowerExp = self._getLists(newUnit)

        # determine if the SI bases are identical
        otherSIBase = self._getSIBaseUnit(otherUpper, otherUpperExp, otherLower, otherLowerExp)

        if unit._assertEqualStatic(self._SIBaseUnit, otherSIBase) == False:
            raise ValueError(f'You tried to convert from {self} to {newUnit}. But these do not have the same base units')
        
        return self._getConverter(otherUpper, otherUpperPrefix, otherUpperExp, otherLower, otherLowerPrefix, otherLowerExp)

    def getUnitWithoutPrefix(self):
        return self._removePrefixFromUnit(self.unitStr)[0]

    def getLogarithmicConverter(self, unitStr = None):
        if unitStr is None:
            unitStr = self.unitStr
        u, p = self._removePrefixFromUnit(unitStr)
        return knownUnits[u][1]
   
    def _getConverter(self, otherUpper, otherUpperPrefix, otherUpperExp, otherLower, otherLowerPrefix, otherLowerExp):
        
        # initialize the scale and offset
        out = _unitConversion(1, 0)

        # get conversions for all upper and lower units in self
        upperConversions = [knownUnits[elem][1] for elem in self.upper]
        lowerConversions = [knownUnits[elem][1] for elem in self.lower]

        # modify the scale and offset using the conversions
        conversions = upperConversions + lowerConversions
        nUpper = len(upperConversions)
        prefixes = self.upperPrefix + self.lowerPrefix
        exponents = self.upperExp + self.lowerExp
        for i, (conv, prefix, exp) in enumerate(zip(conversions, prefixes, exponents)):
            if exp > 1: conv = conv ** exp
            if not prefix is None:
                conv *= knownPrefixes[prefix]
            if i < nUpper:
                out *= conv
            else:
                out /= conv
                        
        # get all conversions from the upper and lower units in the new unit
        upperConversions = [knownUnits[elem][1] for elem in otherUpper]
        lowerConversions = [knownUnits[elem][1] for elem in otherLower]

        # modify the scale and offset based on the conversions
        conversions = upperConversions + lowerConversions
        nUpper = len(otherUpper)
        prefixes = otherUpperPrefix + otherLowerPrefix
        exponents = otherUpperExp + otherLowerExp
        for i, (conv, prefix, exp) in enumerate(zip(conversions, prefixes, exponents)):
            if exp > 1: conv = conv ** exp
            if not prefix is None:
                conv *= knownPrefixes[prefix]
            if i < nUpper:
                out /= conv
            else:
                out *= conv

        return out

    def convert(self, unitStr):
        if unitStr == '':
            unitStr = '1'

        # remove any unknown characters
        unitStr = self._formatUnit(unitStr)

        # split the unit in upper and lower
        self.upper, self.upperPrefix, self.upperExp, self.lower, self.lowerPrefix, self.lowerExp = self._getLists(unitStr)

        # create the unit string
        self.unitStr = self._createUnitString()

        self._SIBaseUnit = self._getSIBaseUnit(self.upper, self.upperExp, self.lower, self.lowerExp)
        otherUpper, otherUpperPrefix, otherUpperExp, otherLower, otherLowerPrefix, otherLowerExp = self._getLists(self._SIBaseUnit)
        self._converterToSI = self._getConverter(otherUpper, otherUpperPrefix, otherUpperExp, otherLower, otherLowerPrefix, otherLowerExp)


if __name__ == "__main__":
    a = unit('L/min')
    b = unit('Pa')
    
    c = unit(a * b)
    
    print(c)
    d = unit('m3-Pa/s')
    print(unit._assertEqualStatic(c._SIBaseUnit, d._SIBaseUnit))