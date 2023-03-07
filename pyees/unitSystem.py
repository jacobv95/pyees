import numpy as np

class _unitConversion():

    def __init__(self, scale, offset=0) -> None:
        self.scale = scale
        self.offset = offset

    def __mul__(self, other):
        if isinstance(other, _unitConversion):
            scale = self.scale * other.scale
            offset = self.offset * other.scale + other.offset
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
            scale = self.scale / other.scale
            offset = self.offset - other.offset / other.scale
        else:
            scale = self.scale / other
            offset = self.offset
        return _unitConversion(scale, offset)


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

## TODO octave
## TODO decade
    
_baseUnit = {
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
    'octave': _unitConversion(1),
    'decade': _unitConversion(1)
}

_knownUnitsDict = {
    'kg-m/s2': force,
    'kg/m-s2': pressure,
    's': time,
    'K': temperature,
    'DELTAK': temperatureDifference,
    'm3': volume,
    'm': length,
    'kg-m2/s2': energy,
    'kg-m2/s3': power,
    'kg': mass,
    'A': current,
    'kg-m2/s3-A': voltage,
    '1': _baseUnit,
    '1/s': frequency,
    'rad': angle,
    'kg-m2/s3-A2' : resistance,
    'm2/s' : kinematicViscosity,
    'Np': logrithmicUnits
}

_knownPrefixes = {
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

def getDicts():

    _knownUnits = {}
    for key, d in _knownUnitsDict.items():
        for item, _ in d.items():
            if item not in _knownUnits:
                _knownUnits[item] = [key, _knownUnitsDict[key][item]]
            else:
                raise Warning(f'The unit {item} known in more than one unit system')



    # determine the known characters within the unit system
    _knownCharacters = list(_knownUnits.keys()) + list(_knownPrefixes.keys())
    _knownCharacters = ''.join(_knownCharacters)
    _knownCharacters += '-/'
    _knownCharacters += '0123456789'
    _knownCharacters += '()'
    _knownCharacters = ''.join(list(set(_knownCharacters)))

    # check if all unit and prefix combinations can be distiguished
    unitPrefixCombinations = []
    for u in _knownUnits:
        unitPrefixCombinations += [u]
        if u not in _baseUnit:
            unitPrefixCombinations += [p + u for p in _knownPrefixes]

    for elem in unitPrefixCombinations:
        count = sum([1 if u == elem else 0 for u in unitPrefixCombinations])
        if count > 1:
            prefix = elem[0:1]
            unit = elem[1:]

            unitType1 = ''
            for key, item in _knownUnitsDict.items():
                if elem in item:
                    unitType1 = [key for key, a in locals().items() if a == item][0]
            if unitType1 == '':
                raise ValueError(f'The unit {elem} was not found.')

            unitType2 = ''
            for key, item in _knownUnitsDict.items():
                if unit in item:
                    unitType2 = [key for key, a in locals().items() if a == item][0]
            if unitType2 == '':
                raise ValueError(f'The unit {unit} was not found.')

            raise ValueError(f'The unit {elem} can be interpreted as a {unitType1} or a {unitType2} with the prefix {prefix}. The cannot be distiguished.')

    return _knownUnits, _knownCharacters, _knownUnitsDict, _knownPrefixes, _baseUnit






