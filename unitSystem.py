

class unitConversion():

    def __init__(self, scale, offset) -> None:
        self.scale = scale
        self.offset = offset

    def convertToSI(self, upper=True, isComposite=False):
        if upper:
            if isComposite:
                return [self.scale, 0]
            else:
                return [self.scale, self.offset]
        else:
            return self.convertFromSI(not upper, isComposite)

    def convertFromSI(self, upper=True, isComposite=False):
        if upper:
            if isComposite:
                return [1 / self.scale, 0]
            else:
                return [1 / self.scale, -self.offset / self.scale]
        else:
            return self.convertToSI(not upper, isComposite)


class unit():
    def __init__(self) -> None:

        unit = {
            '1': unitConversion(1, 0)
        }

        force = {
            'N': unitConversion(1, 0)
        }

        mass = {
            'g': unitConversion(1 / 1000, 0)
        }

        energy = {
            'J': unitConversion(1, 0),
        }

        effect = {
            'W': unitConversion(1, 0)
        }

        pressure = {
            'Pa': unitConversion(1, 0),
            'bar': unitConversion(1e5, 0)
        }

        temperature = {
            'K': unitConversion(1, 0),
            'C': unitConversion(1, 273.15),
            'F': unitConversion(5 / 9, 273.15 - 32 * 5 / 9)
        }

        time = {
            's': unitConversion(1, 0),
            'min': unitConversion(60, 0),
            'h': unitConversion(60 * 60, 0),
            'yr': unitConversion(60 * 60 * 24 * 365, 0)
        }

        volume = {
            'm3': unitConversion(1, 0),
            'L': unitConversion(1 / 1000, 0)
        }

        length = {
            'm': unitConversion(1, 0)
        }

        self.units = {
            'kg-m/s2': force,
            'kg/m-s2': pressure,
            's': time,
            'K': temperature,
            'm3': volume,
            'm': length,
            'kg-m2/s2': energy,
            'kg-m2/s3': effect,
            'kg': mass,
            '1': unit
        }

        self.prefixes = {
            'µ': 1e-6,
            'm': 1e-3,
            'k': 1e3,
            'M': 1e6
        }

    def convertToSI(self, value, unit):

        upper, lower = self.splitCompositeUnit(unit)

        isComposite = not (len(lower) == 0 and len(upper) == 1)
        unitUpper = []
        unitLower = []
        for unit in upper:
            conversion, u, exp = self.convert(unit, toSI=True, upper=True, isComposite=isComposite)
            for _ in range(exp):
                value = value * conversion[0] + conversion[1]

            siUpper, siLower = self.splitCompositeUnit(u)
            siUpperExp = []
            siLowerExp = []
            for i, up in enumerate(siUpper):
                u, siExp = self.removeExponentFromUnit(up)
                siUpper[i] = u
                siUpperExp.append(siExp * exp)
            for i, low in enumerate(siLower):
                u, siExp = self.removeExponentFromUnit(low)
                siLower[i] = u
                siLowerExp.append(siExp * exp)

            for up, upExp in zip(siUpper, siUpperExp):
                if upExp != 1:
                    up += str(upExp)
                unitUpper.append(up)
            for low, lowExp in zip(siLower, siLowerExp):
                if lowExp != 1:
                    low += str(lowExp)
                unitLower.append(low)

        for unit in lower:
            conversion, u, exp = self.convert(unit, toSI=True, upper=False, isComposite=isComposite)
            for _ in range(exp):
                value = value * conversion[0] + conversion[1]

            siUpper, siLower = self.splitCompositeUnit(u)
            siUpperExp = []
            siLowerExp = []
            for i, up in enumerate(siUpper):
                u, siExp = self.removeExponentFromUnit(up)
                siUpper[i] = u
                siUpperExp.append(siExp * exp)
            for i, low in enumerate(siLower):
                u, siExp = self.removeExponentFromUnit(low)
                siLower[i] = u
                siLowerExp.append(siExp * exp)

            for up, upExp in zip(siUpper, siUpperExp):
                if upExp != 1:
                    up += str(upExp)
                unitLower.append(up)
            for low, lowExp in zip(siLower, siLowerExp):
                if lowExp != 1:
                    low += str(lowExp)
                unitUpper.append(low)

        upperUpper = []
        upperLower = []
        lowerUpper = []
        lowerLower = []
        for u in unitUpper:
            up, low = self.splitCompositeUnit(u)
            upperUpper += up
            upperLower += low
        for u in unitLower:
            up, low = self.splitCompositeUnit(u)
            lowerUpper += up
            lowerLower += low
        unitUpper = upperUpper + lowerLower
        unitLower = upperLower + lowerUpper

        # cancle out upper and lower
        unitUpper, unitLower = self.cancleUnits(unitUpper, unitLower)

        # combine the upper and lower
        outUnit = self.combineUpperAndLower(unitUpper, unitLower)

        return value, outUnit

    def convertFromSI(self, value, unit):

        upper, lower = self.splitCompositeUnit(unit)
        isComposite = not (len(lower) == 0 and len(upper) == 1)

        for u in upper:
            conversion, u, exp = self.convert(u, toSI=False, upper=True, isComposite=isComposite)
            for _ in range(exp):
                value = value * conversion[0] + conversion[1]
        for u in lower:
            conversion, u, exp = self.convert(u, toSI=False, upper=False, isComposite=isComposite)
            for _ in range(exp):
                value = value * conversion[0] + conversion[1]

        return value, unit

    def splitCompositeUnit(self, compositeUnit):

        special_characters = """!@#$%^&*()+?_=,.<>\\"""
        if any(s in compositeUnit for s in special_characters):
            raise ValueError('The unit can only contain slashes (/) and hyphens (-)')

        # remove spaces
        compositeUnit = compositeUnit.replace(' ', '')
        slash = '/'
        if slash in compositeUnit:
            index = compositeUnit.find('/')
            upper = compositeUnit[0:index]
            lower = compositeUnit[index + 1:]

            # check for multiple slashes
            if slash in upper or slash in lower:
                raise ValueError('A unit can only have a single slash (/)')

            # split the upper and lower
            upper = upper.split('-')
            lower = lower.split('-')

        else:
            upper = compositeUnit.split('-')
            lower = []
        return upper, lower

    def removeExponentFromUnit(self, unit):

        # find any integers in the unit
        num = []
        num_indexes = []
        for i, s in enumerate(unit):
            if s.isdigit():
                num.append(s)
                num_indexes.append(i)

        # determine if all integers are placed consequtively
        for i in range(len(num_indexes) - 1):
            elem_curr = num_indexes[i]
            elem_next = num_indexes[i + 1]
            if not elem_next == elem_curr + 1:
                raise ValueError('All numbers in the unit has to be grouped together')

        # determien if the last integer is placed at the end of the unit
        if len(num) != 0:
            if max(num_indexes) != len(unit) - 1:
                raise ValueError('Any number has to be placed at the end of the unit')

        # remove the inters from the unit
        if len(num) != 0:
            for i in reversed(num_indexes):
                unit = unit[0:i] + unit[i + 1:]

       # combine the exponent
        if len(num) != 0:
            exponent = int(''.join(num))
        else:
            exponent = 1

        # check if the intire unit has been removed by the integers.
        if len(unit) == 0:
            # check if the exponent is equal to 1
            if exponent == 1:
                unit = '1'
        return unit, exponent

    def convert(self, unit, toSI=True, upper=True, isComposite=False):
        unit, exponent = self.removeExponentFromUnit(unit)

        # search for the unit
        isFound = False
        for siUnit, unitDict in self.units.items():
            if unit in unitDict:
                conversion = unitDict[unit]
                isFound = True
                break

        # check if the unit is found
        if isFound:
            # retrun the conversion if it is found
            if toSI:
                out = conversion.convertToSI(upper, isComposite)
            else:
                out = conversion.convertFromSI(upper, isComposite)

            # the unti was found without looking for the prefix. Therefore the prefix must be 1
            prefix = 1
        else:
            # The unit was not found. This must be because the unit has a prefix

            prefix = unit[0:1]
            unit = unit[1:]
            if prefix not in self.prefixes:
                raise ValueError(f'The unit ({prefix}{unit}) was not found. Therefore it was interpreted as a prefix and a unit. However the prefix ({prefix}) was not found')

            # look for the unit without the prefix
            isFound = False
            for siUnit, unitDict in self.units.items():
                if unit in unitDict:
                    conversion = unitDict[unit]
                    isFound = True
                    break

            # check if the unit was found
            if not isFound:
                raise ValueError(f'The unit ({prefix}{unit}) was not found. Therefore it was interpreted as a prefix and a unit. However the unit ({unit}) was not found')

            # create the conversion
            if toSI:
                out = conversion.convertToSI(upper, isComposite)
            else:
                out = conversion.convertFromSI(upper, isComposite)

            # The prefix is inverted if the conversion is not to SI
            prefix = self.prefixes[prefix]
            if not upper:
                prefix = 1 / prefix
            if not toSI:
                prefix = 1 / prefix

        out[0] *= prefix

        return out, siUnit, exponent

    def divide(self, unit1, unit2):
        # determine the upper and lower units of unit 2
        upperUnit2, lowerUnit2 = self.splitCompositeUnit(unit2)

        # flip unit 2
        lowerUnit2, upperUnit2 = upperUnit2, lowerUnit2

        unit2 = ''
        if len(upperUnit2) != 0:
            unit2 += '-'.join(upperUnit2)
        else:
            unit2 += '1'

        if len(lowerUnit2) != 0:
            if len(lowerUnit2) == 1:
                if lowerUnit2[0] == '1':
                    pass
                else:
                    unit2 += '/' + '-'.join(lowerUnit2)
            else:
                unit2 += '/' + '-'.join(lowerUnit2)

        return self.multiply(unit1, unit2)

    def multiply(self, unit1, unit2):

        # determine the upper and lower units of unit 1
        upperUnit1, lowerUnit1 = self.splitCompositeUnit(unit1)

        # determine the upper and lower units of unit 2
        upperUnit2, lowerUnit2 = self.splitCompositeUnit(unit2)

        # determine the combined upper and lower unit
        upper = upperUnit1 + upperUnit2
        lower = lowerUnit1 + lowerUnit2

        # cancle the upper and lower
        upper, lower = self.cancleUnits(upper, lower)

        # combine the upper and lower
        u = self.combineUpperAndLower(upper, lower)
        return u

    def cancleUnits(self, upper, lower):

        # replace units with exponents with multiple occurances of the unit in upper
        unitsToRemove = []
        unitsToAdd = []
        for up in upper:
            u, e = self.removeExponentFromUnit(up)
            if e != 1:
                unitsToRemove.append(up)
                unitsToAdd += [u] * e

        for u in unitsToRemove:
            upper.remove(u)
        for u in unitsToAdd:
            upper.append(u)

        # replace units with exponents with multiple occurances of the unit in lower
        unitsToRemove = []
        unitsToAdd = []
        for low in lower:
            u, e = self.removeExponentFromUnit(low)
            if e != 1:
                unitsToRemove.append(low)
                unitsToAdd += [u] * e

        for u in unitsToRemove:
            lower.remove(u)
        for u in unitsToAdd:
            lower.append(u)

        # cancle the upper and lower units
        unitsToRemove = []
        done = False
        while not done:
            done = True
            for low in lower:
                if low in upper:
                    upper.remove(low)
                    lower.remove(low)
                    done = False
            if done:
                break

        # remove '1'
        if len(upper) > 1:
            if '1' in upper:
                upper.remove('1')

        if len(lower) > 1:
            if '1' in lower:
                lower.remove('1')

        # determine the exponents of each unit in the upper
        upperWithExponents = []
        if len(upper) != 0:
            done = False
            while not done:
                up = upper[0]
                exponent = upper.count(up)
                if exponent != 1:
                    upperWithExponents.append(up + str(exponent))
                else:
                    upperWithExponents.append(up)
                upper = list(filter((up).__ne__, upper))
                if len(upper) == 0:
                    done = True

        # determine the exponents of each unit in the lower
        lowerWithExponents = []
        if len(lower) != 0:
            done = False
            while not done:
                low = lower[0]
                exponent = lower.count(low)
                if exponent != 1:
                    lowerWithExponents.append(low + str(exponent))
                else:
                    lowerWithExponents.append(low)
                lower = list(filter((low).__ne__, lower))
                if len(lower) == 0:
                    done = True
        return upperWithExponents, lowerWithExponents

    def combineUpperAndLower(self, upper, lower):

        # combine the upper and lower
        u = ''
        if len(upper) != 0:
            u += '-'.join(upper)
        else:
            u += '1'

        if len(lower) != 0:
            if len(lower) == 1:
                if lower[0] == '1':
                    pass
                else:
                    u += '/' + '-'.join(lower)
            else:
                u += '/' + '-'.join(lower)

        return u


def testConvertToSiAndBack():
    u = 'mm3'
    valOut, unitOut = unit().convertToSI(1, u)
    print(valOut, unitOut)

    valOut, unitOut = unit().convertFromSI(valOut, u)
    print(valOut, unitOut)


def testMultiply():
    unit1 = 'L/min'
    unit2 = '1/min3'
    unitOut = unit().divide(unit1, unit2)
    print(unitOut)


if __name__ == '__main__':
    testConvertToSiAndBack()
    # testMultiply()
