# Add new units


You can add new units to the unitsymstem using the method "addNewUnits"

```
addNewUnit(newUnit : str, scale : float, existingUnit : str, offset : float = 0)
```

The arguments for the method "addNewUnit" has to be read in the following way:
    1 [newUnit] = [scale] * [existingUnit] + [offset]

Any new units added to the unitsystem can be used in combination with any prefix in the unitsystem.

## exmaple
```
import pyees as pe

pe.addNewUnit('inch', 25.4, 'mm')
a = pe.variable(1, 'inch')
a.convert('mm')
print(a)
>> 25.4 [mm]
print(a.value)
>> 25.4
## 1 inch is equal to 25.4 mm according to WolframAlpha

addNewUnit('Rø', 40/21, 'C', -7.5 * 40/21)
b = pe.variable(83.1, 'Rø')
b.convert('F')
print(b)
>> 291 [F]
print(b.value)
>> 291.19999999999993
## 83.1 Rømer is equal to 291.2 Fahrenheit according to WolframAlpha


c = pe.variable(143.6, 'mRø')
c.convert('K')
print(c)
>> 259 [K]
print(c.value)
>> 259.1378095238095
## 143.6 milli rømer is equal to 259.13781 kelvin according to WolframAlpha
```