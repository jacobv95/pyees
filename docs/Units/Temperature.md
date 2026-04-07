# Temperature

Temperatures are wierd. The units celsius and fahrenheit are relative units, meaning that 20 [C] is equal to 20 [K] above the temperature 273.15 [K]. This makes addition and subtraction of temperatures ambiguis. Pyees implements the following rules with respect to tempatures

 - A tempaturedifference is returned, when subtracing two temperatures
 - Addition of two temperatures are handled like any other unit
 
```
import pyees as pe
a = pe.variable(30,'C')
b = pe.variable(20,'C')
a - b
## 30 [C] - 20 [C] = 10 [DELTAC]
>> 10 [DELTAC]


c = pe.variable(20,'K')
a - c
## 30 [C] - 20 [K] = (30 [K] + 273.15 [K]) - 20[K] = 300.15 [K] - 20 [K] = 283.15 [DELTAK]
>> 283.15 [DELTAK]

a + b
## 30 [C] + 20 [C] = 50 [C]
>> 50 [C]
```



 - A temperature cannot be subtracted from a tempertaure difference

```
import pyees as pe
a = pe.variable(30,'DELTAC')
b = pe.variable(20,'C')
a - b
>> ValueError: You tried to subtract a temperature from a temperature differnce. This is not possible.
```
