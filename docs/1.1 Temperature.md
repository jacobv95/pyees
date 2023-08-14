# Temperature

Temperatures are wierd. The units celsius and fahrenheit are relative units, meaning that 20 [C] is equal to 20 [K] above the temperature 273.15 [K]. This makes addition and subtraction of temperatures ambiguis. Pyees implements the following rules with respect to tempatures

 - A tempaturedifference is returned, when subtracing two temperatures
 
```
a = variable(30,'C')
b = variable(20,'C')
a - b
## 30 [C] - 20 [C] = 10 [DELTAC]
>> 10 [DELTAC]


c = variable(20,'K')
a - c
## 30 [C] - 20 [K] = (30 [K] + 273.15 [K]) - 20[K] = 300.15 [K] - 20 [K] = 283.15 [DELTAK]
>> 283 [DELTAK]
```


 - If temperature difference is added to or subtracted from a temperature, then a tempeature is returned in the same unit as the original temperature
```
a = variable(30,'C')
b = variable(10,'DELTAK')
a + b 
## 30 [C] + 10 [DELTAK] = 30 [C] + 10 [DELTAC] = 40 [C]
>> 40 [C]


a - b
## 30 [C] - 10 [DELTAK] = 30 [C] - 10 [DELTAC] = 20 [C]
>> 40 [C]


c = variable(13, 'F')
a - c
## 30 [C] - 13 [F] = 30 [C] - 7.22 [DELTAC] = 22.77 [C]
>> 22.8 [C]
```



 - A temperature cannot be subtracted from a tempertaure difference

```
a = variable(30,'DELTAC')
b = variable(20,'C')
a - b
>> ValueError: You tried to subtract a temperature from a temperature differnce. This is not possible.
```
