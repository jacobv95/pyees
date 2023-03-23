# Variables


Variables are the backbone of pyees. A variable has a value, a unit and an uncertanty. The value and the uncertanty can be either a float or an array. However they have to have the same size. The unit is a string.

```
var = variable(val: float | list | ndarray, unit: str|None, uncert=None: float | list | ndarray | None)
```


## Printing
The uncertanty is printed with one significant digit. The measurement is printed with the same number of decimal places as the uncertanty. This means that if the uncertanty is a factor of 10 larger than the measurement, then the variable is printed as zero.

```
print(variable(12.3,'m',0.01))
>> 12.30 +/- 0.01 [m]

print(variable(12.34,'m',0.))
>> 12.3 +/- 0.1 [m]

print(variable(1234,'m',16))
>> 1230 +/- 20 [m]

print(variable(12.34,'m',16))
>> 10 +/- 20 [m]

print(variable(1.234,'m',16))
>> 0 +/- 20 [m]
```


## Operators
The variable object supports the following operators
```
a = variable(1)
b = variable(1)

a + b
a - b
a * b
a / b
a ** b
max(a)
min(a)
abs(a)
a < b
a <= b
a > b
a >= b
a == b
a != b
numpy.sqrt(a)
numpy.exp(a)
numpy.log(a)
numpy.log10(a)
numpy.sin(a)
numpy.cos(a)
numpy.tan(a)
numpy.mean(a), numpy.mean([a,b])
numpy.min(a), numpy.min([a,b])
numpy.max(a), numpy.max([a,b])
a[0]
a.len()
a.append(b)
```



## Units
The unit is a string. The unit uses 4 special characters: the hyphen "-", the slash "/" and the open parenthesis "(", and the close parenthesis ")". The denominator and numenator of the unit is seperated using the slash. All units multiplied with each other are seperated using the hyphen. Furthermore, the parenthesis is used to make the unit more readable. The parenthesis are unpacked when creating a variable. This means that the parenthesis are no longer visible when printing the variable.

The following units are known:
 - unitless: 1, '', %, None
 - force: N
 - mass: g
 - Energy: J
 - power: W
 - pressure: Pa, bar
 - Temperature: K, C, F
 - Temperature difference: DELTAK, DELTAC, DELTAF
 - time: s, min, h, yr
 - volume: m3, L
 - length: m, ly, Å
 - current: A
 - Voltage: V
 - Angles: rad, deg
 - Resistance: ohm
 - kinematic viscosity: St
 - logarithmic units: B, Np, oct, dec


The following prefixes are known:
 - T: 1e12
 - G: 1e9
 - M: 1e6
 - k: 1e3
 - h: 1e2
 - da: 1e1
 - d: 1e-1
 - c: 1e-2
 - m: 1e-3
 - mu: 1e-6
 - n: 1e-9
 - p: 1e-12

Any combination of unit and prefix can be used.

The unit is used to determine which computations can be performed on the variable:
 - Two variables can be added together or subtracted from each other if their units are identical
 - Any two units can be multiplied or divided
 - Exponents cannot have any units
 - A variable with a unit can be raised to an integer power
 - The n'th root of a variable can be taken if the exponent of the unit of the variable is divisible by n


## Conversion of the variable
A variable can be converted in to any other unit with the same SI base unit using the convert-method

```
a = variable(10,'L/min')
a.convert('m3/h')
>> 0.6 [m3/h]
```


## exponents
The exponent will always apply to the unit AND the prefix. The unit 'mm3' is interpreted as "cubic millimeters" (0.001 [m] * 0.001 [m] * 0.001 [m]) and not "milli cubicmeters" (0.001 * (1 [m] * 1 [m] * 1 [m])). 

Furhtermore, 1 kilometer multiplied with 1 meter returns 1 kilometer-meter. This is beacuse there is not prefix, x, in the known prefixes, such that 
<img src="https://render.githubusercontent.com/render/math?math=\left(xm\right)^2 \quad \rightarrow \quad x^2 = 1000 \quad \rightarrow \quad x = 31.62">. However, the result, 1 kilometer-meter, can be converted in to square meters using the convert method.



## Uncertanty propagation

If a parameter 'c' is calculated from two parameters: 'a' and 'b', then the uncertanty of 'c' depends on the uncertanty of 'a' and 'b' in the following way.

<img src="https://render.githubusercontent.com/render/math?math=\sigma_C = \sqrt{  \left(\frac{\partial C}{\partial A} \sigma_A\right)^2 %2B \left(\frac{\partial C}{\partial B} \sigma_B\right)^2 %2B 2\frac{\partial C}{\partial A}\frac{\partial C}{\partial B}\sigma_{AB}}">

Here <img src="https://render.githubusercontent.com/render/math?math=\sigma_{AB}"> is the covariance of the measurements of <img src="https://render.githubusercontent.com/render/math?math=A"> and <img src="https://render.githubusercontent.com/render/math?math=B">.

When using variables from pyees these calculations happen automatically.


## array methods
The arraymehtods such as append, __setitem__, __len__, etc. are only avaible for arrayVariables. These are variables initialized with a list-like-object.

Any arraymethod which alters the variable affectivle creates a new variable. The variable is still the same object, however any other variable sees the variable as a new variable. This requires an example:

```
## create a variable
a = variable([1,2,3], 'm', [0.1, 0.2, 0.3])

## use the variable 'a' to create the variable 'b'
b = a**2

## change the variable a using the __setitem__ method
a[1] = variable(5,'m',0.5)

## modify 'b' using 'a'
b *= a
print(b)
>> [1.0, 20, 27] +/- [0.2, 4, 6] [m3]


## create a new variable 'A1' which is identical to 'a'
A1 = variable([1,2,3], 'm', [0.1, 0.2, 0.3])

## create a new variable 'A2' which is identical to 'a' after the variable 'a' has been modified using the __setitem__ method
A2 = variable([1,5,3], 'm', [0.1, 0.5, 0.3])

## use the variable 'A1' and 'A2' to create the variable 'B'
B = A1**2 * A2
print(B)
>> [1.0, 20, 27] +/- [0.2, 4, 6] [m3]
```

If the variable 'a' had not been treated as a new variable, then the uncertanty of 'b' would have been different from the uncertanty of 'B'. This is because the variable productrule would have to have been used when multiplying 'b' with 'a' again after the __setitem__ method had been used. However, this is handled internally, and the variable 'a' acts as a new variable on the variable 'b' after the __setitem__ method has been used on 'a'.

This is valid for all arraymethods, which alteres a variable

