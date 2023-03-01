
# Variables

## Example
We would like to calculate <img src="https://render.githubusercontent.com/render/math?math=C=A\cdot B"> given to measurements <img src="https://render.githubusercontent.com/render/math?math=A=12.3"> and <img src="https://render.githubusercontent.com/render/math?math=B=35.1"> both with uncertanties <img src="https://render.githubusercontent.com/render/math?math=\sigma_A=2.6"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_B=8.9">. The value of <img src="https://render.githubusercontent.com/render/math?math=C"> is simply computed as <img src="https://render.githubusercontent.com/render/math?math=C=12.3\cdot 35.1 = 431.73">. The uncertanty of <img src="https://render.githubusercontent.com/render/math?math=C"> is determined using the following equation

<img src="https://render.githubusercontent.com/render/math?math=\sigma_C = \sqrt{  \left(\frac{\partial C}{\partial A} \sigma_A\right)^2 %2B \left(\frac{\partial C}{\partial B} \sigma_B\right)^2 %2B 2\frac{\partial C}{\partial A}\frac{\partial C}{\partial B}\sigma_{AB}}">

Here <img src="https://render.githubusercontent.com/render/math?math=\sigma_{AB}"> is the covariance of the measurements of <img src="https://render.githubusercontent.com/render/math?math=A"> and <img src="https://render.githubusercontent.com/render/math?math=B">. We will assumed that this is zero for now. The equation above is evaluated as follows:

<img src="https://render.githubusercontent.com/render/math?math=\sigma_C = \sqrt{  \left(B \sigma_A\right)^2 %2B \left(A\sigma_B\right)^2 } = \sqrt{  \left(35.1 \cdot 2.6\right)^2 %2B \left(12.3 \cdot 8.9\right)^2 } = \sqrt{(91.26)^2 %2B (109.47)^2}=142.52">

You can create a variables as follows

```
var = variable(val: float | list | ndarray, unit: str, uncert=None: float | list | ndarray | None, nDigits = 3: int)
```

 - val is the value of the variable
 - unit is the unit of the variable
 - uncert is the uncertanty of the variable
 - nDigits is the number of significant digits used to print the variable, if the uncertanty is None or 0

## Units
The unit is used to determine which computations can be performed:
 - Two variables can be added together or subtracted from each other if their units are identical
 - Any two units can be multiplied or divided
 - Exponents cannot have any units
 - A variable with a unit can be raised to an integer power
 - The n'th root of a variable can be taken if the exponent of the unit of the variable is divisible by n

The denominator and the numerator of the unit is seperated using a dash (/).
The units in the denominator or numerator are sperated using a hyphen (-)


The following units are known:
 - unitless: 1, '', %
 - force: N
 - mass: g
 - Energy: J
 - power: W
 - pressure: Pa, bar
 - Temperature: K, C, F, °C, °F (ASCII 0176)
 - Temperature difference: DELTAK, DELTAC, DELTAF
 - time: s, min, h, yr
 - volume: m3, L
 - length: m, ly, Å
 - current: A
 - Voltage: V
 - Angles: rad, ° (ASCII 0176)


The following prefixes are known:
 - T: 1e12
 - G: 1e9
 - M: 1e6
 - k: 1e3
 - h: 1e2
 - d: 1e-1
 - c: 1e-2
 - m: 1e-3
 - µ: 1e-6 (ASCII 230)
 - n: 1e-9
 - p: 1e-12

## exponents
The exponent will always apply to the unit AND the prefix. The unit 'mm3' is interpreted as "cubic millimeters" and not "milli cubicmeters". 

Furhtermore, 1 kilometer multiplied with 1 meter returns 1 kilometer-meter. This is beacuse there is not prefix, x, in the known prefixes, such that 
<img src="https://render.githubusercontent.com/render/math?math=\left(xm\right)^2 \quad \rightarrow \quad x^2 = 1000 \quad \rightarrow \quad x = 31.62">. However, the result, 1 kilometer-meter, can be converted in to square meters using the convert method.


### examples
 - milli Liters per minute:               'mL/min'
 - Cubicmeter-kilogram per second:  'm3-kg/s


## Printing
The uncertanty is printed with one significant digit. The measurement is printed with the same number of decimal places as the uncertanty. This means that if the uncertanty is a factor of 10 larger than the measurement, then the variable is printed as zero.

### Examples
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


## Convert
A variable can be converted to another unit using the convert method.

```
variable.convert(unit: str)
```



## operators
 - add
 - subtract
 - multiply
 - divide
 - power
 - np.sqrt
 - np.exp
 - np.sin
 - np.cos
 - np.tan
 - np.mean
 - np.min
 - np.max
 - np.log
 - np.log10
 - np.sqrt

