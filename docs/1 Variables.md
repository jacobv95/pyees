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

## scalar variables vs array variables
The return type is either "scalarVariable" or "arrayVariable" when initializing a variable. This depends on the supplied value is a list-like object or not. The arrayVariable is basically a list which holds scalarVariables. 

The array variable also includes some array methods:
 - __setitem__
 - __len__
 - __getitem__

It should be noted, that the method __setitem__ changes some of the scalarVariables in the arrayVariable. This will have consequences for uncertanty propagations.


Three methods are defined which returns the same output. These three methods are created in order to show how changing a scalarVariable in an arrayVariable will affect the uncertanty propagation.

``` 
def arrayMethod():
    ## create a variable
    a = variable([1,2,3], 'm', [0.1, 0.2, 0.3])

    ## use the variable 'a' to create the variable 'b'
    b = a**2

    ## change the variable a using the __setitem__ method
    a[1] = variable(5,'m',0.5)

    ## modify 'b' using 'a'
    b *= a
    return b.value, b.uncert



def scalarMethod1():
    ## Lets calculate each term on by one using scalarVariables
    a0 = variable(1,'m', 0.1)
    a1 = variable(2,'m', 0.2)
    a2 = variable(3,'m', 0.3)

    ## use the variablea a0, a1 and a2 to create the variables b0, b1 and b2
    b0 = a0**2
    b1 = a1**2
    b2 = a2**2

    ## change the variable a1
    a1 = variable(5, 'm', 0.5)

    ## modify b0, b1 and b2 using a0, a1 and a2
    b0 *= a0
    b1 *= a1
    b2 *= a2

    ## return values of b0, b1 and b2 wrapped in a numpy array
    ## this makes comparing the result from the arrayVariable-method and the scalarVariable-method easy
    values = np.array([elem.value for elem in [b0,b1,b2]])
    uncerts = np.array([elem.uncert for elem in [b0,b1,b2]])
    return values, uncerts


def scalarMethod2():
    ## lets calculate each term in a slightly faster way
    ## as the scalarVariables a0 and a2 are not overwritten, then b0 and b2 is simply a0**3 and b0**3
    ## furthermore, we will not overwrite a1 but rather create two seperate variables
    a0 = variable(1,'m', 0.1)
    a1_1 = variable(2,'m', 0.2)
    a1_2 = variable(5, 'm', 0.5)
    a2 = variable(3,'m', 0.3)

    b0 = a0**3
    b1 = a1_1**2 * a1_2
    b2 = a2**3

    ## return values of b0, b1 and b2 wrapped in a numpy array
    ## this makes comparing the result from the arrayVariable-method and the scalarVariable-method easy
    values = np.array([elem.value for elem in [b0,b1,b2]])
    uncerts = np.array([elem.uncert for elem in [b0,b1,b2]])
    return values, uncerts



## run and compare the three methods
print(arrayMethod())
>> (array([ 1., 20., 27.]), array([0.3       , 4.47213595, 8.1       ]))

print(scalarMethod1())
>> (array([ 1., 20., 27.]), array([0.3       , 4.47213595, 8.1       ]))

print(scalarMethod2())
>> (array([ 1., 20., 27.]), array([0.3       , 4.47213595, 8.1       ]))

## All methods return the same result
```


