# Variables


Variables are the backbone of pyees. A variable has a value, a unit and an uncertanty. The value and the uncertanty can be either a float or an array. However they have to have the same size. The unit is a string.

```
var = variable(val: float | list | ndarray, unit: str|None, uncert=None: float | list | ndarray | None)
```


## Printing
The uncertanty is printed with one significant digit. The measurement is printed with the same number of decimal places as the uncertanty. This means that if the uncertanty is a factor of 10 larger than the measurement, then the variable is printed as zero.

```
import pyees as pe
print(pe.variable(12.3,'m',0.01))
>> 12.30 +/- 0.01 [m]

print(pe.variable(12.34,'m',0.1))
>> 12.3 +/- 0.1 [m]

print(pe.variable(1234,'m',16))
>> 1230 +/- 20 [m]

print(pe.variable(12.34,'m',16))
>> 10 +/- 20 [m]

print(pe.variable(1.234,'m',16))
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
numpy.linspace(a,b)
```




## Conversion of the variable
A variable can be converted in to any other unit with the same SI base unit using the convert-method

```
import pyees as pe
a = pe.variable(10,'L/min')
a.convert('m3/h')
>> 0.6 [m3/h]
```



## Uncertanty propagation

If a parameter 'c' is calculated from two parameters: 'a' and 'b', then the uncertanty of 'c' depends on the uncertanty of 'a' and 'b' in the following way.

```math
\sigma_C = \sqrt{  \left(\tfrac{\partial C}{\partial A} \sigma_A\right)^2 + \left(\tfrac{\partial C}{\partial B} \sigma_B\right)^2 + 2\tfrac{\partial C}{\partial A}\tfrac{\partial C}{\partial B}\sigma_{AB}}
```
Here $\sigma_{AB}$ is the covariance of the measurements of $A$ and $B$.

When using variables from pyees these calculations happen automatically.


## scalar variables vs array variables
The return type is either "scalarVariable" or "arrayVariable" when initializing a variable. This depends on the supplied value is a list-like object or not. The arrayVariable is basically a list which holds scalarVariables. 

The array variable also includes some array methods:
```
import pyees as pe
a = pe.variable([1,2,3], 'm')

a[0]
>> 1 [m]

len(a)
>> 3

a[2] = pe.variable(5, 'm')
print(a)
>> [1,2,5] [m]

a.pop(1)
print(a)
>> [1,5] [m]
```



## significant contributors

The significant contributors to the uncertanty of a variable 'c' can be found using the getUncertantyContributors() method

```
variables, significance = variable.getUncertantyContributors()
```

 - significance is an array variable which contains the significance of the elements in the output 'variables'.
 - variables in a list of lists. Each element of the list is it self a list which describes where what contributes to the significance. An element of the output 'variables' which has a length of 1 correspondons to the contribution from the uncertanty of that single element in the list. An element of the output 'variables' which has a length of 2 corresponds to the constribution from the covariance between the two elements in the list.

 The significance is defined as the ratio of the magnitude of the variance contribution from each contributor to the uncertanty of the parameter c

```math
s_i = \frac{\left(\frac{\partial c}{\partial x_i}\sigma_{x_j}\right)^2}{\sum_j^n \left(\left(\frac{\partial c}{\partial x_j} \sigma_{x_j}\right)^2\right) + \mathop{\sum_k^n \sum_l^n}_{k\neq l} \left| \frac{\partial c}{\partial x_k}\frac{\partial c}{\partial x_l} \sigma_{x_l, x_k} \right|}
```

### Example 1
```
a = variable(23, 'L/min', 5.7)
b = variable(11, 'mbar', 1.1)
c = a * b
variables, significance = c.getUncertantyContributors()

print(variables)
>> [[23 +/- 6 [L/min]], [11 +/- 1 [mbar]]]

print(significance)
>> [85.99788247750134 [%], 14.002117522498677 [%]]
```

### Example 2
```
a = variable(23, 'L/min', 5.7)
b = variable(11, 'mbar', 1.1)
a.addCovariance(b, -0.02, 'mbar-L/min')
c = a * b
variables, significance = c.getUncertantyContributors()

print(variables)
>> [[23 +/- 6 [L/min]], [11 +/- 1 [mbar]], [23 +/- 6 [L/min], 11 +/- 1 [mbar]]]

print(significance)
>> [84.13531710532511 [%], 13.698855878337024 [%], 2.1658270163378694 [%]]
```

