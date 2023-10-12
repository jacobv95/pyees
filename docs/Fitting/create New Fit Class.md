# Crate new fit class


Pyees includes a method to create a new fitting class.

```
crateNewFitClass(func: Callable, funcNameFunc: Callable, getVariableUnitsFunc: Callable, nParameters: int) -> fit
```

The method takes 4 arguments, 3 of which has to be callable. These 3 callables has to have the following structure

```
func(coefficients: List[variables], x: variable) -> double
funcNameFunc(coefficients: List[variables]) -> str
getVariableUnitsFunc(xUnit: variableUnit, yUnit: variableUnit) -> List[str]
```


Here the method "func" is the function of the new fit class, which takes the function coefficients and the function inputs as arguments. The methods "funcNameFunc" has the purpose of creating a string representing the fit. It takes the function coefficients as an argument. Finally, the methods "getVariableUnitsFunc" declares the unit of the coefficients of the function. It takes the unit of the x variable and the unit of the y variable as arguments. The type of these arguments are "unit", and they can be used in multiplication, division and exponensiation, which all returns a string.


## Example

```
## define the second order polynomial given a list of the coefficients and the function input "x"
def func(coefficients, x):
    a = coefficients[0]
    return a*x**2


## define a function used to print the function given a list of the coefficients
def funcName(coefficients):
    a = coefficients[0]
    return f'a*x, a={a}'

## define the units of the coefficients given the unit of the x variable and the unit of the y variable
def getVariableUnitsFunc(xUnit, yUnit):
    return [yUnit / (xUnit**2)]

## define the number of parameters
nParameters = 1

## create a new class
newFit = crateNewFitClass(func, funcName, getVariableUnitsFunc, nParameters)

## define x and y variables to fit the class to
x = variable([1,2,3], 'm')
y = variable([2,4,6], 'C')

## create an instance of the new fit class given the x and y variables
f = newFit(x,y)

## print the fit
print(f)
>> a*x, a=0.9 +/- 0.2 [DELTAC/m2], $R^2 = 0.95194$
```