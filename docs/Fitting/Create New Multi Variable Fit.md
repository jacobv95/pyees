# Crate new multi variable fit class



Pyees includes a method to create a new single variable fitting class.

```
crateNewFitClass(func: Callable, funcNameFunc: Callable, getVariableUnitsFunc: Callable, nParameters: int) -> fit
```

The method takes 4 arguments, 3 of which has to be callable. These 3 callables has to have the following structure

```
func(coefficients: List[variables], x: list[variable]) -> variable
funcNameFunc(coefficients: List[variables]) -> str
getVariableUnitsFunc(xUnit: variableUnit, yUnit: variableUnit) -> List[str]
```


Here the method "func" is the function of the new fit class, which takes the function coefficients and the function inputs as arguments. The methods "funcNameFunc" has the purpose of creating a string representing the fit. It takes the function coefficients as an argument. Finally, the methods "getVariableUnitsFunc" declares the unit of the coefficients of the function. It takes the unit of the x variable and the unit of the y variable as arguments. The type of these arguments are "unit", and they can be used in multiplication, division and exponensiation, which all returns a string.
