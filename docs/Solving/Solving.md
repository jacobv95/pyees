# Solving

In pyees it is possible to solve a system of equations.

```
x = solve(func: callable, x0: variable | list[variable], *args, bound = None: callable | list[list[float]], parametric = None: varialbe | list[variable], **kwargs)
```


## func
The function has to accepct one input for each variable in the 'x0' argument. Furthermore, the function has to accept one input for each variable in the 'parametric' argument.

The function supplied to pyees.solve has to be callable and have a special return type. The function has to return a list of "equations" or a single "equation". The "equation" is a list of two elements, where both elements are variables objects. When solving the equation system the two sides of the "equation" will be equal.

## x0
The initial guess.

## bounds
The bounds of the variable. This can either be callable or non callable.

### non callable bounds
If the bounds are not callable, then they are passed in to scipy.optimize.minimize as they are. Therefore, the bounds has to fulfil the requirements from scipy.optimize.minimize

### callable bounds
If the bounds are callable, then the bounds are used as the callback of the scipy.optimize.minimize method. This means that the bounds are being called after each iteration of the minimization algortihm.

The bounds has to accepct one input for each variable in x0.

The bounds has to return a list of "inequalities". An "inequality" is a list of 3 variables, where the second element (the middle element) has to be one of the variables in x0. The first and last elements (the left and right elements) are the lower and upper most acceptable value, respectively. 

The lower and upper most acceptable values can be a function of the input. Therefore, this allows for nonlinear bounds.

The bounds does not have to return 1 "inequality" for each variable in x0.

## parametrics
The argument 'parametric' is used for solving a system of equations while varying some inputs for the system. If the parametric variables are used, then the method 'solve' will return array variables with the same length as the parametric variables

## Example 1
```
from pyees import variable, solve

## solve:
##      x^2 + y = 4
##      y^2 + x = -2

def func(x,y):
    equation_1 = [x**2 + y, variable(2.3)]
    equation_2 = [y**2 + x, variable(-1.3)]
    equations = [equation_1, equation_2]
    return equations

x0 = [variable(1), variable(2)]

x,y = solve(func, x0)

print(x)
>> -1.41 

print(y)
>> 0.325 

out = func(x,y)
for equations in out:
    print(*equations)
>> 2.3  2.3
>> -1.3  -1.3
```


## Example 2
```
from pyees import variable, solve

## solve:
##      x^2 + y = a
##      y^2 + x = -2
## for a = [2,3,4,5]

def func(x, y, a):
    equation_1 = [x**2 + y, a]
    equation_2 = [y**2 + x, variable(-1.3)]
    equations = [equation_1, equation_2]
    return equations

x0 = [variable(1), variable(2)]
A = variable([2,3,4,5])
x,y = solve(func, x0, parametric = A)

print(x)
>> [-1.340852060929057, -1.57377450896645, -1.8122578611093851, -2.035312470275505] 

print(y)
>> [0.20211576369679993, 0.5232338053202181, 0.7157214488736241, 0.857503188812464]
    
out = func(x,y,A)
for equation in out:
    print(equation)
>> [[2.0000000129944997, 3.0000000103928093, 4.000000004026387, 5.000000040471442] , [2.0, 3.0, 4.0, 5.0] ]
>> [[-1.3000012789943165, -1.3000008939365741, -1.3000006687316255, -1.3000007514519607] , -1.3 ]
```



