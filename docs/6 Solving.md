# Solving

Solving in pyees is done using the scipy.optimize.minimize method. This method takes a lot of inputs. The behaviour of 3 of the inputs to scipy.optimize.minimze has been modified: func, x0, bounds


## func
The function has to accepct one input for each variable in x0.

The function supplied to pyees.solve has to be callable and have a special return type. The function has to return a list of "equations" or a single "equation". The "equation" is a list of two elements, where both elements are variables objects. When solving the equation system the two sides of the "equation" will be equal.

## x0
The initial guess has to be a list of variables.

## bounds
Bounds can be defined in two ways.

### non callable bounds
If the bounds are not callable, then they are passed in to scipy.optimize.minimize as they are. Therefore, the bounds has to fulfil the requirements from scipy.optimize.minimize

### callable bounds
If the bounds are callable, then the bounds are used as the callback of the scipy.optimize.minimize method. This means that the bounds are being called after each iteration of the minimization algortihm.

The bounds has to accepct one input for each variable in x0.

The bounds has to return a list of "inequalities". An "inequality" is a list of 3 variables, where the second element (the middle element) has to be one of the variables in x0. The first and last elements (the left and right elements) are the lower and upper most acceptable value, respectively. 

The lower and upper most acceptable values can be a function of the input. Therefore, this allows for nonlinear bounds.

The bounds does not have to return 1 "inequality" for each variable in x0.

## Example
´´´
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
´´´