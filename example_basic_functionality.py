from pyees import *


system = System()                   # create a system of equations
system.A = variable(1, '')         # create a variable without a unit and an initial guess of 10
system.B = variable(2, '')         # create a variable without a unit and an initial guess of 5

# create a function to evaluate all equations in the system.
# The function has to take "self" as an argument
# The function has to return a list of equations
# Each equation has to be a list-like-object.
# The seperations (,) in the equation can be read as equal signs


def f(self):
    listOfEquations = []

    # A = 2*B = 11
    equation = (self.A, 2 * self.B, 11)
    listOfEquations.append(equation)

    return listOfEquations


# parse the function f to the system
system.addEquations(f)
system.solve()
system.printVariables()
