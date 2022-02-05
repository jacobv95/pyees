# pyees
python package to replace EES. The main issue with EES is that large systems become very difficult to read. This is becuase the EES language does not require any order of the equations. The pyees package adresses this by enabeling the system to be split up in to subsystems.



# Basic functionality

pyees works in the same way as EES: You set up n equations of n variables which are solved simulatniously.


```
from pyees import *
system = System()                   # create a system of equations
system.A = variable(10, '')         # create a variable without a unit and an initial guess of 10
system.B = variable(5, '')         # create a variable without a unit and an initial guess of 5

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

>> System.A     11.000 
>> System.B     5.500 
```


# Subsystems

Using pyees you can create a system comprised of subsystems. Say you have a pump connected to a valve. The pump and the valve can be modelled as two independent systems, both with two variables: flow and dP. The entire system is modelled in order to connect the pump and the valve. Subsystem can of course be created in seperate files in order to increase readability. 


```
from pyees import *

# Valve
valve = System()
valve.flow = variable(15, 'L/min')
valve.dP = variable(1, 'bar')

def f_valve(self):
    eq = []
    # the pressure loss in a valve is modelled as follows
    # dP = kv * flow ^ 2
    eq.append((self.dP, variable(8.65, 'min-bar/L') * self.flow**2))
    return eq

valve.addEquations(f_valve)


# Pump
pump = System()
pump.flow = variable(20, 'L/min')
pump.dP = variable(200, 'bar')

def f_pump(self):
    eq = []
    # the pressure increase of the pump is modelled as follows
    # dP = a - b * flow ^ 2
    eq.append((self.dP, 12345 - variable(0.5, 'min-bar/L') * self.flow**2))
    return eq

pump.addEquations(f_pump)


# System
sys = System()
sys.pump = pump
sys.valve = valve

def f_sys(self):
    eq = []
    # the pressure loss of the valve is set equal to the pressure increase from the pump
    # the flow through the pump is set equal to the flow through the valve
    # this will return the steady state flow and pressure 
    eq.append((self.pump.dP, self.valve.dP))
    eq.append((self.pump.flow, self.valve.flow))
    return eq

sys.addEquations(f_sys)

sys.solve()
sys.printVariables()

>> System.pump.flow      28.452 L/min
>> System.pump.dP        0.117 bar

>> System.valve.flow     28.452 L/min
>> System.valve.dP       0.117 bar
```




# Units

## Basic usage
The variables have units. The unit can only have a single forwards slash. However, they can have as many dashes as needed.

Examples:
1 Cubic meter:
variable(1, 'm3')

1 kg per cubic meter:
variable(1,'kg/m3')

1 bar of pressure lossper L/min = 1 bar / (L/min) = 1 bar*min / L
variable(1, 'bar-min/L')


## Addition and subtraction
As all variables are converted to the SI unit system, only the SI units has to be identical

```
a = variable(1, 'L/min')
b = variable(1, 'm3/h')
c = a + b

print(a,b,c)
>> a = variable(1.6666666666666667e-05, m3/s)
>> b = variable(0.0002777777777777778, m3/s)
>> c = variable(0.00029444444444444445, m3/s)
```


The variables can be converted back in to their original unit.


```
a.convertToOriginalUnit()
b.convertToOriginalUnit()
c.convertToOriginalUnit()

print(a,b,c)
>> a = variable(1.0, L/min)
>> b = variable(1.0, m3/h)
>> c = variable(0.00029444444444444445, m3/s)
```

Notice, that c is not conveted from the SI unit syste. This is because it was created from a and b, which has differing units. Therefore the units of c was set as the SI unit of a and b.


## Multiplication and division

New variables enherit the original units even though all variables are conveted to the SI unit system

```
A = variable(1, 'kJ/kg-K')
B = variable(1, 'kg/m3')
C = A * B

print(a,b,c)
>> a = variable(1000.0, m2/K-s2)
>> b = variable(1.0, kg/m3)
>> c = variable(1000.0, kg/K-s2-m)
```


The variables can be converted back in to their original unit.


```
a.convertToOriginalUnit()
b.convertToOriginalUnit()
c.convertToOriginalUnit()

print(a,b,c)
>> a = variable(1.0, kJ/kg-K)
>> b = variable(1.0, kg/m3)
>> c = variable(1.0, kJ/K-m3)
```



# Coolprob
A wrapper has been made around CoolProp.CoolProp.PropsSI. This returns a pyees variable with the correct unit.

```
# get density of water at 300 K and 101325 Pa.
A = prop('D','T',300,'P',101325,'Water')
print(A)
>> variable(996.5569352652021, kg/m3)
```


## Writing variables on diagrams
One benefit of EES is the ability to print the variables on a diagram. This can be achived in pyees using the function writeVariablesOnDiagram. This function takes a single argument, which is an existing .pdf file. All variables, which has been given the argument pos will be printed on the .pdf file.




# Functions

## System
A = System()

A.addEquation(f)
 - f is a function which takes self as a single argument

A.solve()

A.printVariables()

A.writeVariablesOnDiagraom(existingPDF, font='Helvetica', fontSize=8)
 - existingPDF is the path to an existing pdf file
 - font is a string
 - fontSize is an integer


## Variable
A = variable(value, unit, upperBound=np.inf, lowerBound=-np.inf, nDigits=3, fix=False, pos=None)
 - value is the initial guess of the variable.
 - unit is the original unit of the variable.
 - upperBound is the upper bound when solving the system
 - lowerBound is the lower bound when solving the system¨
 - nDigits is the number of significant digits shown when printing
 - fix is a boolean. If fix is true the variable is defined and is not solved for when the system is solved.
 - pos is the position the variable will be printed at in the existing pdf, when System().writeVariablesOnDiagraom is used. The argument has to be a list of two elements with x,y coordinates in the .pdf file.


## prop
A = prop(property, paramenterA, valueA, paramenterB, valueB, fluid)
 - property is the material property to be found. This can be either 'D', 'V', 'C' or 'H'
 - parameterA is a string indicating the first parameter given to coolprop
 - valueA is the value for the first parameter given to coolprop
 - parameterB is a string indicating the second parameter given to coolprop
 - valueB is the second for the first parameter given to coolprop
 - fluid is a string indicating the fluid


## Components

### Pipe
A = Pipe(d, L, epsilon)
 - d is the diameter of the pipe
 - L is the length of the pipe
 - epsilon is the relative roughness of the pipe

dP = A.curve(flow, rho, mu)
 - flow is the flow of the pipe
 - rho is the density of the fluid in the pipe
 - mu is the viscosity of the fluid in the pipe
 - dP is the pressure loss through the pipe

### Pump
A = Pump(datasheet, flowName, pressureName, sheetNr=1, kind='linear')
 - datasheet is an .xlsx file with the pump curve of the pump. The file has to have a coloumn with flow and a coloumn with pressure. Each coloumn has to have a nave in row 1 and a unit in row 2. All other rows has to be numerical.
 - flowName is the name of the coloumn with the flow values
 - pressureName is the name of the coloumn with the pressure values
 - sheetNr is the sheet of the .xslx file with the flow and pressure information
 - kind is the interpolation method.

dP = A.curve(flow)
 - flow is the flow of the pump
 - dP is the pressure increase across the pump

### Valve
A = Valve(dP, dPUnit, flow, flowUnit)
 - dP is the pressure loss defining the kv value of the valve
 - dPUnit is the unit of the pressure loss defining the kv value of the valve
 - flow is the flow defining the kv value of the valve
 - flowUnit is the unit of the flow defining the kv value of the valve

dP = A.curve(flow, opening = 100)
 - flow is the flow through the valve
 - opening is the opening degree of the valve
 - dP is the pressure loss through the valve