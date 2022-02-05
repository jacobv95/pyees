# pyees
python package to replace EES



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

Using pyees you can create a system comprised of subsystems. Say you have a pump connected to a valve. The pump and the valve can be modelled as two independent systems, both with two variables: flow and dP. The entire system is modelled in order to connect the pump and the valve. 


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
