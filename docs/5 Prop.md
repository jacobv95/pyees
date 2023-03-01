# Prop

A wrapper has been built around the pyfluids library. The wrapper creates a dataUncert variable of the property and calculates the uncertanties.

```
from dataUncert import variable, prop
T = variable(30,'C',0.1)
P = variable(1,'bar', 0.01)
mu = prop('mu', 'water', T, P)
print(mu)
>> 0.000797 +/- 2e-06 [Pa-s]
```

The following fluids are included in this project
 - water
 - MEG (ethylene-glycol water)
 - Air

The following properties are included in this project
 - 'rho' (density)
 - 'cp' (specific heat capacity)
 - 'mu' (kinematic viscosity)


