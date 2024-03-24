# Prop

A wrapper has been built around the pyfluids library. The wrapper creates a dataUncert variable of the property and calculates the uncertanties.

```
out: variable = prop(property: str, fluid: str | list[str], C = None | variable | list[variable], **state)
```

 - The arguments 'property' and 'fluid' describe the desired property of the desired fluid
 - C is the concentration. This is used for some brines. Furthermore it is used in mixtures. Mixtures are made by parsing a list of fluids as the argument 'fluid', and parsing a list of variables to the 'concentration' argument which corresponds to the concentrations of the mixture
 - state describes the state of the fluid. Arguments such as 'temperature', 'pressure', 'relative_humidity' is parse as keyword arguments. Some common arguments can be abriviated. This works for 'temperature', 'pressure' and 'relative_humidity'



## Example 1
```
from pyees import variable, prop
T = variable(30,'C',0.1)
P = variable(1,'bar', 0.01)
mu = prop('dynamic_viscosity', 'water', T=T, P=P)
print(mu)
>> 0.000797 +/- 2e-06 [Pa-s]
```

```
from pyees import variable, prop
C = [variable(60, '%'), variable(40, '%')]
p = variable(200e3, 'Pa')
T = variable(4, 'C')
rho = prop('density', ['water', 'Ethanol'], C = C , P = P, T = T)
print(rho)
>> 883.3922771627963 [kg/m3]
```

All fluid from [pyfluids](https://github.com/portyanikhin/PyFluids) are supported

The following properties are included in this project
 - compressibility          (1)
 - conductivity             (W/m-K)
 - critical_pressure        (Pa)
 - critical_temperature     (C)
 - density                  (kg/m3)
 - dynamic_viscosity        (Pa-s)
 - enthalpy                 (J/kg)
 - entropy                  (J/kg-K)
 - freezing_temperature     (C)
 - internal_energy          (J/kg)
 - kinematic_viscosity      (m2/s)
 - max_pressure             (Pa)
 - max_temperature          (C)
 - min_pressure             (Pa)
 - min_temperature          (C)
 - molar_mass               (kg/mol)
 - prandtl                  (1)
 - pressure                 (Pa)
 - quality                  (%)
 - sound_speed              (m/s)
 - specific_heat            (J/kg-K)
 - specific_volume          (m3/kg)
 - surface_tension          (N/m)
 - temperature              (C)
 - triple_pressure          (Pa)
 - triple_temperature       (C)
 - dew_temperature          (C)
 - humidity                 (1)
 - partial_pressure         (Pa)
 - relative_humidity        (%)
 - wet_bulb_temperature     (C)


