
# Units
The unit is a string. The unit uses 4 special characters: the hyphen "-", the slash "/" and the open parenthesis "(", and the close parenthesis ")". The denominator and numenator of the unit is seperated using the slash. All units multiplied with each other are seperated using the hyphen. Furthermore, the parenthesis is used to make the unit more readable. The parenthesis are unpacked when creating a variable. This means that the parenthesis are no longer visible when printing the variable.

The following units are known:
 - unitless: 1, '', %, None
 - force: N
 - mass: g
 - Energy: J
 - power: W
 - pressure: Pa, bar
 - Temperature: K, C, F
 - Temperature difference: DELTAK, DELTAC, DELTAF
 - time: s, min, h, yr
 - volume: m3, L
 - length: m, ly, Å
 - current: A
 - Voltage: V
 - Angles: rad, deg
 - Resistance: ohm
 - kinematic viscosity: St
 - logarithmic units: B, Np, oct, dec


The following prefixes are known:
 - T: 1e12
 - G: 1e9
 - M: 1e6
 - k: 1e3
 - h: 1e2
 - da: 1e1
 - d: 1e-1
 - c: 1e-2
 - m: 1e-3
 - mu: 1e-6
 - n: 1e-9
 - p: 1e-12

Any combination of unit and prefix can be used.

The unit is used to determine which computations can be performed on the variable:
 - Two variables can be added together or subtracted from each other if their units are identical
 - Any two units can be multiplied or divided
 - Exponents cannot have any units
 - A variable with a unit can be raised to an integer power
 - The n'th root of a variable can be taken if the exponent of the unit of the variable is divisible by n



## exponents
The exponent will always apply to the unit AND the prefix. The unit 'mm3' is interpreted as "cubic millimeters" (0.001 [m] * 0.001 [m] * 0.001 [m]) and not "milli cubicmeters" (0.001 * (1 [m] * 1 [m] * 1 [m])). 

Furhtermore, 1 kilometer multiplied with 1 meter returns 1 kilometer-meter.

