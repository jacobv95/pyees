# Logarithmic units


Arithmatic on logarithmic units work as any other unit.

```
import pyees as pe

a = pe.variable(19, 'dB')
b = pe.variable(11, 'dB')
c = a + b
>> 30 [dB]
```

However, the user can also use logarithmic arithmatic on these when variables has these units. The only downside is that the syntax is not as nice as the linear version.

```
import pyees as pe

a = pe.variable(19, 'dB')
b = pe.variable(11, 'dB')
c = pe.logarithmic.add(a,b)
>> 19.6 [dB]
```

The follwowing methods are available
 - logarithmic.add
 - logarithmic.subtract
 - logarithmic.mean