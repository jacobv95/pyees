from dataUncert import *

dat1 = readData('example1.xlsx', 'A-B', 'C-D')
q1 = dat1.s1.a + dat1.s1.b
print(q1)

dat2 = readData('example2.xlsx', 'A-B', 'C-D')
q2 = dat2.s1.a + dat2.s1.b
print(q2)
