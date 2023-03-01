
# Importing data

Data can be imported using the function "readData"

```
dat = readData(xlFile: str, dataRange: str, uncertRange=None: str)
```

 - xlFile - path to the excel file to be read
 - dataRange - The coloumns with the data. The start coloumn and the end coloum has to be seperated with a hyphen (-)
 - uncertRange - The coloumns with the data. The start coloumn and the end coloum has to be seperated with a hyphen (-)

The excel file has to follow the following structure:
 - The first row is the header
 - The second row is the unit of the data
 - If uncertanty data is given, there has to be an equal number of coloums in the uncertanty range as there is in the data range

The uncertanty can follow one of two structures:
 1. There is one row of uncertanty per row in the range of data.
 2. There is one matrix of size nxn for each row in the range of data, where n is the number of different measurements

Structure 1 is used if only the uncertanty of the measurement is known. Structure 2 is unsed if covariance between the measurements are known as well.

Two examples are given:

![Example 1](/docs/examples/example1.png)
![Example 2](/docs/examples/example2.png)

```
from dataUncert import *

dat1 = readData('example1.xlsx', 'A-B', 'C-D')
q1 = dat1.s1.a + dat1.s1.b
print(q1)
>> [85.0] +/- [0.6] [m]

dat2 = readData('example2.xlsx', 'A-B', 'C-D')
q2 = dat2.s1.a + dat2.s1.b
print(q2)
>> [85] +/- [2] [m]
```

Notice that the uncertanty of q2 is larger than the uncertanty of q1. This is because example1.xlsx includes information about the covariance of the measurement of a and b.

Notice that both dat1 and dat2 has an object called s1. That is because the data for a and b are both located on sheet 1 of the .xlsx file.



## printContents

It is possible to print the contents of a data-object. 

```
from dataUncert import *

dat1 = readData('example1.xlsx', 'A-B', 'C-D')
dat1.printContents()
>> .s1.a
>> .s1.b
```

