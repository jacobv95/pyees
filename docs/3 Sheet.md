
# Sheet

Data can be read from an excel-sheet in to a sheet-objects using the sheetsFromFile-method

```
sheet = sheetsFromFile(xlFile: str, dataRange: str | list[str], uncertRange: None, str, list[str] = None, sheets: int, list[int] = None)
```

 - xlFile - path to the excel file to be read
 - dataRange - The coloumns with the data. The start coloumn and the end coloum has to be seperated with a hyphen (-)
 - uncertRange - The coloumns with the data. The start coloumn and the end coloum has to be seperated with a hyphen (-)
 - sheets - the index of the sheets to read from
 - return - a list of sheets or a sheet-object based on the length in the inputs

The excel file has to follow the following structure:
 - The first row is the header
 - The second row is the unit of the data
 - If uncertanty data is given, there has to be an equal number of coloums in the uncertanty range as there is in the data range

The uncertanty can follow one of two structures:
 1. There is one row of uncertanty per row in the range of data.
 2. The data includes covariance data. Therefore there is one matrix of size nxn for each row in the range of data, where n is the number of different measurements.

Two examples are given:

![Example 1](/docs/examples/example1.png)
![Example 2](/docs/examples/example2.png)

```
from pyees import readData

sheet1 = sheetsFromFile('example1.xlsx', dataRange='A-B', uncertRange='C-D')
q1 = sheet1.a + sheet1.b
print(q1)
>> [85.0] +/- [0.6] [m]

dat2 = sheetsFromFile('example2.xlsx', dataRange='A-B', uncertRange='C-D')
q2 = sheet2.a + sheet2.b
print(q2)
>> [85] +/- [2] [m]
```

Notice that the uncertanty of q2 is larger than the uncertanty of q1. This is because example1.xlsx includes information about the covariance of the measurement of a and b.


## printContents

It is possible to print the contents of a sheetobject. 

```
from pyees import sheetsFromFile

sheet = sheetsFromFile('example1.xlsx', dataRange='A-B', uncertRange'C-D')
sheet.printContents()
>> a
>> b
```

