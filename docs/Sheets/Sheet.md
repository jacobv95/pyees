
# Sheets

A sheet is a collection of variables. This allows the user to iterate over variables easily

```
import pyees as pe
sheet = pe.sheet()

sheet.a = pe.variable(10, 'm')
sheet.b = pe.variable(20, 'L')

for elem in sheet:
    print(elem)

>> 10 [m]
>> 20 [L]
```


## Sheet From File

A sheet can be created from an excel file using the sheetsFromFile-method

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
import pyees as pe

sheet1 = pe.sheetsFromFile('example1.xlsx', dataRange='A-B', uncertRange='C-D')
q1 = sheet1.a + sheet1.b
print(q1)
>> [85.0] +/- [0.6] [m]

dat2 = pe.sheetsFromFile('example2.xlsx', dataRange='A-B', uncertRange='C-D')
q2 = sheet2.a + sheet2.b
print(q2)
>> [85] +/- [2] [m]
```

Notice that the uncertanty of q2 is larger than the uncertanty of q1. This is because example1.xlsx includes information about the covariance of the measurement of a and b.



## File From Sheet

An excel-file can be created from a sheet using the method "fileFromSheets"

fileFromSheets(sheets: sheet | list[sheet], fileName: str, sheetNames: str | list[str] = None, showUncert: bool = True)

A new excel-file is created with one worksheet for each sheet in the input "sheet". Each worksheet will have one coloumn for each variable within the corresponding sheet-object. The first and second row of the worksheet is the name and the unit. The remaining rows of the worksheet is the value and the uncertanty of the variable presented as "A +/- u_a"

## List-like methods

The sheet object supports the following methods
```
import pyees as pe

sheet1 = pe.sheetsFromFile('example1.xlsx', dataRange='A-B', uncertRange'C-D')
sheet2 = pe.sheetsFromFile('example1.xlsx', dataRange='A-B', uncertRange'C-D')

len(sheet1)
>> 1

sheet1.append(sheet2)
print(sheet1.A)
>> [33, 33] [m] +/- [1, 1]
print(sheet1.B)
>> [52, 52] [m] +/- [2, 2]

sheet1.pop(0)
print(sheet1.A)
>> [33] [m] +/- [1]
print(sheet1.B)
>> [52] [m] +/- [2]
```


## printContents

It is possible to print the contents of a sheetobject. This will print the names of the variables within the sheet.

```
import pyees as pe

sheet = pe.sheetsFromFile('example1.xlsx', dataRange='A-B', uncertRange'C-D')
sheet.printContents()
>> a
>> b
```



