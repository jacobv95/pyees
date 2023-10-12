# pyees
EES is a nice program with a lot of uses. However, the syntax is poor and the it can be quite combersome to import and export data and figures. Therefore pyees was created.


# How to install
Just run ```pip install pyees```

## Scope of this package
The scope of this package is to
 - Easily calculate uncertanty propagation
 - Read data with uncertanty from an .xls or .xlsx file
 - Print measurements with the correct number of significant digits based on the uncertanty
 - Plot data with errorbars
 - Perform regression where the regression constants are affected by the uncertanty of the data
 - solve n equations with n variables
 - look up material properties

## Documentation
The documentation is split to parts:
 - [Variables](/docs/Variables/Variables.md)
   - [Units](/docs/Units/Units.md)
     - [Temperature](/docs/Units/Temperature.md)
     - [Add new unit](/docs/Units/Add%20new%20units.md)
 - [Sheets](/docs/Sheets/Sheet.md)
 - [Fitting](/docs/Fitting/Fitting.md)
  - [Create new fit class](/docs/Fitting/create%20New%20Fit%20Class.md)
 - [Prop](/docs/Prop/Prop.md)
 - [Solving](/docs/Solving/Solving.md)
