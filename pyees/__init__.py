# import the necessary modules
from pyees.variable import variable
from pyees.fit import dummy_fit, pol_fit, lin_fit, exp_fit, pow_fit, logistic_fit, logistic_100_fit
from pyees.readData import sheetsFromFile, fileFromSheets
from pyees.prop import prop
from pyees.solve import solve
from pyees.unit import addNewUnit


## TODO undersøg om man kan bruge evalueringer af funktioner til at bestemme uncertanty propagation - selv med produktreglen
## TODO fit - få det til at virke som i bogen
## TODO fit - lav normalized residuals plot
## TODO fit - evaluer, om 68% af datapunkterne ligger 1 standard afvigelse fra fittet

    