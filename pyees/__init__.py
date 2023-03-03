# import the necessary modules
try:
    from .variable import variable
    from .fit import dummy_fit, pol_fit, lin_fit, exp_fit, pow_fit, logistic_fit, logistic_100_fit
    from .readData import readData
    from .prop import prop
    from .solve import solve
    from .unitSystem import getDicts
except ImportError:
    from variable import variable
    from fit import dummy_fit, pol_fit, lin_fit, exp_fit, pow_fit, logistic_fit, logistic_100_fit
    from readData import readData
    from prop import prop
    from solve import solve
    from unitSystem import getDicts


knownUnits, knownCharacters, knownUnitsDict, knownPrefixes, baseUnit = getDicts()
    