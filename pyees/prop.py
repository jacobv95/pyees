import autograd.numpy as np
from autograd.extend import primitive, defvjp
import CoolProp.CoolProp as CP
try:
    from pyees.pyees import variable
except ModuleNotFoundError:
    from pyees import variable


def prop(property, parameterA, valueA, parameterB, valueB, fluid):
    units = {
        'D': 'kg/m3',
        'V': 'kg/m-s',
        'C': 'J/kg-K',
        'H': 'J/kg'
    }
    if property not in units:
        raise ValueError(f'The property {property} is unkown. The known properties are {[key for key in units]}')
    unit = units[property]

    if isinstance(valueA, variable):
        valueA = valueA.value
    if isinstance(valueB, variable):
        valueB = valueB.value

    val = _prop(property, parameterA, valueA, parameterB, valueB, fluid)
    var = variable(val, unit=unit)
    var.unit = unit

    return var


@primitive
def _prop(property, parameterA, valueA, parameterB, valueB, fluid):
    return CP.PropsSI(property, parameterA, valueA, parameterB, valueB, fluid)


def _g_prop_parameterA(ans, property, parameterA, valueA, parameterB, valueB, fluid):
    try:
        diff = CP.PropsSI(f'd({property})/d({parameterA})|{parameterB}', parameterA, valueA, parameterB, valueB, fluid)
        return lambda g: g * diff
    except ValueError:
        return lambda g: g * 0


def _g_prop_parameterB(ans, property, parameterA, valueA, parameterB, valueB, fluid):
    try:
        diff = CP.PropsSI(f'd({property})/d({parameterB})|{parameterA}', parameterA, valueA, parameterB, valueB, fluid)
        return lambda g: g * diff
    except ValueError:
        return lambda g: g * 0


defvjp(_prop, _g_prop_parameterA, _g_prop_parameterB, argnums=[2, 4])


# TODO fix gradient for specific parameters: viscosity
