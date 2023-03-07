from pyfluids import Fluid, FluidsList, Input, HumidAir, InputHumidAir
try:
    from variable import variable, scalarVariable, arrayVariable
    from unit import unit
except ImportError:
    from pyees.unit import unit
    from pyees.variable import variable, scalarVariable, arrayVariable
dx = 0.000001

def prop(property, fluid, **kwargs):

    if not fluid in knownFluids:
        raise ValueError(f"The fluid {fluid} is unknown")

    method = knownFluids[fluid][0]
        
    return method(property, kwargs)

def isAllArgumentsUsed(arguments: dict):
    if (len(arguments.keys()) > 0):
        raise ValueError(f'The inputs {list(arguments.keys())} are not appropriate for this fluid')

def findArgument(arguments : dict, parameterName, unitStr, raiseError = True):

    desiredSIBaseUnit = unit(unitStr)._SIBaseUnit
    
    parameter = None
    if  parameterName in arguments.keys():
        parameter = arguments[parameterName]
        del arguments[parameterName]
    else:
        if raiseError:
            raise ValueError(f'Could not find an arugment matching that of a {parameterName}')
    if not parameter is None:
        if parameter._unitObject._SIBaseUnit != desiredSIBaseUnit:
            raise ValueError(f'The input {parameterName} did not have to correct unit of {unitStr}')

    return arguments, parameter

def differentialsBrine(fluid : Fluid, fluidName,  property, C, parameters):

    vars, grads = differentials(fluid, property, parameters)
    
    if (C.uncert == 0):
        return  vars, grads


    y1 = getattr(Fluid(fluidName,C.value*(1 + dx)).with_state(*fluid._inputs), property)
    y2 = getattr(Fluid(fluidName,C.value*(1 - dx)).with_state(*fluid._inputs), property)

    grads.append((y2-y1) / (2*dx * C.value))
    vars.append(C)
    return vars, grads


def differentials(fluid : Fluid, property : str, parameters):
    vars = []
    indexes = []
    for i, param in enumerate(parameters):
        if param.uncert != 0:
            indexes.append(i)
            vars.append(param)
    
    
    inputs = list(fluid._inputs)

    
    grads = []
    for i in indexes:
        
        i0 = inputs[i]
        i1 = Input(i0.coolprop_key, i0.value * (1-dx))
        i2 = Input(i0.coolprop_key, i0.value * (1+dx))
        
        inputs[i] = i1
        fluid.update(*inputs)
        y1 = getattr(fluid, property)
        
        inputs[i] = i2
        fluid.update(*inputs)
        y2 = getattr(fluid, property)
        
        inputs[i] = i0
        fluid.update(*inputs)
        
        grads.append((y2-y1) / (2*dx*i0.value))
    
    return vars, grads
    

def outputFromParameters(scalarMethod, property, params):
    
    
    if (all([type(elem) == scalarVariable for elem in params if not elem is None])):
        ## all inputs are scalars
        out = scalarMethod(property, *params)
        return out
    
    out = []
    ns = []
    paramVecs = [None] * len(params)

    for i, param in enumerate(params):
        if type(param) == arrayVariable:
            paramVecs[i] = param
            ns.append(len(param))
    
    if not (all([ns[0] == n] for n in ns)):
        raise ValueError('All parameters has to have the same length')
    
    
    n = ns[0]
    for i, param in enumerate(params):
        if isinstance(param, scalarVariable):
            paramVecs[i] = variable([param.value] * n, param.unit, [param.uncert] * n)
    
    for i in range(n):
        params = [elem[i] for elem in paramVecs]
        out.append(scalarMethod(property, *params))
    
    return variable([elem.value for elem in out], out[0].unit, [elem.uncert for elem in out])

def propWater(property, arguments):
    
    ## find the appropriate arguments from the list
    arguments, T = findArgument(arguments, 'T', 'K')
    arguments, P = findArgument(arguments, 'P', 'Pa')
    isAllArgumentsUsed(arguments)
    
    ## store the default unit of the arguments
    Tunit = T.unit
    Punit = P.unit
    
    ## convert in to the units of pyfluids
    T.convert('C')
    P.convert('Pa')
    
    out = outputFromParameters(propWaterScalar, property, [T,P])
        
    ## convert the arguments back in to the original units
    T.convert(Tunit)
    P.convert(Punit)
    
    return out

def propWaterScalar(property, T,P):
    ## update the fluid state
    fluid = Fluid(FluidsList.Water)
    fluid = fluid.with_state(Input.temperature(T.value), Input.pressure(P.value))
    
    ## create a variable from the fluid
    var = getattr(fluid, property)
    var = variable(var, propertyUnits[property])
    vars, grads = differentials(fluid, property, [T, P])
    var._addDependents(vars, grads)
    var._calculateUncertanty()
    
    ## return the variable
    return var
   
    
def propMEG(property, arguments):
 
    
    ## find the appropriate arguments from the list
    arguments, T = findArgument(arguments, 'T', 'C')
    arguments, P = findArgument(arguments, 'P', 'Pa')
    arguments, C = findArgument(arguments, 'C', '%')
    isAllArgumentsUsed(arguments)

    ## store the default unit of the arguments
    Tunit = T.unit
    Punit = P.unit
    
    ## convert in to the units of pyfluids
    T.convert('C')
    P.convert('Pa')

    out = outputFromParameters(propMEGScalar, property, [T,P,C])

    ## convert the arguments back in to the original units
    T.convert(Tunit)
    P.convert(Punit)

    return out

def propMEGScalar(property, T,P,C):
    fluid = Fluid(FluidsList.MEG, C.value)    
    
    ## store the default unit of the arguments
    Tunit = T.unit
    Punit = P.unit
    
    ## convert in to the units of pyfluids
    T.convert('C')
    P.convert('Pa')
    
    ## update the fluid state
    fluid = fluid.with_state(Input.temperature(T.value), Input.pressure(P.value))
    
    ## create a variable from the fluid
    var = getattr(fluid, property)
    var = variable(var, propertyUnits[property])
    vars, grads = differentialsBrine(fluid, FluidsList.MEG, property, C, [T, P])
    var._addDependents(vars, grads)
    var._calculateUncertanty()
    
    ## convert the arguments back in to the original units
    T.convert(Tunit)
    P.convert(Punit)
    
    ## return the variable
    return var
    

def propHumidAir(property, arguments):


    arguments, H = findArgument(arguments, 'h', 'm', raiseError=False)
    arguments, P = findArgument(arguments, 'P', 'Pa', raiseError=False)
    arguments, T = findArgument(arguments, 'T', 'C', raiseError=True)
    arguments, Rh = findArgument(arguments, 'Rh', '', raiseError=True)
    isAllArgumentsUsed(arguments)

    if H is None and P is None:
        raise ValueError('You have to specify either an altitude or a pressure')
    if not (H is None) and not (P is None):
        raise ValueError('You cannot specify both an altitude and a pressure')
    
    vars = [T,Rh]
    if not (H is None):
        vars.append(H)
    else:
        vars.append(P)
    
    varUnits = [elem.unit for elem in vars]
    
    desiredUnits = ['C','']
    if not (H is None):
        desiredUnits.append('m')
    else:
        desiredUnits.append('Pa')
        
    for Var, desiredUnit in zip(vars, desiredUnits):
        Var.convert(desiredUnit)
    
    out = outputFromParameters(propHumidAirScalar, property, [T, Rh, H, P])
    
    for Var, varUnit in zip(vars, varUnits):
        Var.convert(varUnit)
    
    return out

def propHumidAirScalar(property, T,Rh,H = None, P = None):
    
    inputs = []
    inputs.append(InputHumidAir.temperature(T.value))
    inputs.append(InputHumidAir.relative_humidity(Rh.value))
    if not (H is None):
        inputs.append(InputHumidAir.altitude(H.value))
    else:
        inputs.append(InputHumidAir.pressure(P.value))
    
    vars = []
    vars.append(T)
    vars.append(Rh)
    if not (H is None):
        vars.append(H)
    else:
        vars.append(P)
        
    fluid = HumidAir()    
    fluid = fluid.with_state(*inputs)
    
    ## create a variable from the fluid
    var = getattr(fluid, property)
    var = variable(var, propertyUnits[property])
    Vars, grads = differentials(fluid, property, vars)
    var._addDependents(Vars, grads)
    var._calculateUncertanty()
    
    return var 

propertyUnits = {
    'density': 'kg/m3',
    'specific_heat': 'J/kg-K',
    'dynamic_viscosity': 'Pa-s'
}

knownProperties = [
    "density",
    "specific_heat",
    "dynamic_viscosity"
]

knownFluids = {
    'water': [propWater],
    'MEG': [propMEG],
    'air': [propHumidAir]
}


if __name__ == '__main__':
    rho = prop('density', 'water', P = variable(1,'bar'), T = variable(30,'C'))
    print(rho)
    