from pyfluids import Fluid, FluidsList, Input, HumidAir, InputHumidAir, Mixture
try:
    from variable import variable, arrayVariable
except ImportError:
    from pyees.variable import variable, arrayVariable
dx = 0.000001


def differentials(fluid : Fluid, property : str, concentration, parameters):
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
        
        try:
            i1 = Input(i0.coolprop_key, i0.value * (1-dx))
            i2 = Input(i0.coolprop_key, i0.value)
            inputs[i] = i1
            fluid.update(*inputs)
            y1 = getattr(fluid, property)
            
            inputs[i] = i2
            fluid.update(*inputs)
            y2 = getattr(fluid, property)
        
            inputs[i] = i0
            fluid.update(*inputs)
        except ValueError:
            i1 = Input(i0.coolprop_key, i0.value)
            i2 = Input(i0.coolprop_key, i0.value * (1 + dx))
            inputs[i] = i1
            fluid.update(*inputs)
            y1 = getattr(fluid, property)
            
            inputs[i] = i2
            fluid.update(*inputs)
            y2 = getattr(fluid, property)
        
            inputs[i] = i0
            fluid.update(*inputs)
        
        grads.append((y2-y1) / (dx*i0.value))
    
    if not concentration is None:
    
        if not hasattr(fluid, 'name'):
            
            for i in range(len(concentration)):
                vars.append(concentration[i])
                try:
                    frac1 = [elem for elem in fluid.fractions]
                    frac2 = [elem for elem in fluid.fractions]
                    
                    frac1[i] *= (1+dx)
                    # frac2[i] *= (1-dx)
                    
                    frac1 = [elem / sum(frac1) * 100 for elem in frac1]
                    frac2 = [elem / sum(frac2) * 100 for elem in frac2]
                    
                    f1 = Mixture(fluid.fluids, frac1)
                    f2 = Mixture(fluid.fluids, frac2)
                except ValueError:
                    
                    frac1 = [elem for elem in fluid.fractions]
                    frac2 = [elem for elem in fluid.fractions]
                    
                    # frac1[i] *= (1+dx)
                    frac2[i] *= (1-dx)
                    
                    frac1 = [elem / sum(frac1) * 100 for elem in frac1]
                    frac2 = [elem / sum(frac2) * 100 for elem in frac2]
                    
                    f1 = Mixture(fluid.fluids, frac1)
                    f2 = Mixture(fluid.fluids, frac2)
                
                f1.update(*inputs)
                f2.update(*inputs)
                
                y1 = getattr(f1, property)
                y2 = getattr(f2, property)
                
                grads.append((y2-y1) / (frac1[i] - frac2[i]))            
        else:
            vars.append(concentration)
        
            try:
                f1 = Fluid(fluid.name, fluid.fraction * (1 + dx))
                f2 = Fluid(fluid.name, fluid.fraction)
            except ValueError:
                f1 = Fluid(fluid.name, fluid.fraction)
                f2 = Fluid(fluid.name, fluid.fraction * (1 - dx))
                
            f1.update(*inputs)
            f2.update(*inputs)
            
            y1 = getattr(f1, property)
            y2 = getattr(f2, property)
            grads.append((y2-y1) / (dx*concentration.value))
            
    
    return vars, grads

    

    
fluidTranslator = {
    'water': 'Water',
    'WATER': 'Water',
    'air': 'Air',
    'AIR': 'Air',
    'meg': 'MEG'
}

propertyTranslator = {
    'T': 'temperature',
    't' : 'temperature',
    'P': 'pressure',
    'p': 'pressure',
    'Rh': 'relative_humidity',
    'rh': 'relative_humidity'
}

propertyUnits = {
    'altitude':             'm',
    'compressibility':      '1',
    'conductivity':         'W/m-K',
    'critical_pressure':    'Pa',
    'critical_temperature': 'C',
    'density':              'kg/m3',
    'dynamic_viscosity':    'Pa-s',
    'enthalpy':             'J/kg',
    'entropy':              'J/kg-K',
    'freezing_temperature': 'C',
    'internal_energy':      'J/kg',
    'kinematic_viscosity':  'm2/s',
    'max_pressure':         'Pa',
    'max_temperature':      'C',
    'min_pressure':         'Pa',
    'min_temperature':      'C',
    'molar_mass':           'kg/mol',
    'prandtl':              '1',
    'pressure':             'Pa',
    'quality':              '%',
    'sound_speed':          'm/s',
    'specific_heat':        'J/kg-K',
    'specific_volume':      'm3/kg',
    'surface_tension':      'N/m',
    'temperature':          'C',
    'triple_pressure':      'Pa',
    'triple_temperature':   'C',
    'dew_temperature':      'C',
    'humidity':             '1',
    'partial_pressure':     'Pa',
    'relative_humidity':    '%',
    'wet_bulb_temperature': 'C'
    ''
}

def prop(property, fluid, C = None, **state):

    params = list(state.values())
    
    ## determine which of the parameters that are arrayVariables
    isArrayVariable = [hasattr(param, '__len__') for param in params]    
    
    indexesOfArrayVariables = [i for i, elem in enumerate(isArrayVariable) if elem == True]
    if sum(isArrayVariable) > 0:
        ## make sure, that all arrayVariables have the same length
        n = len(params[indexesOfArrayVariables[0]])
        for i in indexesOfArrayVariables:
            if len(params[i]) != n:
                raise ValueError('All parameters has to have the same length')
    
        concentrations = [None for i in range(n)]
        if not C is None:
            if isinstance(C, list):
                if hasattr(C[0], '__len__'):
                    n = len(C[0])
                    for elem in C:
                        if len(elem) != n:
                            raise ValueError('All concentrations has to have the same length')
                    concentrations = []
                    for i in range(n):
                        con = []
                        for j in range(len(C)):
                            con.append(C[j][i])
                        concentrations.append(con)
                else:
                    concentrations = [C] * n
            else:
                if hasattr(C, '__len__'):
                    if len(C) != n:
                        raise ValueError('All parameters has to have the same length')
                    concentrations = C
                else:
                    concentrations = [C for i in range(n)]
    else:
        if isinstance(C, list):
            if hasattr(C[0], '__len__'):
                n = len(C[0])
                for elem in C:
                    if len(elem) != n:
                        raise ValueError('All concentrations has to have the same length')
                concentrations = []
                for i in range(n):
                    con = []
                    for j in range(len(C)):
                        con.append(C[j][i])
                    concentrations.append(con)
            else:
                n = 1
                concentrations = [C]
                    
        else:
            if hasattr(C, '__len__'):
                n = len(C)
                concentrations = C
            else:
                return _propScalar(property, fluid, C, **state)
    
    ## create a list of scalar variables. This list will have the same length as the arrayVariables supplied
    listOfScalarVariables = []
    
    ## loop over the length of the arrayVariables
    for i in range(n):
        
        ## create a list of scalarparameter
        scalarParams = {}
        
        ## loop over the parameters
        for ii, param in enumerate(params):
            
            ## if the current parameter is an arrayVariable, then append the i'th scalarVaraible of the arrayVariable to the list of the scalar parameters
            if ii in indexesOfArrayVariables:
                # scalarParams.append(param[i])
                scalarParams[list(state.keys())[ii]] = param[i]
            
            ## if the current parameter is a scalarvariable, then simply append the parameter to the scalarparameters
            else:
                scalarParams[list(state.keys())[ii]] = param
                
        ## append the output of the scalarmethods to the list of scalar variables
        listOfScalarVariables.append(_propScalar(property, fluid, concentrations[i], **scalarParams))
        
        
    if n == 1: return listOfScalarVariables[0]
    ## return an arrayVariable, which is created from a list of scalarvariables
    return arrayVariable(scalarVariables=listOfScalarVariables)
    
def _propScalar(property, fluid, concentration = None, **state):
    
    
    listOfNonInputs = []
    if not concentration is None:
        if isinstance(concentration, list):
            originalConcentrationUnit = []
            for elem in concentration:
                originalConcentrationUnit.append(elem.unit)
                elem.convert('%')
            listOfNonInputs = concentration
        else:
            originalConcentrationUnit = concentration.unit
            concentration.convert('%')
            listOfNonInputs = [concentration.value]
    
    originalUnits = []
    listOfInputs = []
    if isinstance(fluid, list):
        input = Input
    else:
        input = InputHumidAir if fluid.lower() == 'air' else Input
    for key, value in state.items():
        
        originalUnits.append(value.unit)

        if key in propertyTranslator:
            key = propertyTranslator[key]
        value.convert(propertyUnits[key])        

        method = getattr(input, key)
            
        listOfInputs.append(method(value.value))
    
    
    ## update the fluid state
    if isinstance(fluid, list):
        listOfFluids = []
        for i in range(len(fluid)):
            try:
                elem = getattr(FluidsList, fluid[i])
            except AttributeError:
                elem = getattr(FluidsList, fluidTranslator[fluid[i]])
            listOfFluids.append(elem)
        fluid = Mixture(listOfFluids, [elem.value for elem in concentration])
    else:
        if fluid.lower() == 'air':
            fluid = HumidAir(*listOfNonInputs)
        else:
            try:
                fluid = Fluid(getattr(FluidsList, fluid), *listOfNonInputs) 
            except AttributeError:
                fluid = Fluid(getattr(FluidsList, fluidTranslator[fluid]), *listOfNonInputs) 
            
    fluid = fluid.with_state(*listOfInputs)
    
    

    vars = [item for item in state.values()]

    ## create a variable from the fluid
    var = getattr(fluid, property)
    var = variable(var, propertyUnits[property])
    vars, grads = differentials(fluid, property, concentration, vars)
    for v, g in zip(vars, grads):
        var._addDependent(v,g)
    var._calculateUncertanty()
    
    for i, (key, value) in enumerate(state.items()): value.convert(originalUnits[i])
    if not concentration is None:
        if isinstance(concentration, list):
            for i, elem in enumerate(concentration):
                elem.convert(originalConcentrationUnit[i])
        else:
            concentration.convert(originalConcentrationUnit)
    
    return var


if __name__ == "__main__":
    
    # humidity = prop('humidity', 'air', rh = variable(60, '%'), p = variable( 101325, 'Pa'), T = variable(20, 'C'))
    # print(humidity)
    humid_air = HumidAir().with_state(
        InputHumidAir.pressure(1e5),
        InputHumidAir.temperature(35),
        InputHumidAir.humidity(0.008890559976462207),
    )   
    print(humid_air.specific_heat)