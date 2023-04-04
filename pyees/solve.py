from scipy.optimize import minimize
import numpy as np
import warnings
try:
    from variable import scalarVariable, arrayVariable
except ImportError:
    from pyees.variable import scalarVariable, arrayVariable



def solve(func, x, *args, bounds = None, **kwargs):

    
    ## find the number of variables
    if isinstance(x, arrayVariable):
        isArrayVariable = True
        isVariableList = False
    else:
        isVariableList = isinstance(x,list)
        isArrayVariable = False
    if not isVariableList: x = [x]
    
    ## create a wrapper around the function
    def _func(*x):
        np.seterr('ignore')
        out = func(*x)
        np.seterr('warn')
        if not isinstance(out[0], list): out = [out]        
        return out
    
    ## determine if the equaions are arrayVariables
    try:
        A = _func(*x)
    except Exception:
        raise ValueError('The function returned an exception when supplied the initial guesses. Somtehing is wrong.')
    isArrayEquations = []
    for elem in A:
        isArrayEquations.append(isinstance(elem[0], arrayVariable))
    
    
    ## create another wrapper around the funciton, which will flatten the equations
    def ffunc(*x):
        out = _func(*x)
        oout = []
        for elem, isArrayEquation in zip(out, isArrayEquations):
            if not isArrayEquation:
                oout.append(elem)
            else:
                side1, side2 = elem
                for sside1, sside2 in zip(side1, side2):
                    oout.append([sside1, sside2])
        return oout
    
    ## test if the right number of variables were supplied for the function
    try:
        out = ffunc(*x)
    except TypeError:
        raise ValueError('You supplied the wrong amount of variables for the system of equations')
    
    ## test the equations
    doesUnitsOfEquationsMatch = []
    for o in out:
        if (len(o) != 2):
            raise ValueError('Each equation has to be a list of 2 elements')
        for elem in o:
            if not isinstance(elem, scalarVariable):
                raise ValueError('Each side of each equation has to be a variable')        
        ## test if the units match
        if (o[0]._unitObject._SIBaseUnit != o[1]._unitObject._SIBaseUnit):
                raise ValueError('The units of the equations does not match')
        doesUnitsOfEquationsMatch.append(o[0].unit == o[1].unit)
    
    
    ## determine the number of variables and if the variables are arrayVariables
    isArrayVariables = []
    nVariables = 0
    for elem in x:
        if isinstance(elem, arrayVariable):
            nVariables += len(elem)
            isArrayVariables.append(True)
        else:
            nVariables += 1
            isArrayVariables.append(False)


    ## check the number of equations and variables
    if (len(out) != nVariables):
        raise ValueError(f'You supplied {len(out)} equations but {len(x)} variables. The number of equations and the vairables has to match')

    def fbounds(*x):
        out = bounds(*x)
        if not isinstance(out[0], list): out = [out]
        return out

    # check the bounds
    doesUnitsOfBoundsMatch = []
    boundIndexes = []
    if not bounds is None:
        if callable(bounds):
            ## test if the right number of variables were supplied for the function
            try:
                out = fbounds(*x)
            except TypeError:
                raise ValueError('The bounds takes the wrong number of variables')
            
            for o in out:
                if len(o) != 3:
                    raise ValueError('Each bound has to have 3 elements')
                for elem in o:
                    if not isinstance(elem, scalarVariable):
                        raise ValueError('Each side of the bounds has to be a variable')         
                ## check the bounds
                if (o[0]._unitObject._SIBaseUnit != o[1]._unitObject._SIBaseUnit or o[1]._unitObject._SIBaseUnit != o[2]._unitObject._SIBaseUnit):
                    raise ValueError('The units of the bounds does not match')
                
                lower = o[0]
                var = o[1]
                upper = o[2]
                if isinstance(var, arrayVariable):
                    try:
                        if not (len(lower) == len(var) == len(upper)):
                            raise ValueError('Each element of the bounds has to have the same length')
                    except TypeError:
                        raise ValueError('Each element of the bounds has to have the same length')
                doesUnitsOfBoundsMatch.append([o[0].unit == o[1].unit, o[1].unit == o[2].unit])
                
                if not o[1] in x:
                    raise ValueError('The middle element in each bound has to be one of the variable.')
                
                boundIndexes.append(x.index(o[1]))            
        else:
            if not isinstance(bounds[0],list): bounds = [bounds]
            bbounds = []
            for i, elem in enumerate(bounds):
                for e in elem:
                    if not isinstance(e, scalarVariable):
                        raise ValueError('All bounds has to be variables')
                    e.convert(x[i].unit)
                bbounds.append([e.value for e in elem])
            bounds = bbounds

    ## define the scalings of each equation. This is only valid if there are more than 1 equaiton
    scales = np.ones(nVariables)
    if nVariables != 1:
        out = ffunc(*x)
        currentIndex = 0
        for i,o in enumerate(out):
            for elem in o:
                elem.convert(elem._unitObject._SIBaseUnit) 
            scales[i] = 1/(o[0].value - o[1].value)**2
    
    def keepVariablesFeasible(_ = None):
        bbounds = fbounds(*x)
        for i, bound in enumerate(bbounds):
            bound0, bound1, bound2 = bound
            if isinstance(bound1, arrayVariable):
                vals = []
                for b0, b1, b2 in zip(bound0, bound1, bound2):
                    if not b0 < b1 < b2:
                        if not doesUnitsOfBoundsMatch[i]:
                            b0.convert(b0._unitObject._SIBaseUnit)
                            b1.convert(b1._unitObject._SIBaseUnit)
                            b2.convert(b2._unitObject._SIBaseUnit)
                        var = np.min([np.max(b0, b1), b2])
                        vals.append(var.value)
                    else:
                        vals.append(b1.value)
                for xi, val in zip(x[boundIndexes[i]], vals):
                    xi._value = val
            else:
                if not bound0 < bound1 < bound2:
                    if not doesUnitsOfBoundsMatch[i]:
                        bound0.convert(bound0._unitObject._SIBaseUnit)
                        bound1.convert(bound0._unitObject._SIBaseUnit)
                        bound2.convert(bound0._unitObject._SIBaseUnit)
                    var = np.min([np.max(bound0, bound1), bound2])
                    x[boundIndexes[i]]._value = var.value
                    
    
    ## define the minimization problem
    def minimizationProblem(xx):
       
        ## update the values of the variables
        currentIndex = 0
        for i, xi in enumerate(x):
            if isArrayVariables[i]:
                for ii in range(len(xi)):
                    xi[ii]._value = xx[currentIndex + ii]
                currentIndex += len(xi)
            else:
                xi._value = xx[currentIndex]
                currentIndex += 1
      
        ## evaluate the function
        out = ffunc(*x)
        
        ## convert all equations to the SI unit system, if each side of the units are not identical
        for b,o in zip(doesUnitsOfEquationsMatch, out):
            if not b:
                for elem in o:
                    elem.convert(elem._unitObject._SIBaseUnit)

        return sum([(e[0].value - e[1].value)**2 * s for e,s in zip(out, scales)])

    ## run the minimization
    if callable(bounds): keepVariablesFeasible()
    
    x0 = []
    for i, elem in enumerate(x):
        if isArrayVariables[i]:
            for eelem in elem:
                x0.append(eelem.value)
        else:
            x0.append(elem.value)
    
    warnings.filterwarnings('ignore')
    out = minimize(
        minimizationProblem,
        x0,
        *args,
        **kwargs,
        bounds = bounds if not callable(bounds) else None,
        callback=keepVariablesFeasible if callable(bounds) else None
        )
    warnings.filterwarnings('default')
    if callable(bounds): keepVariablesFeasible()

    ## loop over all equations and create a list of the residuals
    residuals, J = [], np.zeros([nVariables,nVariables])
    for i, equation in enumerate(_func(*x)):

        ## add the residual
        res = equation[0] - equation[1]
        res.convert(res._unitObject._SIBaseUnit)

        if isArrayEquations[i]:
            for elem in res:
                residuals.append(elem)
        else:
            residuals.append(res)

    ## create the jacobian matrix from the residual
    for i, res in enumerate(residuals):
        ## loop over the variables
        currentIndex = 0
        for j, xj in enumerate(x):
            if isArrayVariables[j]:
                n = len(xj)
                for jj, xjj in enumerate(xj):
                    if xjj in res.dependsOn:               
                        ## add the gradient d(residual)/d(xj) to the jacobian matrix
                        J[i, currentIndex + jj : currentIndex + jj + 1] += res.dependsOn[xjj][2]
            else:
                n = 1
                if xj in res.dependsOn:
                        ## add the gradient d(residual)/d(xj) to the jacobian matrix
                        J[i, currentIndex : currentIndex + 1] += res.dependsOn[xj][2]
            currentIndex += n
           
    # inverse the jacobian
    Jinv = np.linalg.inv(J)

    ## add the residuals and a row of the inverse jacobian to each variable and calculate the uncertanty
    for i, xi in enumerate(x):
        currentIndex = i
        if isArrayVariables[i]:
            for xii in xi:
                for j, rj in enumerate(residuals):
                    xii._addDependent(rj, Jinv[currentIndex,j])
                currentIndex+=1
        else:    
            for j, rj in enumerate(residuals):
                xi._addDependent(rj, Jinv[currentIndex,j])
            currentIndex += 1
        xi._calculateUncertanty()
        
    
    if isArrayVariable or (nVariables == 1 and not isVariableList): x = x[0]
    return x
