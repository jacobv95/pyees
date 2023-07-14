from scipy.optimize import minimize
import numpy as np
import warnings
try:
    from variable import scalarVariable, arrayVariable
except ImportError:
    from pyees.variable import scalarVariable, arrayVariable



def solve(func, x, *args, bounds = None, **kwargs):

    ## determine if the input is a list and if the input is an arrayVariable
    ## this will affect the return type of solve
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
    
    ## determine if the funciton is properly defined 
    try:
        _func(*x)
    except Exception:
        raise ValueError('The function returned an exception when supplied the initial guesses. Somtehing is wrong.')

    ## create another wrapper around the funciton, which will flatten the equations
    def ffunc(*x):
        out = []
        for equation in _func(*x):
            side1, side2 = equation
            for elemSide1, elemSide2 in zip(side1, side2):
                out.append([elemSide1, elemSide2])
        return out
    
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
        for bound in o:
            if not isinstance(bound, scalarVariable):
                raise ValueError('Each side of each equation has to be a variable')        
        ## test if the units match
        if (o[0]._unitObject.unitDictSI != o[1]._unitObject.unitDictSI):
                raise ValueError('The units of the equations does not match')
        doesUnitsOfEquationsMatch.append(o[0].unit == o[1].unit)
    
    
    ## determine the number of variables and if the variables are arrayVariables
    nVariables = 0
    for xi in x:
        for bound in xi:
            nVariables += 1 


    ## check the number of equations and variables
    if (len(out) != nVariables):
        raise ValueError(f'You supplied {len(out)} equations but {nVariables} variables. The number of equations and the vairables has to match')

    def fbounds(*x):
        out = bounds(*x)
        if not isinstance(out[0], list): out = [out]
        return out

    # check the bounds
    if not bounds is None:
        if callable(bounds):
            ## the bounds are callable
            
            ## test if the right number of variables were supplied for the function
            try:
                out = fbounds(*x)
            except TypeError:
                raise ValueError('The bounds takes the wrong number of variables')
            
            for o in out:
                ## check the length of the bound
                if len(o) != 3:
                    raise ValueError('Each bound has to have 3 elements')
                
                ## check that all elements of the bounds are variables
                for bound in o:
                    if not isinstance(bound, scalarVariable):
                        raise ValueError('Each side of the bounds has to be a variable')         
                
                ## check the units bounds
                if (o[0]._unitObject.unitDictSI != o[1]._unitObject.unitDictSI or o[1]._unitObject.unitDictSI != o[2]._unitObject.unitDictSI):
                    raise ValueError('The units of the bounds does not match')
                
                ## chekc that all elements in the bounds have the same length
                try:
                    for _,_,_ in zip(o[0], o[1], o[2], strict = True): pass
                except ValueError:
                    raise ValueError('Each element of the bounds has to have the same length')
                
                ## check that the middle element is a part of the variables
                if not o[1] in x:
                    raise ValueError('The middle element in each bound has to be one of the variable.')
                
                ## then covert the upper and lower to the unit of the variable
                o[0].convert(o[1].unit)
                o[2].convert(o[1].unit)
                
                     
        else:
            ## the bounds are not callable
            
            if not isinstance(bounds[0],list): bounds = [bounds]
            
            ## create a matrix to hold the bounds
            bbounds = np.zeros([nVariables,2])
            
            currentIndex = 0
            for i, bound in enumerate(bounds):
                
                for elem in bound:
                    ## check that the elements are variables
                    if not isinstance(elem, scalarVariable):
                        raise ValueError('All bounds has to be variables')
                     ## check the units bounds
                    if (elem._unitObject.unitDictSI != x[i]._unitObject.unitDictSI):
                        raise ValueError('The units of the bounds does not match with the corresponding variable')
                    
                    ## convert the bounds to the units of the corresponding variable
                    elem.convert(x[i].unit)
                
                ## collect the bounds in bbounds
                try:
                    for low, up, _ in zip(bound[0], bound[1], x[i], strict = True):
                        bbounds[currentIndex,:] = [low.value, up.value]
                        currentIndex+=1
                except ValueError:
                    raise ValueError('Each element of the bounds has to have the same length as the corresponding variable')
                
            ## update the object "bound"
            bounds = bbounds

    ## define the scalings of each equation. This is only valid if there are more than 1 equaiton
    scales = np.ones(nVariables)
    if nVariables != 1:
        out = ffunc(*x)
        currentIndex = 0
        for i,o in enumerate(out):
            if not doesUnitsOfEquationsMatch[i]:
                for elem in o:
                    elem.convert(elem._unitObject.unitStrSI) 
            scales[i] = 1/(o[0].value - o[1].value)**2
    
    
    ## method to keep the variables within the bounda
    ## the method will be given a single input during the minimzation. 
    ## Therefore a single input argument is defined which is never used
    def keepVariablesFeasible(_ = None):
        bbounds = fbounds(*x)
        for bound in bbounds:
            for b0, b1, b2 in zip(*bound):
                if not b0 < b1 < b2:
                    b1._value = np.min([np.max([b0.value, b1.value]), b2.value])

    
    ## define the minimization problem
    def minimizationProblem(xx):
       
        ## update the values of the variables
        currentIndex = 0
        for xi in x:
            for xii in xi:
                xii._value = xx[currentIndex]
                currentIndex += 1
      
        ## evaluate the function
        out = ffunc(*x)
        
        ## convert all equations to the SI unit system, if each side of the units are not identical
        for b,o in zip(doesUnitsOfEquationsMatch, out):
            if not b:
                for elem in o:
                    elem.convert(elem._unitObject.unitStrSI)

        return sum([(e[0].value - e[1].value)**2 * s for e,s in zip(out, scales)])


    ## move the initial guess in to the feasible area if the bounds are callable
    if callable(bounds): keepVariablesFeasible()
    
    
    ## create a vector of the initial values
    x0 = np.zeros(nVariables)
    currentIndex = 0
    for xi in x:
        for xii in xi:
            x0[currentIndex] = xii.value
            currentIndex+=1
    
    
    ## minimize the minimizationproblem
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
        res = equation[0] - equation[1]
        res.convert(res._unitObject.unitStrSI)
        for bound in res:
            residuals.append(bound)


    ## create the jacobian matrix from the residual
    for i, res in enumerate(residuals):
        ## loop over the variables
        currentIndex = 0
        for xj in x:
            for xjj in xj:
                if xjj in res.dependsOn:               
                    ## add the gradient d(residual)/d(xj) to the jacobian matrix
                    J[i, currentIndex] += res.dependsOn[xjj][1]
                currentIndex += 1
           
    # inverse the jacobian
    Jinv = np.linalg.inv(J)

    ## add the dependency from the residual to the variables
    for i, res in enumerate(residuals):
        currentIndex = 0
        for xi in x:
            for xii in xi:
                if xii in res.dependsOn:
                    xii._addDependent(res, Jinv[currentIndex, i])
                currentIndex +=1
    
    ## calculate the uncertanty of all variables
    for xi in x:
        for xii in xi:
            xii._calculateUncertanty()
    
    if isArrayVariable or (nVariables == 1 and not isVariableList): x = x[0]
    return x


