from scipy.optimize import minimize
import numpy as np
import warnings
try:
    from variable import scalarVariable, arrayVariable
except ImportError:
    from pyees.variable import scalarVariable, arrayVariable


def solve(func, x, *args, bounds = None, **kwargs):
    
    
    variablesAreArrays = False
    if isinstance(x, list):
        variablesAreArrays =  isinstance(x[0], arrayVariable)
    else:
        variablesAreArrays =  isinstance(x, arrayVariable)

    if not variablesAreArrays:
        return _solve(func, x, *args, bounds = bounds, **kwargs)

    ## TODO solve vector problems
    raise NotImplementedError()

def _solve(func, x, *args, bounds, **kwargs):
    ## find the number of variables
    isVariableList = isinstance(x,list)
    if not isVariableList: x = [x]
    n = len(x)
    
    ## create a wrapper around the function
    def ffunc(*x):
        np.seterr('ignore')
        out = func(*x)
        np.seterr('warn')
        if not isinstance(out[0], list): out = [out]
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
        for elem in o:
            if not isinstance(elem, scalarVariable):
                raise ValueError('Each side of each equation has to be a variable')        
        ## test if the units match
        if (o[0]._unitObject._SIBaseUnit != o[1]._unitObject._SIBaseUnit):
                raise ValueError('The units of the equations does not match')
            

        doesUnitsOfEquationsMatch.append(o[0].unit == o[1].unit)
    
    ## check the number of equations and variables
    if (len(out) != len(x)):
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
    scales = np.ones(n)
    if n != 1:
        for i,o in enumerate(out):
            for elem in o:
                elem.convert(elem._unitObject._SIBaseUnit) 
            scales[i] = 1/(o[0].value - o[1].value)**2
    
    def keepVariablesFeasible(_ = None):
        if bounds is None: return
        bbounds = fbounds(*x)
        for i, bound in enumerate(bbounds):
            if not bound[0] < bound[1] < bound[2]:
                if not doesUnitsOfBoundsMatch[i]:
                    for elem in bound:
                        elem.convert(elem._unitObject._SIBaseUnit)
                var = np.min([np.max(bound[0], bound[1]), bound[2]])
                x[boundIndexes[i]]._value = var.value
                    
    
    ## define the minimization problem
    def minimizationProblem(xx):
       
        ## update the values of the variables
        for xi, xxi in zip(x,xx):
            xi._value = xxi
      
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
    warnings.filterwarnings('ignore')
    out = minimize(
        minimizationProblem,
        [elem.value for elem in x],
        *args,
        **kwargs,
        bounds = bounds if not callable(bounds) else None,
        callback=keepVariablesFeasible if callable(bounds) else None
        )
    warnings.filterwarnings('default')
    if callable(bounds): keepVariablesFeasible()

    ## loop over all equations and create a list of the residuals and create the jacobian matrix
    residuals, J = [], np.zeros([n,n])
    for i, equation in enumerate(ffunc(*x)):

        ## add the residual
        res = equation[0] - equation[1]
        res.convert(res._unitObject._SIBaseUnit)
        residuals.append(res)
        
        ## loop over the variables
        for j, xj in enumerate(x):
            ## add the gradient d(residual)/d(xj) to the jacobian matrix
            if xj in res.dependsOn:
                for dependency in res.dependsOn[xj].values():
                    J[i,j] += dependency[2]



    # inverse the jacobian
    Jinv = np.linalg.inv(J)
    
    ## add the residuals and a row of the inverse jacobian to each variable and calculate the uncertanty
    for i, xi in enumerate(x):
        xi._addDependents(residuals, Jinv[i,:])
        xi._calculateUncertanty()
    if (n == 1 and not isVariableList): x = x[0]
    return x



if __name__=="__main__":
    from variable import variable
    a = variable([23.7, 12.3], '', [0.1, 0.05])
    b = variable([943, 793], '', [12.5, 9.4])
        
    def func(x):
        return [a * x, b]
    
    x = solve(func, variable([1,1],''), tol = 1e-6)

    correct = b / a
    print(x, correct)
