from scipy.optimize import minimize
import numpy as np
import warnings
from copy import deepcopy
try:
    from variable import variable, scalarVariable, arrayVariable
except ImportError:
    from .variable import variable, scalarVariable, arrayVariable

class _solve:
    
    def __init__(self, func, x, args, bounds, parametric, kwargs):
        self.func = func
        self.x = x
        self.args = args
        self.bounds = bounds
        self.parametric = parametric
        self.kwargs = kwargs   
    
        self.isInputListOrArrayVariable()
        self.checkParametric()
        self.isFunctionProperlyDefined()
        self.testLengthOfEquations()
        self.testOutputOfEquations()
        self.testFunction()
        self.doesUnitsMatch()
        self.findNumberOfVariables()
        self.testNumberOfEquationsAndVariables()
        self.testBounds()
        self.nudgeVariables()
        self.findScales()
        
        if self.solveParametric:
            out = [variable([], self.x[i].unit) for i in range(self.nVariables)]
            for self.currentIndex in range(len(self.parametric[0])):
                for xi in self.x:
                    for xii in xi:
                        xii.dependsOn = {}
                        xii.covariance = {}
                        xii._calculateUncertanty()
                self.solve()
                self.solveUncertanty()
                for i in range(self.nVariables):
                    out[i].append(deepcopy(self.x[i]))
            self.x = out
            self.formatOutput()
        else:
            self.solve()
            self.solveUncertanty()
            self.formatOutput()
            
    def isInputListOrArrayVariable(self):    
        ## determine if the input is a list and if the input is an arrayVariable
        ## this will affect the return type of solve
        if isinstance(self.x, arrayVariable):
            self.isArrayVariable = True
            self.isVariableList = False
        else:
            self.isVariableList = isinstance(self.x,list)
            self.isArrayVariable = False
        if not self.isVariableList: self.x = [self.x]
        
    def checkParametric(self):
        
        if self.parametric is None:
            self.solveParametric = False
        else:
            self.solveParametric = True
            self.currentIndex = 0
    
            if isinstance(self.parametric, arrayVariable):
                self.parametric = [self.parametric]
            else:
                if not isinstance(self.parametric, list):
                    raise ValueError('The parametrics has to be an array variable or a list of array variable')
            
        
        
    
    ## create a wrapper around the function
    def _func(self, *x):
        np.seterr('ignore')
        inputs = []
        inputs += x
        if self.solveParametric:
            inputs += [elem[self.currentIndex] for elem in self.parametric]
        out = self.func(*inputs)
        np.seterr('warn')
        if not isinstance(out[0], list): out = [out]        
        return out

    def isFunctionProperlyDefined(self):
        ## determine if the funciton is properly defined 
        try:
            self._func(*self.x)
        except Exception:
            raise ValueError('The function returned an exception when supplied the initial guesses. Somtehing is wrong.')

    def testLengthOfEquations(self):
        ## test the length of all equations
        for i, equation in enumerate(self._func(*self.x)):
            if len(equation) != 2:
                raise ValueError(f'Equation {i+1} is a list of {len(equation)} elements. This corresponds with an equation with {len(equation)} sides. All equations has to have 2 sides')
        
    def testOutputOfEquations(self):        
        ## test weather both sides of all equations return variables
        for i, equation in enumerate(self._func(*self.x)):
            for j, side in enumerate(equation):
                if not isinstance(side, scalarVariable):
                    side = 'Left' if j == 0 else 'Right'
                    raise ValueError(f'The {side} side of equation {i+1} is not a variable. Both side of each equation has to be a variable')

    ## create another wrapper around the funciton, which will flatten the equations
    def ffunc(self, *x):
        out = []
        for side1, side2 in self._func(*x):
            for elemSide1, elemSide2 in zip(side1, side2):
                out.append([elemSide1, elemSide2])
        return out  

    def testFunction(self):      
        ## test if the right number of variables were supplied for the function
        try:
            self.ffunc(*self.x)
        except TypeError:
            raise ValueError('You supplied the wrong amount of variables for the system of equations')
      
      
    def doesUnitsMatch(self):
        
        ## test the equations
        self.doesUnitsOfEquationsMatch = []
        for o in self.ffunc(*self.x):
            if (len(o) != 2):
                raise ValueError('Each equation has to be a list of 2 elements')
            for bound in o:
                if not isinstance(bound, scalarVariable):
                    raise ValueError('Each side of each equation has to be a variable')        
            ## test if the units match
            if (o[0]._unitObject.unitDictSI != o[1]._unitObject.unitDictSI):
                    raise ValueError('The units of the equations does not match')
            self.doesUnitsOfEquationsMatch.append(o[0].unit == o[1].unit)
        
    def findNumberOfVariables(self):
        
        ## determine the number of variables and if the variables are arrayVariables
        self.nVariables = 0
        for xi in self.x:
            for _ in xi:
                self.nVariables += 1 

        
    def testNumberOfEquationsAndVariables(self):
        ## check the number of equations and variables
        out = self.ffunc(*self.x)
        if (len(out) != self.nVariables):
            raise ValueError(f'You supplied {len(out)} equations but {self.nVariables} variables. The number of equations and the vairables has to match')

    def fbounds(self, *x):
        out = self.bounds(*x)
        if not isinstance(out[0], list): out = [out]
        return out

    def testBounds(self):
        # check the bounds
        if not self.bounds is None:
            if callable(self.bounds):
                ## the bounds are callable
                
                ## test if the right number of variables were supplied for the function
                try:
                    out = self.fbounds(*self.x)
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
                    if not o[1] in self.x:
                        raise ValueError('The middle element in each bound has to be one of the variable.')
                    
                    ## then covert the upper and lower to the unit of the variable
                    o[0].convert(o[1].unit)
                    o[2].convert(o[1].unit)
                    
                        
            else:
                ## the bounds are not callable
                
                if not isinstance(self.bounds[0],list): self.bounds = [self.bounds]
                
                ## create a matrix to hold the bounds
                bbounds = np.zeros([self.nVariables,2])
                
                currentIndex = 0
                for i, bound in enumerate(self.bounds):
                    
                    for elem in bound:
                        ## check that the elements are variables
                        if not isinstance(elem, scalarVariable):
                            raise ValueError('All bounds has to be variables')
                        ## check the units bounds
                        if (elem._unitObject.unitDictSI != self.x[i]._unitObject.unitDictSI):
                            raise ValueError('The units of the bounds does not match with the corresponding variable')
                        
                        ## convert the bounds to the units of the corresponding variable
                        elem.convert(self.x[i].unit)
                    
                    ## collect the bounds in bbounds
                    try:
                        for low, up, _ in zip(bound[0], bound[1], self.x[i], strict = True):
                            bbounds[currentIndex,:] = [low.value, up.value]
                            currentIndex+=1
                    except ValueError:
                        raise ValueError('Each element of the bounds has to have the same length as the corresponding variable')
                    
                ## update the object "bound"
                self.bounds = bbounds


    def nudgeVariables(self):
        ## nudge of the variables if one or more equation has already been solved
        done = False
        while not done:
            out = self.ffunc(*self.x)
            if sum([1 if elem[0] - elem[1] == 0 else 0 for elem in out]) != 0:
                index = np.random.randint(0, self.nVariables)
                self.x[index]._value *= 1 + np.random.uniform(-0.0001, 0.0001)
            else:
                done = True
    
    def findScales(self):
        ## define the scalings of each equation. This is only valid if there are more than 1 equaiton
        self.scales = np.ones(self.nVariables)
        if self.nVariables != 1:
            out = self.ffunc(*self.x)
            for i,o in enumerate(out):
                if not self.doesUnitsOfEquationsMatch[i]:
                    for elem in o:
                        elem.convert(elem._unitObject.unitStrSI)
                self.scales[i] = 1/(o[0].value - o[1].value)**2    
    
    def keepVariablesFeasible(self, _ = None):
        ## method to keep the variables within the bounda
        ## the method will be given a single input during the minimzation. 
        ## Therefore a single input argument is defined which is never used
        bbounds = self.fbounds(*self.x)
        for bound in bbounds:
            for b0, b1, b2 in zip(*bound):
                if not b0 < b1 < b2:
                    b1._value = np.min([np.max([b0.value, b1.value]), b2.value])

    
    ## define the minimization problem
    def minimizationProblem(self, xx):
       
        ## update the values of the variables
        currentIndex = 0
        for xi in self.x:
            for xii in xi:
                xii._value = xx[currentIndex]
                currentIndex += 1
      
        ## evaluate the function
        out = self.ffunc(*self.x)
        
        ## convert all equations to the SI unit system, if each side of the units are not identical
        for b,o in zip(self.doesUnitsOfEquationsMatch, out):
            if not b:
                for elem in o:
                    elem.convert(elem._unitObject.unitStrSI)

        return sum([(e[0].value - e[1].value)**2 * s for e,s in zip(out, self.scales)])

   
    def solve(self):
        ## move the initial guess in to the feasible area if the bounds are callable
        if callable(self.bounds): self.keepVariablesFeasible()
        
        ## create a vector of the initial values
        x0 = np.zeros(self.nVariables)
        currentIndex = 0
        for xi in self.x:
            for xii in xi:
                x0[currentIndex] = xii.value
                currentIndex+=1
        
        
        ## minimize the minimizationproblem
        warnings.filterwarnings('ignore')
        self.out = minimize(
            self.minimizationProblem,
            x0,
            *self.args,
            **self.kwargs,
            bounds = self.bounds if not callable(self.bounds) else None,
            callback=self.keepVariablesFeasible if callable(self.bounds) else None
            )
        warnings.filterwarnings('default')
        if callable(self.bounds): self.keepVariablesFeasible()

    def solveUncertanty(self):
        
        ## loop over all equations and create a list of the residuals
        residuals, J = [], np.zeros([self.nVariables,self.nVariables])
        for i, equation in enumerate(self._func(*self.x)):
            res = equation[0] - equation[1]
            res.convert(res._unitObject.unitStrSI)
            for r in res:
                residuals.append(r)


        ## create the jacobian matrix from the residual
        for i, res in enumerate(residuals):
            ## loop over the variables
            currentIndex = 0
            for xj in self.x:
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
            for xi in self.x:
                for xii in xi:
                    if xii in res.dependsOn:
                        ## i do not know why the gradient has to be scaled by the scale of variable
                        ## before the dependency is added to the variable
                        ## but this works
                        xii._addDependent(res,  Jinv[currentIndex, i] / xii._unitObject._converterToSI.scale)
                    currentIndex +=1

        ## calculate the uncertanty of all variables
        for xi in self.x:
            for xii in xi:
                xii._calculateUncertanty()
        
    def formatOutput(self):
        if self.isArrayVariable or (self.nVariables == 1 and not self.isVariableList): self.x = self.x[0]
        return self.x


def solve(func, x, *args, bounds = None, parametric = None, **kwargs):

    s = _solve(func, x, args, bounds, parametric, kwargs)
    return s.x
    

