import unittest
import numpy as np

from solve import solve
from variable import variable
from prop import prop
    
tol = 1e-5
solveTol = 1e-12

class test(unittest.TestCase):
    
    def assertRelativeDifference(self, a, b, r):
        assert abs(a-b) <= abs(b * r), f"The value {a} and {b} has a greater relative difference than {r}. The difference was {abs(a-b)} and was allowed to be {b*r}"
      
    def testSolveOneLinearEquation(self):
        a = variable(23.7, '', 0.1)
        b = variable(943, '', 12.5)
        def func(x):
            return [a * x, b]
        
        x = solve(func, variable(1,''), tol = solveTol)
        correct = b / a

        self.assertRelativeDifference(x.value, correct.value, tol)
        self.assertRelativeDifference(x.uncert, correct.uncert, tol)
    
    def testSolveOneNonlinearEquation(self):
        a = variable(23.7, '', 0.1)
        b = variable(943, '', 12.5)
        
        def func(x):
            return [a * x**2, b]
        
        x = solve(func, variable(1,''), tol = solveTol)
        correct = (b / a)**(1/2)
        
        self.assertRelativeDifference(x.value, correct.value, tol)
        self.assertRelativeDifference(x.uncert, correct.uncert, tol)
    
    def testSolveTwoLinearequations(self):
        a = variable(23.7, '', 0.1)
        b = variable(943, '', 12.5)
        c = variable(7.5, '', 0.05)
        d = variable(638, '', 19.7)
        e = variable(293.4, '', 0.3)
        f = variable(156.2, '', 4.2)

        correctY = (f - d*c/a) / (e - d*b/a)
        correctX = (c - b*correctY) / a
        
        def func(x,y):
            eq1 = [a * x + b * y, c]
            eq2 = [d * x + e * y, f]
            eqs = [eq1, eq2]
            return eqs
        
        x0 = [variable(1,''), variable(1,'')]
        x,y = solve(func, x0, tol = solveTol)
                
        self.assertRelativeDifference(x.value, correctX.value, tol)
        self.assertRelativeDifference(y.value, correctY.value, tol)
        self.assertRelativeDifference(x.uncert, correctX.uncert, tol)
        self.assertRelativeDifference(y.uncert, correctY.uncert, tol)

    def testSolveTwoNonlinearEquations(self):
        a = variable(23.7, '', 0.1)
        b = variable(943, '', 12.5)
        c = variable(7.5, '', 0.05)
        d = variable(638, '', 19.7)
        e = variable(293.4, '', 0.3)
        f = variable(156.2, '', 4.2)

        ## correct values were found in EES
        correctX=variable(0.244808471375732378, '' ,0.0100245696158427206) 
        correctY=variable(0.0064471164978235798 , '', 0.000159290537914747218) 

        def func(x,y):
            eqs = []
            eq1 = [a * x**2 + b * y, c]
            eq2 = [d * x + e * y**2, f]
            eqs = [eq1, eq2]
            return eqs
        
        x0 = [variable(1,''), variable(1,'')]
        x,y = solve(func, x0, tol = solveTol)
                
        self.assertRelativeDifference(x.value, correctX.value, tol)
        self.assertRelativeDifference(y.value, correctY.value, tol)
        self.assertRelativeDifference(x.uncert, correctX.uncert, tol)
        self.assertRelativeDifference(y.uncert, correctY.uncert, tol)
                
    def testSolveOneLinearEquationWithDifferentUnits(self):
        a = variable(23.7, 'L/min', 0.1)
        b = variable(943, 'm3/h', 12.5)
        def func(x):
            return [a * x, b]
        
        x = solve(func, variable(1,''), tol = solveTol)
        correct = b / a
        
        self.assertRelativeDifference(x.value, correct.value, tol)
        self.assertRelativeDifference(x.uncert, correct.uncert, tol)

    def testSolveOneNonlinearEquationWithDifferentUnits(self):
        a = variable(23.7, 'L/min', 0.1)
        b = variable(943, 'm3/h', 12.5)
        
        def func(x):
            return [a * x**2, b]
        
        x = solve(func, variable(1,''), tol = solveTol)
        correct = (b / a)**(1/2)
        
        self.assertRelativeDifference(x.value, correct.value, tol)
        self.assertRelativeDifference(x.uncert, correct.uncert, tol)
    
    def testSolveTwoLinearEquationsWithDifferentUnits(self):
        a = variable(23.7, 'L/min', 0.1)
        b = variable(943, 'm3/h', 12.5)
        c = variable(7.5, 'L/s', 0.05)
        d = variable(638, 'L/h', 19.7)
        e = variable(293.4, 'm3/s', 0.3)
        f = variable(156.2, 'm3/min', 4.2)
        
        
        def func(x,y):
            eqs = []
            eq1 = [a * x + b * y, c]
            eq2 = [d * x + e * y, f]
            eqs = [eq1, eq2]
            return eqs
        
        x0 = [variable(14,''), variable(1,'')]
        x,y = solve(func, x0, tol = 1e-10)

        correctY = (f - d*c/a) / (e - d*b/a)
        correctX = (c - b*correctY) / a
            
        self.assertRelativeDifference(x.value, correctX.value, 1e1 * tol)
        self.assertRelativeDifference(y.value, correctY.value, tol)
        self.assertRelativeDifference(x.uncert, correctX.uncert, tol)
        self.assertRelativeDifference(y.uncert, correctY.uncert, tol)

    def testSolveTwoNonlinearEquationsWithDifferentUnits(self):
        a = variable(0.237, 'L/min', 0.1)
        b = variable(943, 'm3/h', 12.5)
        c = variable(7.5, 'L/s', 0.05)
        d = variable(638, 'L/h', 19.7)
        e = variable(0.2934, 'm3/s', 0.3)
        f = variable(156.2, 'm3/min', 4.2)

        ## correct values were found in EES
        correctX=variable(443.2067798094616700,'',144.9960332892821860) 
        correctY=variable(-2.93347464506229083,'',1.48884088824273296) 

        def func(x,y):
            eqs = []
            eq1 = [a * x**2 + b * y, c]
            eq2 = [d * x + e * y**2, f]
            eqs = [eq1, eq2]
            return eqs
        
        x0 = [variable(1,''), variable(1,'')]
        x,y = solve(func, x0, tol = solveTol)
                
        self.assertRelativeDifference(x.value, correctX.value, tol)
        self.assertRelativeDifference(y.value, correctY.value, tol)
        self.assertRelativeDifference(x.uncert, correctX.uncert, tol)
        self.assertRelativeDifference(y.uncert, correctY.uncert, tol)
   
    def testSolveOneNonlinearEquationWithBounds1(self):
        lbs = [variable(-10,'L/min'), variable(10, 'L/min')]
        ubs = [variable(5,'L/min'), variable(100, 'L/min')]
        
        
        for lb in lbs:
            for ub in ubs:
                lb, ub = min([lb, ub]), max([lb,ub])
                bounds = [lb,ub]
                
                a = variable(23.7, 'mbar-min2/L2', 0.1)
                b = variable(943, 'mbar', 12.5)
                correct = (b / a)**(1/2)
                correct = [correct, -correct]
                correct = [np.min([np.max([lb, elem]), ub]) for elem in correct]
                
                def func(x):
                    return [a * x**2, b]

                    
                x = solve(func, variable(100,'L/min'), bounds = bounds, tol = solveTol)
                
                minIndex = np.argmin([abs((elem - x).value) for elem in correct])
                correct = correct[minIndex]
                self.assertRelativeDifference(x.value, correct.value, tol)

    def testSolveOneNonelinearEquationWithBounds2(self):
        tii = variable(50,'')
        tio = variable(30)
        toi = variable(15)
        # too = variable(45)
        lmdt = variable(25)

        def func(too):
            dt1 = tii - toi
            dt2 = too - tio
            return [lmdt, (dt1 - dt2) / (np.log(dt1) - np.log(dt2))]
        
        def bounds(too):
            return [tio + 1e-6, too, variable(np.inf)]
        
        x0 = variable(-16,'')
        x = solve(func, x0, tol = 1e-10, bounds = bounds)
        
        eq = func(x)
        residual = eq[0] - eq[1]
        self.assertAlmostEqual(residual.value,0)

    def testSolveOneNonelinearEquationWithBounds3(self):
        tii = variable(50,'')
        tio = variable(30)
        # toi = variable(15)
        too = variable(45)
        lmdt = variable(25)

        def func(toi):
            dt1 = tii - toi
            dt2 = too - tio
            return [lmdt, (dt1 - dt2) / (np.log(dt1) - np.log(dt2))]
        
        def bounds(toi):
            return [variable(-np.inf), toi, tii]
        
        x0 = variable(-16,'')
        x = solve(func, x0, tol = 1e-10, bounds = bounds)
        
        eq = func(x)
        residual = eq[0] - eq[1]
        self.assertAlmostEqual(residual.value,0)
    
    def testSolveOneNonelinearEquationWithBounds4(self):
        tii = variable(50,'')
        # tio = variable(30)
        toi = variable(15)
        too = variable(45)
        lmdt = variable(25)

        def func(tio):
            dt1 = tii - toi
            dt2 = too - tio
            return [lmdt, (dt1 - dt2) / (np.log(dt1) - np.log(dt2))]
        
        def bounds(tio):
            return [variable(-np.inf), tio, too]
        
        x0 = variable(-16,'')
        x = solve(func, x0, tol = 1e-10, bounds = bounds)
        
        eq = func(x)
        residual = eq[0] - eq[1]
        self.assertAlmostEqual(residual.value,0)
    
    def testSolveOneNonelinearEquationWithBounds4(self):
        # tii = variable(50,'')
        tio = variable(30)
        toi = variable(15)
        too = variable(45)
        lmdt = variable(25)

        def func(tii):
            dt1 = tii - toi
            dt2 = too - tio
            return [lmdt, (dt1 - dt2) / (np.log(dt1) - np.log(dt2))]
        
        def bounds(tii):
            return [toi + 1e-6, tii, variable(np.inf)]
        
        x0 = variable(-16,'')
        x = solve(func, x0, tol = 1e-10, bounds = bounds)
        
        eq = func(x)
        residual = eq[0] - eq[1]
        self.assertAlmostEqual(residual.value,0)

    def testSolveOneNonlinearEquationWithBoundsUsingDifferentUnits1(self):
                
        a = variable(23.7, 'mbar-min2/L2', 0.1)
        b = variable(943, 'mbar', 12.5)
        correct = (b / a)**(1/2)
        
        def func(x):
            return [a * x**2, b]

        bounds = [variable(0.06, 'm3/h'), variable(1.667,'L/s')]
            
        x = solve(func, variable(100,'L/min'), bounds = bounds, tol = solveTol)
        
        self.assertRelativeDifference(x.value, correct.value, tol)
    
    def testSolveOneNonlinearEquationWithBoundsUsingDifferentUnits2(self):
                
        a = variable(23.7, 'mbar-min2/L2', 0.1)
        b = variable(943, 'mbar', 12.5)
        correct = (b / a)**(1/2)
        
        def func(x):
            return [a * x**2, b]

        def bounds(x):
            return [variable(0.06, 'm3/h'), x, variable(1.667,'L/s')]
            
        x = solve(func, variable(100,'L/min'), bounds = bounds, tol = solveTol)
        
        self.assertRelativeDifference(x.value, correct.value, tol)

    def testSolveOneEquationUsingProp(self):
        
        p = variable(1,'bar', 0.01)
        c = variable(50,'%')
        t_in = variable(60,'C', 1.2)
        phi = variable(120, 'kW', 2.3)
        flow = variable(350, 'L/min', 9.1)
        
        def func(t_out):
            dt = t_in - t_out
            t_avg = t_in + dt / 2
            rho = prop('density', 'MEG', P = p, T = t_avg, C = c)
            cp = prop('specific_heat', 'MEG', P = p, T = t_avg, C = c)
            phi_calc = rho * cp * flow * (t_in - t_out)
            phi_calc.convert('kW')
            return [phi, phi_calc]

        x0 = variable(20, 'C')
        t_out = solve(func, x0, tol = solveTol)
        
        self.assertRelativeDifference(t_out.value, 54.36480373585032900, tol)
        self.assertEqual(t_out.unit, 'C')
        self.assertRelativeDifference(t_out.uncert, 1.217157659256291610, tol)
     
    def testSolveOneNonlinearEquationWithVectors(self):
        a = variable([23.7, 12.3], '', [0.1, 0.05])
        b = variable([943, 793], '', [12.5, 9.4])
        
        def func(x):
            return [a * x**2, b]
        
        x = solve(func, variable([10, 10],''), tol = solveTol)
        correct = (b / a)**(1/2)
        
        for i in range(len(x)):
            self.assertRelativeDifference(x[i].value, correct[i].value, tol)
            self.assertRelativeDifference(x[i].uncert, correct[i].uncert, tol)

    def testSolveOneNonlinearEquationWithVectorsWithBounds1(self):
        a = variable([23.7, 12.3], '', [0.1, 0.05])
        b = variable([943, 793], '', [12.5, 9.4])
        
        def func(x):
            return [a * x**2, b]
        
        lower = variable([10,10])
        def bounds(x):
            return [lower, x, variable([np.inf, np.inf])]
            
        x = solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        correct = (b / a)**(1/2)
        
        for c, l in zip(correct, lower):
            if c < l:
                c._value = l.value
        
        for i in range(len(x)):
            self.assertRelativeDifference(x[i].value, correct[i].value, tol)

    def testSolveOneNonlinearEquationWithVectorsWithBounds2(self):
        a = variable([23.7, 12.3], '', [0.1, 0.05])
        b = variable([943, 793], '', [12.5, 9.4])
        
        def func(x):
            return [a * x**2, b]
        
        lower = variable([10,10])
        upper = variable([np.inf, np.inf])
        bounds = [lower, upper]
            
        x = solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        correct = (b / a)**(1/2)
        
        for c, l in zip(correct, lower):
            if c < l:
                c._value = l.value
        
        for i in range(len(x)):
            self.assertRelativeDifference(x[i].value, correct[i].value, tol)

    def testCallableBoundInputs(self):
        a = variable([23.7, 12.3], '', [0.1, 0.05])
        b = variable([943, 793], '', [12.5, 9.4])
        
        def func(x):
            return [a * x**2, b]
        
        upper = variable([np.inf, np.inf])
        lower = variable([10,10],'m')
        def bounds(x):
            return [lower, x, upper]
        with self.assertRaises(Exception) as context:
            solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("The units of the bounds does not match" in str(context.exception))
                              
        lower = variable([10])
        def bounds(x):
            return [lower, x, upper]
        with self.assertRaises(Exception) as context:
            solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("Each element of the bounds has to have the same length" in str(context.exception))
        
        lower = variable(10)
        def bounds(x):
            return [lower, x, upper]
        with self.assertRaises(Exception) as context:
            solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("Each element of the bounds has to have the same length" in str(context.exception))
   
    
        lower = variable([-np.inf, -np.inf])
        upper = variable([10,10],'m')
        def bounds(x):
            return [lower, x, upper]
        with self.assertRaises(Exception) as context:
            solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("The units of the bounds does not match" in str(context.exception))
                
        upper = variable([10])
        def bounds(x):
            return [lower, x, upper]
        with self.assertRaises(Exception) as context:
            solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("Each element of the bounds has to have the same length" in str(context.exception))
        
        upper = variable(10)
        def bounds(x):
            return [lower, x, upper]
        with self.assertRaises(Exception) as context:
            solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("Each element of the bounds has to have the same length" in str(context.exception))
        
        lower = variable([-np.inf, -np.inf])
        upper = variable([np.inf, np.inf])
        with self.assertRaises(Exception) as context:
            solve(func, variable([20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("You supplied 2 equations but 1 variables. The number of equations and the vairables has to match" in str(context.exception))
        
        with self.assertRaises(Exception) as context:
            solve(func, variable(20,''), tol = solveTol, bounds=bounds)
        self.assertTrue("You supplied 2 equations but 1 variables. The number of equations and the vairables has to match" in str(context.exception))

    def testNonCallableBoundInputs(self):
        a = variable([23.7, 12.3], '', [0.1, 0.05])
        b = variable([943, 793], '', [12.5, 9.4])
        
        def func(x):
            return [a * x**2, b]
        
        upper = variable([np.inf, np.inf])
        lower = variable([10,10],'m')
        bounds = [lower,upper]
        with self.assertRaises(Exception) as context:
            solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("The units of the bounds does not match" in str(context.exception))
                              
        lower = variable([10])
        bounds = [lower,upper]  
        with self.assertRaises(Exception) as context:
            solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("Each element of the bounds has to have the same length" in str(context.exception))
        
        lower = variable(10)
        bounds = [lower,upper]  
        with self.assertRaises(Exception) as context:
            solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("Each element of the bounds has to have the same length" in str(context.exception))
   
    
        lower = variable([-np.inf, -np.inf])
        upper = variable([10,10],'m')
        bounds = [lower,upper]  
        with self.assertRaises(Exception) as context:
            solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("The units of the bounds does not match" in str(context.exception))
                
        upper = variable([10])
        bounds = [lower,upper]  
        with self.assertRaises(Exception) as context:
            solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("Each element of the bounds has to have the same length" in str(context.exception))
        
        upper = variable(10)
        bounds = [lower,upper]  
        with self.assertRaises(Exception) as context:
            solve(func, variable([20, 20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("Each element of the bounds has to have the same length" in str(context.exception))
        
        lower = variable([-np.inf, -np.inf])
        upper = variable([np.inf, np.inf])
        bounds = [lower,upper]  
        with self.assertRaises(Exception) as context:
            solve(func, variable([20],''), tol = solveTol, bounds=bounds)
        self.assertTrue("You supplied 2 equations but 1 variables. The number of equations and the vairables has to match" in str(context.exception))
        
        with self.assertRaises(Exception) as context:
            solve(func, variable(20,''), tol = solveTol, bounds=bounds)
        self.assertTrue("You supplied 2 equations but 1 variables. The number of equations and the vairables has to match" in str(context.exception))

    def testAlreadySolvedEquation(self):
        a = 1.2
        b = 2.3
        total = variable(23.0)
        
        def func(x1, x2):
            eq1 = [x1 + x2, total]
            eq2 = [a*x1, b*x2]
            return [eq1, eq2]
            
        x0 = [variable(20), variable(3)]
        x = solve(func, x0)
    
    def testNonVariableEquation(self):
        a = 1.2
        b = 2.3
        total = 23
        
        def func(x1, x2):
            eq1 = [x1 + x2, total]
            eq2 = [a*x1, b*x2]
            return [eq1, eq2]
            
        x0 = [variable(20), variable(3)]
        with self.assertRaises(Exception) as context:
            solve(func, x0)
        self.assertTrue("The Right side of equation 1 is not a variable. Both side of each equation has to be a variable" in str(context.exception))
   
    def testNumberOfSidesInEquations(self):
        a = 1.2
        b = 2.3
        total = variable(23)
        
        def func(x1, x2):
            eq1 = [x1 + x2, total]
            eq2 = [a*x1, b*x2, 3]
            return [eq1, eq2]
            
        x0 = [variable(20), variable(3)]
        with self.assertRaises(Exception) as context:
            solve(func, x0)
        self.assertTrue("Equation 2 is a list of 3 elements. This corresponds with an equation with 3 sides. All equations has to have 2 sides" in str(context.exception))
   
        
        def func(x1, x2):
            eq1 = [x1 + x2]
            eq2 = [a*x1, b*x2]
            return [eq1, eq2]
            
        x0 = [variable(20), variable(3)]
        with self.assertRaises(Exception) as context:
            solve(func, x0)
        self.assertTrue("Equation 1 is a list of 1 elements. This corresponds with an equation with 1 sides. All equations has to have 2 sides" in str(context.exception))
   
    def testCovariance(self):
        a = variable(23.7, '', 0.1)
        b = variable(943, '', 12.5)
        a.addCovariance(b, -3.2, '1')
        def func(x):
            return [a * x, b]
        
        x = solve(func, variable(1,''), tol = solveTol)
        correct = b / a

        self.assertRelativeDifference(x.value, correct.value, tol)
        self.assertRelativeDifference(x.uncert, correct.uncert, tol)


    def testSolveParametric(self):
        n = 20
    
        a = variable(9, '', 0.3)
        B = variable(np.linspace(23, 15, n), '', np.linspace(0.2, 0.07, n))
        x0 = variable(1)
        
        def func(x, b):
            return [a * x, b]
        
        correct = B / a
        x = solve(func, x0, parametric=B, tol = solveTol)
        for i in range(n):
            self.assertRelativeDifference(x.value[i], correct.value[i], tol)
            self.assertRelativeDifference(x.uncert[i], correct.uncert[i], tol)

    def testSolveWithListOfParametrics(self):
        n = 20
    
        A = variable(np.linspace(3, 14,n), '', np.linspace(0.3, 1.7, n))
        B = variable(np.linspace(23, 15, n), '', np.linspace(0.2, 0.07, n))
        x0 = variable(1)
        
        def func(x, a, b):
            return [a * x, b]
        
        correct = B / A
        x = solve(func, x0, parametric=[A,B], tol = solveTol)
        for i in range(n):
            self.assertRelativeDifference(x.value[i], correct.value[i], tol)
            self.assertRelativeDifference(x.uncert[i], correct.uncert[i], tol)

    def testSolveWithParametricsAndBounds(self):
        n = 20
    
        a = variable(9, '', 0.3)
        B = variable(np.linspace(-5, 15, n), '', np.linspace(0.2, 0.07, n))
        x0 = variable(1)
        
        def bounds(x):
            return [variable(0), x, variable(1)]
        
        def func(x, b):
            return [a * x, b]
        
        correct = B / a
        x = solve(func, x0, parametric=[B], tol = solveTol, bounds = bounds)
        
        for i in range(n):
            checkUncert = True
            if correct[i] < 0:
                correct[i]._value = 0
                checkUncert = False
            if correct[i] > 1:
                correct[i]._value = 1
                checkUncert = False

            self.assertRelativeDifference(x.value[i], correct.value[i], tol)
            if checkUncert:
                ## TODO The uncertanty of an output may not be correct as the value is moved in to the feasible range
                self.assertRelativeDifference(x.uncert[i], correct.uncert[i], tol)

    def testSolveWithParametricsWithDifferentUnits(self):
        n = 20
    
        a = variable(9, 'L/min', 0.3)
        B = variable(np.linspace(23, 15, n), 'm3', np.linspace(0.2, 0.07, n))
        x0 = variable(1, 'h')

        def func(x, b):
            return [a * x, b]
        
        correct = B / a
        correct.convert('h')

        x = solve(func, x0, parametric=B, tol = solveTol)

        x.convert('min')
        correct.convert('min')

        for i in range(n):
            self.assertRelativeDifference(x.value[i], correct.value[i], tol)
            self.assertRelativeDifference(x.uncert[i], correct.uncert[i], tol)

    def testBadlyFormulatedSetOfEquations(self):
            
        from sheet import sheetsFromFile
        from prop import prop

        data = sheetsFromFile("testData/solveData.xlsx", dataRange = "A-AG", uncertRange = "AH-BN")
        
        
        ## override the relative desity the sensor was not connected during this test
        for elem in data.rh_air:
            elem._value = 60
            elem._uncert = 0

        ## calculate the average of the inlet and outlet temperature on the air side
        ## NOTE: the measurement t_air_out_12 was faulty. Therefore the ranges has been set to 11 and not 12
        data.t_air_in = sum([getattr(data, f't_air_in_{str(i+1).zfill(2)}') for i in range(12)]) / 12
        data.t_air_out = sum([getattr(data, f't_air_out_{str(i+1).zfill(2)}') for i in range(11)]) / 11

        d_nozzle = variable(225, 'mm')
        dp_nozzle = data.dP_nozzle
        p_atm = data.p_atm
        T_5 = data.t_air_out 
        T_6 = data.t_air_out 
        P_s5 = data.p_atm - data.dP_air
        P_s6 = data.p_atm - data.dP_air - data.dP_nozzle
        rh_5 = data.rh_air
        rh_6 = data.rh_air
        C_initial = 1
        
        
        ## determine all necessary constants
        alpha = (P_s6 + p_atm) / (P_s5 + p_atm)         ## Eq. 7.11 AMCA 210
        # alpha = P_s6 / P_s5

        ## Ratio of the diameters of the nozzles
        ## Beta is zero due to the chamber approach
        beta = 0
        
        ## ratio of the specific heats of the air
        ## this is set to 1.4 as a constant
        gamma =  1.4

        ## detemine the expansion factor, Y             ## Eq. 7.14 AMCA 210
        Y = np.sqrt(((gamma)/(gamma-1)) * (alpha **(2/gamma)) * ((1-alpha**((gamma-1)/gamma))/(1 - alpha)) * ((1 - beta**4)/(1 - beta**4*alpha**(2/gamma))))

        def SolveDischargeCoefficientAndReynoldsNumber(C,Re, y, rho5, mu6, dp_nozzle):

            Re2 = C * d_nozzle * y / mu6 * np.sqrt(2 * dp_nozzle * rho5)
            C2 = 0.9986 - (7.006/np.sqrt(Re)) + (134.6/Re)
            
            eq1 = [C, C2]
            eq2 = [Re, Re2]
            
            return [eq1, eq2]
            
            
        rho_5 = prop('density', 'air', T = T_5, P = P_s5, rh = rh_5)
        mu_6 = prop('dynamic_viscosity', 'air', T = T_6, P = P_s6, rh = rh_6)
        
        ## unbounded set of equations with bad initial guess
        with self.assertRaises(Exception) as context:
            C, Re = solve(
                SolveDischargeCoefficientAndReynoldsNumber,
                [variable(C_initial), variable(2e5)], 
                tol = 1e-100,
                parametric = [Y, rho_5, mu_6, dp_nozzle]
            )  
        self.assertEqual(str(context.exception), "The solver encountered a NaN value of the minimization function. Try to change the initial value or add boundaries to the variables.")

        ## better initial guesses
        C, Re = solve(
            SolveDischargeCoefficientAndReynoldsNumber,
            [variable(C_initial), variable(1e3)], 
            tol = 1e-100,
            parametric = [Y, rho_5, mu_6, dp_nozzle]
        )  

        ## included bounds
        C, Re = solve(
            SolveDischargeCoefficientAndReynoldsNumber,
            [variable(C_initial), variable(2e5)], 
            tol = 1e-100,
            parametric = [Y, rho_5, mu_6, dp_nozzle],
            bounds = [[variable(1e-10), variable(np.inf)],[variable(1e-10), variable(np.inf)]]
        )  
    

if __name__ == '__main__':
    unittest.main()
    
