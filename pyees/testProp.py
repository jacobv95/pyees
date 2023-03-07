import unittest
try:
    from prop import prop
    from variable import variable
except ImportError:
    from pyees.prop import prop
    from pyees.variable import variable

class test(unittest.TestCase):

    def testAll(self):

        T = variable(30, 'C', 1)
        P = variable(1, 'bar', 0.01)

        rho = prop('density', 'water', T = T, P = P)       
        self.assertAlmostEqual(rho.value, 995.6488633254202760)
        self.assertAlmostEqual(rho.uncert, 0.302056032124829521, 5)
        
        
        cp = prop('specific_heat', 'water', T = T, P = P)       
        self.assertAlmostEqual(cp.value, 4179.823274031785670)
        self.assertAlmostEqual(cp.uncert, 0.197230658613685986, 5)
        
        mu = prop('dynamic_viscosity', 'water', T = T, P = P)       
        self.assertAlmostEqual(mu.value, 0.000797221826825984337)
        self.assertAlmostEqual(mu.uncert, 0.0000169744084027323477, 5)
        
        C = variable(50,'%', 2)
        rho = prop('density', 'MEG', T = T, P = P, C=C)       
        self.assertAlmostEqual(rho.value, 1059.388204247423450)
        self.assertAlmostEqual(rho.uncert, 2.488866117279996050, 5)
        
        cp = prop('specific_heat', 'MEG', T = T, P = P, C=C)       
        self.assertAlmostEqual(cp.value, 3363.55043611633498)
        self.assertAlmostEqual(cp.uncert, 39.8428214910969325, 5)

        mu = prop('dynamic_viscosity', 'MEG', T = T, P = P, C=C)       
        self.assertAlmostEqual(mu.value, 0.00272865313721519455)
        self.assertAlmostEqual(mu.uncert, 0.000163091267222627309, 5)
        

if __name__ == '__main__':
    unittest.main()
