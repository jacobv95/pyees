import unittest
import numpy as np
import openpyxl
try:
    from prop import prop
    from variable import variable
except ImportError:
    from pyees.prop import prop
    from pyees.variable import variable

class test(unittest.TestCase):

    def testAll(self):
        
        eesData = openpyxl.load_workbook('testData/testPropData.xlsx').active        
        def compareVariableAndEESData(var, row, col):
            data = eesData.cell(row+1, col+1).value
            index = data.index('±')
            value = float(data[0:index])
            uncert = float(data[index+1:])
            self.assertAlmostEqual(var.value, value)
            self.assertAlmostEqual(var.uncert, uncert, 5)
        
        C = variable([50,40], '%', [2,2])
        P = variable([1, 0.5], 'bar', [0.1, 0.1])
        T = variable([20, 30], 'C', [1,1])
        
        
        ## test scalar value inputs
        ## loop over all combinations of the concentration, pressure and temperature.
        row = 0
        for c in C:
            for p in P:
                for t in T:
                    rho_water = prop('density', 'water', T = t, P = p)       
                    compareVariableAndEESData(rho_water, row,5)
                    
                    cp_water = prop('specific_heat', 'water', T = t, P = p)
                    cp_water.convert('kJ/kg-K')       
                    compareVariableAndEESData(cp_water, row, 1)
                    
                    mu_water = prop('dynamic_viscosity', 'water', T = t, P = p)       
                    compareVariableAndEESData(mu_water, row, 3)
                    
                    rho_meg = prop('density', 'MEG', T = t, P = p, C = c)       
                    compareVariableAndEESData(rho_meg, row, 4)
                    
                    cp_meg = prop('specific_heat', 'MEG', T = t, P = p, C = c)       
                    cp_meg.convert('kJ/kg-K')
                    compareVariableAndEESData(cp_meg, row, 0)

                    mu_meg = prop('dynamic_viscosity', 'MEG', T = t, P = p, C = c)       
                    compareVariableAndEESData(mu_meg, row, 2)
                    
                    row += 1


        ## supply an arrayVariable for the temperature
        for i,c in enumerate(C):
            for j, p in enumerate(P):
                
                row1 = 0 + 4*i + 2*j
                row2 = 1 + 4*i + 2*j
                
                rho_water = prop('density', 'water', T = T, P = p)
                compareVariableAndEESData(rho_water[0], row1, 5)
                compareVariableAndEESData(rho_water[1], row2, 5)
                
                cp_water = prop('specific_heat', 'water', T = T, P = p)
                cp_water.convert('kJ/kg-K')       
                compareVariableAndEESData(cp_water[0], row1, 1)
                compareVariableAndEESData(cp_water[1], row2, 1)
                
                mu_water = prop('dynamic_viscosity', 'water', T = T, P = p)       
                compareVariableAndEESData(mu_water[0], row1, 3)
                compareVariableAndEESData(mu_water[1], row2, 3)
                
                rho_meg = prop('density', 'MEG', T = T, P = p, C = c)       
                compareVariableAndEESData(rho_meg[0], row1, 4)
                compareVariableAndEESData(rho_meg[1], row2, 4)
                
                cp_meg = prop('specific_heat', 'MEG', T = T, P = p, C = c)       
                cp_meg.convert('kJ/kg-K')
                compareVariableAndEESData(cp_meg[0], row1, 0)
                compareVariableAndEESData(cp_meg[1], row2, 0)

                mu_meg = prop('dynamic_viscosity', 'MEG', T = T, P = p, C = c)       
                compareVariableAndEESData(mu_meg[0], row1, 2)
                compareVariableAndEESData(mu_meg[1], row2, 2)
          
                
        ## supply an arrayVariable for the pressure
        for i,c in enumerate(C):
            for j,t in enumerate(T):
                
                row1 = 0 + 4*i + 1*j
                row2 = 2 + 4*i + 1*j
                
                rho_water = prop('density', 'water', T = t, P = P)
                compareVariableAndEESData(rho_water[0], row1, 5)
                compareVariableAndEESData(rho_water[1], row2, 5)
                
                cp_water = prop('specific_heat', 'water', T = t, P = P)
                cp_water.convert('kJ/kg-K')       
                compareVariableAndEESData(cp_water[0], row1, 1)
                compareVariableAndEESData(cp_water[1], row2, 1)
                
                mu_water = prop('dynamic_viscosity', 'water', T = t, P = P)       
                compareVariableAndEESData(mu_water[0], row1, 3)
                compareVariableAndEESData(mu_water[1], row2, 3)
                
                rho_meg = prop('density', 'MEG', T = t, P = P, C = c)       
                compareVariableAndEESData(rho_meg[0], row1, 4)
                compareVariableAndEESData(rho_meg[1], row2, 4)
                
                cp_meg = prop('specific_heat', 'MEG', T = t, P = P, C = c)       
                cp_meg.convert('kJ/kg-K')
                compareVariableAndEESData(cp_meg[0], row1, 0)
                compareVariableAndEESData(cp_meg[1], row2, 0)

                mu_meg = prop('dynamic_viscosity', 'MEG', T = t, P = P, C = c)       
                compareVariableAndEESData(mu_meg[0], row1, 2)
                compareVariableAndEESData(mu_meg[1], row2, 2)
                
        ## supply an arrayVariable for the concentration
        for i,p in enumerate(P):
            for j,t in enumerate(T):
                
                row1 = 0 + 2*i + 1*j
                row2 = 4 + 2*i + 1*j
                                
                rho_meg = prop('density', 'MEG', T = t, P = p, C = C)       
                compareVariableAndEESData(rho_meg[0], row1, 4)
                compareVariableAndEESData(rho_meg[1], row2, 4)
                
                cp_meg = prop('specific_heat', 'MEG', T = t, P = p, C = C)       
                cp_meg.convert('kJ/kg-K')
                compareVariableAndEESData(cp_meg[0], row1, 0)
                compareVariableAndEESData(cp_meg[1], row2, 0)

                mu_meg = prop('dynamic_viscosity', 'MEG', T = t, P = p, C = C)       
                compareVariableAndEESData(mu_meg[0], row1, 2)
                compareVariableAndEESData(mu_meg[1], row2, 2)


        ## supply an arrayVariable for the temprature and the pressure
        for i, c in enumerate(C):
            row1 = 0 + 4*i
            row2 = 3 + 4*i
            
            rho_water = prop('density', 'water', T = T, P = P)
            compareVariableAndEESData(rho_water[0], row1, 5)
            compareVariableAndEESData(rho_water[1], row2, 5)
            
            cp_water = prop('specific_heat', 'water', T = T, P = P)
            cp_water.convert('kJ/kg-K')       
            compareVariableAndEESData(cp_water[0], row1, 1)
            compareVariableAndEESData(cp_water[1], row2, 1)
            
            mu_water = prop('dynamic_viscosity', 'water', T = T, P = P)       
            compareVariableAndEESData(mu_water[0], row1, 3)
            compareVariableAndEESData(mu_water[1], row2, 3)
            
            rho_meg = prop('density', 'MEG', T = T, P = P, C = c)       
            compareVariableAndEESData(rho_meg[0], row1, 4)
            compareVariableAndEESData(rho_meg[1], row2, 4)
            
            cp_meg = prop('specific_heat', 'MEG', T = T, P = P, C = c)       
            cp_meg.convert('kJ/kg-K')
            compareVariableAndEESData(cp_meg[0], row1, 0)
            compareVariableAndEESData(cp_meg[1], row2, 0)

            mu_meg = prop('dynamic_viscosity', 'MEG', T = T, P = P, C = c)       
            compareVariableAndEESData(mu_meg[0], row1, 2)
            compareVariableAndEESData(mu_meg[1], row2, 2)
        
        ## supply an arrayVariable for the temprature and the concentration
        for i, p in enumerate(P):
            row1 = 0 + 2*i
            row2 = 5 + 2*i
            
            rho_water = prop('density', 'water', T = T, P = p)
            compareVariableAndEESData(rho_water[0], row1, 5)
            compareVariableAndEESData(rho_water[1], row2, 5)
            
            cp_water = prop('specific_heat', 'water', T = T, P = p)
            cp_water.convert('kJ/kg-K')       
            compareVariableAndEESData(cp_water[0], row1, 1)
            compareVariableAndEESData(cp_water[1], row2, 1)
            
            mu_water = prop('dynamic_viscosity', 'water', T = T, P = p)       
            compareVariableAndEESData(mu_water[0], row1, 3)
            compareVariableAndEESData(mu_water[1], row2, 3)
            
            rho_meg = prop('density', 'MEG', T = T, P = p, C = C)       
            compareVariableAndEESData(rho_meg[0], row1, 4)
            compareVariableAndEESData(rho_meg[1], row2, 4)
            
            cp_meg = prop('specific_heat', 'MEG', T = T, P = p, C = C)       
            cp_meg.convert('kJ/kg-K')
            compareVariableAndEESData(cp_meg[0], row1, 0)
            compareVariableAndEESData(cp_meg[1], row2, 0)

            mu_meg = prop('dynamic_viscosity', 'MEG', T = T, P = p, C = C)       
            compareVariableAndEESData(mu_meg[0], row1, 2)
            compareVariableAndEESData(mu_meg[1], row2, 2)
        
        ## supply an arrayVariable for the pressure and the concentration
        for i, t in enumerate(T):
            row1 = 0 + 1*i
            row2 = 6 + 1*i
            
            rho_water = prop('density', 'water', T = t, P = P)
            compareVariableAndEESData(rho_water[0], row1, 5)
            compareVariableAndEESData(rho_water[1], row2, 5)
            
            cp_water = prop('specific_heat', 'water', T = t, P = P)
            cp_water.convert('kJ/kg-K')       
            compareVariableAndEESData(cp_water[0], row1, 1)
            compareVariableAndEESData(cp_water[1], row2, 1)
            
            mu_water = prop('dynamic_viscosity', 'water', T = t, P = P)       
            compareVariableAndEESData(mu_water[0], row1, 3)
            compareVariableAndEESData(mu_water[1], row2, 3)
            
            rho_meg = prop('density', 'MEG', T = t, P = P, C = C)       
            compareVariableAndEESData(rho_meg[0], row1, 4)
            compareVariableAndEESData(rho_meg[1], row2, 4)
            
            cp_meg = prop('specific_heat', 'MEG', T = t, P = P, C = C)       
            cp_meg.convert('kJ/kg-K')
            compareVariableAndEESData(cp_meg[0], row1, 0)
            compareVariableAndEESData(cp_meg[1], row2, 0)

            mu_meg = prop('dynamic_viscosity', 'MEG', T = t, P = P, C = C)       
            compareVariableAndEESData(mu_meg[0], row1, 2)
            compareVariableAndEESData(mu_meg[1], row2, 2)
        
        ## suppliy array variables for all inputs
        row1 = 0
        row2 = 7
        rho_water = prop('density', 'water', T = T, P = P)
        compareVariableAndEESData(rho_water[0], row1, 5)
        compareVariableAndEESData(rho_water[1], row2, 5)
        
        cp_water = prop('specific_heat', 'water', T = T, P = P)
        cp_water.convert('kJ/kg-K')       
        compareVariableAndEESData(cp_water[0], row1, 1)
        compareVariableAndEESData(cp_water[1], row2, 1)
        
        mu_water = prop('dynamic_viscosity', 'water', T = T, P = P)       
        compareVariableAndEESData(mu_water[0], row1, 3)
        compareVariableAndEESData(mu_water[1], row2, 3)
        
        rho_meg = prop('density', 'MEG', T = T, P = P, C = C)       
        compareVariableAndEESData(rho_meg[0], row1, 4)
        compareVariableAndEESData(rho_meg[1], row2, 4)
        
        cp_meg = prop('specific_heat', 'MEG', T = T, P = P, C = C)       
        cp_meg.convert('kJ/kg-K')
        compareVariableAndEESData(cp_meg[0], row1, 0)
        compareVariableAndEESData(cp_meg[1], row2, 0)

        mu_meg = prop('dynamic_viscosity', 'MEG', T = T, P = P, C = C)       
        compareVariableAndEESData(mu_meg[0], row1, 2)
        compareVariableAndEESData(mu_meg[1], row2, 2)

if __name__ == '__main__':
    unittest.main()
