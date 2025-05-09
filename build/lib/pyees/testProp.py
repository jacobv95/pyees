import unittest
import openpyxl


from prop import prop
from variable import variable

class test(unittest.TestCase):

    def assertRelativeDifference(self, a, b, r):
        assert abs(a-b) <= abs(b * r), f"The value {a} and {b} has a greater relative difference than {r}. The difference was {abs(a-b)} and was allowed to be {b*r}"
    

    def testWaterAndMEG(self):
        
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
                    
                    rho_meg = prop('density', 'MEG', T = t, P = p, C=c)       
                    compareVariableAndEESData(rho_meg, row, 4)
                    
                    cp_meg = prop('specific_heat', 'MEG', T = t, P = p, C=c)       
                    cp_meg.convert('kJ/kg-K')
                    compareVariableAndEESData(cp_meg, row, 0)

                    mu_meg = prop('dynamic_viscosity', 'MEG', T = t, P = p, C=c)       
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

    def testDependscies(self):
        T = variable([20,25,30], 'C', [0.1, 0.2, 0.15])
        P = variable([100000, 110000, 90000], 'Pa', [2500, 3000, 4000])

        rho = prop('density', 'water', T = T, P = P)
        
        flow = variable(0.0016, 'm3/s', 0.0001)
        
        massFlow = flow * rho
        
        dMassFlow_dFlow = [998.206543497641114, 997.05155024142658, 995.644405820880468]
        dMassFlow_dP = [7.32946279081818364e-10, 7.21788150328467529e-10, 7.13326743322344253e-10]
        dMassFlow_dT = [-0.000330293371618005199, -0.000410463892510075697, -0.000483157770611803349]
        
        for i, elem in enumerate(massFlow):
            self.assertAlmostEqual(elem.dependsOn[flow][1], dMassFlow_dFlow[i], 6)
            self.assertAlmostEqual(elem.dependsOn[T[i]][1], dMassFlow_dT[i], 6)
            self.assertAlmostEqual(elem.dependsOn[P[i]][1], dMassFlow_dP[i], 6)
        
    def testAir(self):
        
        T = variable(20, 'C', 2)
        P = variable(1, 'bar', 0.1)
        rh = variable(60, '%', 5)
        
        rho = prop('density', 'air', t = T, p = P, rh = rh)
        self.assertRelativeDifference(rho.value, 1.182554388576558, 1e-10)
        self.assertEqual(str(rho.unit), 'kg/m3')

        cp = prop('specific_heat', 'air', t = T, p = P, rh = rh)
        cp.convert('J/kg-K')                
        self.assertRelativeDifference(cp.value, 1013.8442346246147, 1e-10)

        humidity = prop('humidity', 'air', t = T, p = P, rh = rh)
        self.assertRelativeDifference(humidity.value, 0.008890559976462207, 1e-10)
        cp = prop('specific_heat', 'air', t = variable(35, 'C'), p = P, humidity = humidity)
        self.assertRelativeDifference(cp.value, 1014.3730005996277, 1e-10)

        


    def testExamples(self):
        mu = prop('dynamic_viscosity', 'MPG', C = variable(60,'%'), p = variable(100e3, 'Pa'), t = variable(-20, 'C'))
        self.assertAlmostEqual(mu.value, 0.13907391053938878)
        self.assertEqual(mu.unit, 'Pa-s')
        
        rho = prop('density', ['water', 'Ethanol'], C = [variable(60, '%'), variable(40, '%')], p = variable(200e3, 'Pa'), T = variable(4, 'C'))
        self.assertAlmostEqual(rho.value, 883.3922771627963)
        self.assertEqual(rho.unit, 'kg/m3')
        
        wbt = prop('wet_bulb_temperature', 'air', altitude = variable(300, 'm'), t = variable(30, 'C'), rh = variable(50, '%'))
        self.assertAlmostEqual(wbt.value, 21.917569033181564)
        self.assertEqual(wbt.unit, 'C')
    
    
    def testInputBrines(self):    
        mu = prop('dynamic_viscosity', 'MPG', C = variable([60, 60],'%'), p = variable(100e3, 'Pa'), t = variable(-20, 'C'))
        self.assertAlmostEqual(mu.value[0], 0.13907391053938878)
        self.assertAlmostEqual(mu.value[1], 0.13907391053938878)
        self.assertEqual(mu.unit, 'Pa-s')
        
        
        mu = prop('dynamic_viscosity', 'MPG', C = variable(60,'%'), p = variable([100e3, 100e3], 'Pa'), t = variable(-20, 'C'))
        self.assertAlmostEqual(mu.value[0], 0.13907391053938878)
        self.assertAlmostEqual(mu.value[1], 0.13907391053938878)
        self.assertEqual(mu.unit, 'Pa-s')
    
            
        mu = prop('dynamic_viscosity', 'MPG', C = variable(60,'%'), p = variable(100e3, 'Pa'), t = variable([-20,-20], 'C'))
        self.assertAlmostEqual(mu.value[0], 0.13907391053938878)
        self.assertAlmostEqual(mu.value[1], 0.13907391053938878)
        self.assertEqual(mu.unit, 'Pa-s')
        
                    
        mu = prop('dynamic_viscosity', 'MPG', C = variable([60,60],'%'), p = variable([100e3, 100e3], 'Pa'), t = variable(-20, 'C'))
        self.assertAlmostEqual(mu.value[0], 0.13907391053938878)
        self.assertAlmostEqual(mu.value[1], 0.13907391053938878)
        self.assertEqual(mu.unit, 'Pa-s')
        
        
        mu = prop('dynamic_viscosity', 'MPG', C = variable([60,60],'%'), p = variable(100e3, 'Pa'), t = variable([-20,-20], 'C'))
        self.assertAlmostEqual(mu.value[0], 0.13907391053938878)
        self.assertAlmostEqual(mu.value[1], 0.13907391053938878)
        self.assertEqual(mu.unit, 'Pa-s')
        
         
        mu = prop('dynamic_viscosity', 'MPG', C = variable(60,'%'), p = variable([100e3,100e3], 'Pa'), t = variable([-20,-20], 'C'))
        self.assertAlmostEqual(mu.value[0], 0.13907391053938878)
        self.assertAlmostEqual(mu.value[1], 0.13907391053938878)
        self.assertEqual(mu.unit, 'Pa-s')
    
       
        mu = prop('dynamic_viscosity', 'MPG', C = variable([60,60],'%'), p = variable([100e3,100e3], 'Pa'), t = variable([-20,-20], 'C'))
        self.assertAlmostEqual(mu.value[0], 0.13907391053938878)
        self.assertAlmostEqual(mu.value[1], 0.13907391053938878)
        self.assertEqual(mu.unit, 'Pa-s')
    
    def testInputsMixtures(self):
        
        rho = prop('density', ['water', 'Ethanol'], C = [variable([60, 60], '%'), variable([40,40], '%')], p = variable(200e3, 'Pa'), T = variable(4, 'C'))
        self.assertAlmostEqual(rho.value[0], 883.3922771627963)
        self.assertAlmostEqual(rho.value[1], 883.3922771627963)
        self.assertEqual(rho.unit, 'kg/m3')
        
        rho = prop('density', ['water', 'Ethanol'], C = [variable(60, '%'), variable(40, '%')], p = variable([200e3, 200e3], 'Pa'), T = variable(4, 'C'))
        self.assertAlmostEqual(rho.value[0], 883.3922771627963)
        self.assertAlmostEqual(rho.value[1], 883.3922771627963)
        self.assertEqual(rho.unit, 'kg/m3')
        
        rho = prop('density', ['water', 'Ethanol'], C = [variable(60, '%'), variable(40, '%')], p = variable(200e3, 'Pa'), T = variable([4,4], 'C'))
        self.assertAlmostEqual(rho.value[0], 883.3922771627963)
        self.assertAlmostEqual(rho.value[1], 883.3922771627963)
        self.assertEqual(rho.unit, 'kg/m3')
     
        rho = prop('density', ['water', 'Ethanol'], C = [variable([60,60], '%'), variable([40,40], '%')], p = variable([200e3,200e3], 'Pa'), T = variable(4, 'C'))
        self.assertAlmostEqual(rho.value[0], 883.3922771627963)
        self.assertAlmostEqual(rho.value[1], 883.3922771627963)
        self.assertEqual(rho.unit, 'kg/m3')
           
        rho = prop('density', ['water', 'Ethanol'], C = [variable([60,60], '%'), variable([40,40], '%')], p = variable(200e3, 'Pa'), T = variable([4,4], 'C'))
        self.assertAlmostEqual(rho.value[0], 883.3922771627963)
        self.assertAlmostEqual(rho.value[1], 883.3922771627963)
        self.assertEqual(rho.unit, 'kg/m3')
        
        rho = prop('density', ['water', 'Ethanol'], C = [variable(60, '%'), variable(40, '%')], p = variable([200e3,200e3], 'Pa'), T = variable([4,4], 'C'))
        self.assertAlmostEqual(rho.value[0], 883.3922771627963)
        self.assertAlmostEqual(rho.value[1], 883.3922771627963)
        self.assertEqual(rho.unit, 'kg/m3')
        
        rho = prop('density', ['water', 'Ethanol'], C = [variable([60,60], '%'), variable([40,40], '%')], p = variable([200e3,200e3], 'Pa'), T = variable([4,4], 'C'))
        self.assertAlmostEqual(rho.value[0], 883.3922771627963)
        self.assertAlmostEqual(rho.value[1], 883.3922771627963)
        self.assertEqual(rho.unit, 'kg/m3')
        
        
if __name__ == '__main__':
    unittest.main()
