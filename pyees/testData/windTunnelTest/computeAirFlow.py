import numpy as np
from variable import variable
from prop import prop
from solve import solve



def computeAirFlow(n_nozzle, d_nozzle, dp_nozzle, p_atm, T_5, T_6, P_s5, P_s6, rh_5, rh_6, C_initial = 1.00):
    """_summary_

    Args:
        n_nozzle (int): The number of nozzles used
        d_nozzle (pyees variable): The small diameter of the nozzles
        T_5 (pyees variable): The temperature infront of the nozzles
        T_6 (pyees variable): The temperature after the nozzles
        P_s5 (pyees variable): The static pressure infront of the nozzles
        P_s6 (pyees variable): The static pressure after the nozzles
        rh_5 (pyees variable): The relative humidity infront of the nozzles
        rh_6 (pyees variable): The relative humidty after the nozzles
        dp_nozzle (pyees variable): The differential pressure across the nozzles
        C_initial (float, optional): _description_. Defaults to 1.00.

    Returns:
        Q (pyees variable): The air flow through the nozzles 
    """
    
    ## determine all necessary constants
    alpha = P_s6 / P_s5                         ## Eq. 7.11 AMCA 210

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
    
    if (len(Y) == 1):
        rho_5 = variable([rho_5])
        mu_6 = variable([mu_6])
    C, Re = solve(
        SolveDischargeCoefficientAndReynoldsNumber,
        [variable(C_initial), variable(2e5)], 
        tol = 1e-100,
        parametric = [Y, rho_5, mu_6, dp_nozzle],
        bounds = [[variable(1e-12), variable(np.inf)], [variable(1e-12), variable(np.inf)]]
    )    


    ## determine the air flow
    A_nozzle = np.pi/4 * d_nozzle ** 2
    Q = Y * np.sqrt(2 * dp_nozzle / rho_5) * n_nozzle * C * A_nozzle
    Q.convert('m3/s')

    return Q

if __name__ == "__main__":
        

    ## inputs
    n_nozzle = 20
    d_nozzle = variable(225, 'mm')
    rh_air = variable([50]*3, '%', [5]*3)
    T_air_out = variable([20]*3, 'C', [2]*3)
    p_atm = variable([1]*3, 'bar', [0.01]*3)
    dp_cooler = variable([90]*3, 'Pa', [5]*3)
    dp_nozzle = variable([30, 35, 40], 'Pa', [5]*3)

    T_5 = T_air_out
    T_6 = T_air_out

    P_s5 = p_atm - dp_cooler
    P_s6 = p_atm - dp_cooler - dp_nozzle

    rh_5 = rh_air
    rh_6 = rh_air

    Q = computeAirFlow(n_nozzle, d_nozzle, dp_nozzle, p_atm, T_5, T_6, P_s5, P_s6, rh_5, rh_6, C_initial = 1)
    print(Q)