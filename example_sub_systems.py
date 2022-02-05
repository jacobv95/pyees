from pyees import *


# Valve
valve = System()
valve.flow = variable(15, 'L/min')
valve.dP = variable(1, 'bar')


def f_valve(self):
    eq = []
    eq.append((self.dP, variable(8.65, 'min-bar/L') * self.flow**2))
    return eq


valve.addEquations(f_valve)


# Pump
pump = System()
pump.flow = variable(20, 'L/min')
pump.dP = variable(200, 'bar')


def f_pump(self):
    eq = []
    eq.append((self.dP, 12345 - variable(0.5, 'min-bar/L') * self.flow**2))
    return eq


pump.addEquations(f_pump)


# System
sys = System()
sys.pump = pump
sys.valve = valve


def f_sys(self):
    eq = []
    eq.append((self.pump.dP, self.valve.dP))
    eq.append((self.pump.flow, self.valve.flow))
    return eq


sys.addEquations(f_sys)

sys.solve()
sys.printVariables()
