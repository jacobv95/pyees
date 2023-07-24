import cProfile
import pstats
import io
try:
    from variable import variable
except ImportError:
    from pyees.variable import variable
    
pr = cProfile.Profile()
pr.enable()

for _ in range(50_000):    
    c = variable(4.182, 'J/kg-K', 0.130)
    rho = variable(1000, 'kg/m3', 0.01)
    t_in = variable(50, 'C', 1.2)
    t_out = variable([10, 15, 20, 25, 30, 35, 40], 'C', [0.9, 1.1, 1.0, 0.8, 0.9, 1.3, 1.1])
    v_dot = variable(300, 'L/min', 3)
    air_speed = variable([6.5, 6, 5.5, 5, 4.5, 4, 3.5], 'm/s', [0.1, 0.15, 0.12, 0.13, 0.9, 1.1, 1.0])
    q = c * rho * v_dot * (t_in - t_out)
    q.convert('kW')
    
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('profile.txt', 'w+') as f:
    f.write(s.getvalue())
