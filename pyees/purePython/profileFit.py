import cProfile
import pstats
import io
pr = cProfile.Profile()
try:
    from variable import variable
    from sheet import sheetsFromFile
    from fit import exp_fit, pol_fit
except ImportError:
    from pyees.variable import variable
    from pyees.sheet import sheetsFromFile
    from pyees import exp_fit, pol_fit
    
data = sheetsFromFile("testData/profileFitData.xls", "A-G")
data.Time /= variable(1, 's')

pr.enable()
f = pol_fit(data.Time, data.t)
pr.disable()
print(f)

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()
with open('profileFit.txt', 'w+') as f:
    f.write(s.getvalue())
