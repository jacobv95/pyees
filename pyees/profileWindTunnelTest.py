import cProfile
import pstats
import io


pr = cProfile.Profile()
pr.enable()
import testData.windTunnelTest.dataProcessing
pr.disable()


s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats(100)

with open('profileWindTunnelTest.txt', 'w+') as f:
    f.write(s.getvalue())
