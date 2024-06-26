import unittest

from testFit import test as testFit
from testSheet import test as testSheet
from testUnit import test as testUnit
from testVariable import test as testVariable
from testProp import test as testProp
from testSolve import test as testSolve

    

def main():
    tests = [
        testFit,
        testSheet,
        testUnit, 
        testVariable,
        testProp,
        testSolve
    ]

    suites = []
    for test in tests:
        suites.append(unittest.TestLoader().loadTestsFromTestCase(test))

    suite = unittest.TestSuite(suites)
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    main()
