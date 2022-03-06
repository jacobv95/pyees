import autograd.numpy as np
from autograd import jacobian
import networkx as nx
import time
from networkx.algorithms import bipartite
from scipy.sparse import csr_matrix
from scipy.optimize import least_squares
try:
    from pyees.unitSystem import unit as unitConversion
except ModuleNotFoundError:
    from unitSystem import unit as unitConversion


class variable():
    def __init__(self, value, unit, uncert=0, upperBound=np.inf, lowerBound=-np.inf, nDigits=3, fix=False, pos=None) -> None:
        # unit
        self.unitOriginal = unit
        self.valueOriginal = value
        self.uncertOriginal = uncert
        self.unitConversion = unitConversion()

        # value
        valueSI, unitSI = self.unitConversion.convertToSI(self.valueOriginal, self.unitOriginal)
        uncertSI, _ = self.unitConversion.convertToSI(self.uncertOriginal, self.unitOriginal)
        self.unit = unitSI
        self.value = valueSI
        self.uncert = uncertSI

        # bounds
        if not np.isinf(lowerBound):
            lowerBound, _ = self.unitConversion.convertToSI(lowerBound, self.unitOriginal)
        if not np.isinf(upperBound):
            upperBound, _ = self.unitConversion.convertToSI(upperBound, self.unitOriginal)
        self.bounds = [lowerBound, upperBound]

        # other
        self.nDigits = nDigits
        self.isDefined = fix
        self.pos = pos

        # uncertanty
        self.dependsOn = {}

    def fix(self):
        self.isDefined = True

    def printValue(self):
        def decimalString(x):
            if x != 0:
                n_zeros = int(np.ceil(-np.log10(np.abs(x))))
                n_decimals = np.max([n_zeros + self.nDigits - 1, 0])
                return f'{x:.{n_decimals}f}'
            else:
                return f'{x:.{self.nDigits}f}'

        if self.unit == '1':
            unit = '[ ]'
        else:
            unit = self.unit
        val = decimalString(self.value)
        if self.uncert != 0:
            uncert = decimalString(self.uncert)
            return f'{val} +/- {uncert} {unit}'
        else:
            return f'{val} {unit}'

    def convertToOriginalUnit(self):
        self.value, self.unit = self.unitConversion.convertFromSI(self.value, self.unitOriginal)
        self.uncert, _ = self.unitConversion.convertFromSI(self.uncert, self.unitOriginal)

    def convertToSIUnit(self):
        self.value, self.unit = self.unitConversion.convertToSI(self.value, self.unit)
        self.uncert, _ = self.unitConversion.convertFromSI(self.uncert, self.unit)

    def __repr__(self, original=False) -> str:
        if not original:
            value = self.value
            unit = self.unit
        else:
            value = self.valueOriginal
            unit = self.unitOriginal
        return f'variable({value}, {unit})'

    def _addDependents(self, L, grad):
        for i, elem in enumerate(L):
            if elem.dependsOn:
                for key, item in elem.dependsOn.items():
                    if key in self.dependsOn:
                        self.dependsOn[key] += item * grad[i]
                    else:
                        self.dependsOn[key] = item * grad[i]
            else:
                if elem in self.dependsOn:
                    self.dependsOn[elem] += grad[i]
                else:
                    self.dependsOn[elem] = grad[i]

    def _calculateUncertanty(self):
        self.uncert = np.sqrt(sum([(gi * var.uncertOriginal)**2 for gi, var in zip(self.dependsOn.values(), self.dependsOn.keys())]))
        self.uncert, _ = self.unitConversion.convertToSI(self.uncert, self.unitOriginal)

    def __add__(self, other):
        if isinstance(other, variable):
            if self.unit != other.unit:
                raise ValueError(f'You tried to add {self.__repr__(original=True)} to  {other.__repr__(original=True)}, but the units does not match')

            val = self.value + other.value
            unit = self.unit
            grad = [1, 1]
            vars = [self, other]

            if self.unitOriginal == other.unitOriginal:
                val, unit = self.unitConversion.convertFromSI(val, self.unitOriginal)

            var = variable(val, unit)
            var._addDependents(vars, grad)
            var._calculateUncertanty()
            return var
        else:
            other = variable(other, self.unit)
            return self + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, variable):
            if self.unit != other.unit:
                raise ValueError(f'You tried to subtract {other.__repr__(original=True)} from {self.__repr__(original=True)}, but the units does not match')

            val = self.value - other.value
            unit = self.unit
            grad = [1, -1]
            vars = [self, other]

            if self.unitOriginal == other.unitOriginal:
                val, unit = self.unitConversion.convertFromSI(val, self.unitOriginal)

            var = variable(val, unit)
            var._addDependents(vars, grad)
            var._calculateUncertanty()

            return var
        else:
            other = variable(other, self.unit)
            return self - other

    def __rsub__(self, other):
        return - self + other

    def __mul__(self, other):
        if isinstance(other, variable):
            valSelf = self.value
            valOther = other.value
            valSelf, unitSelf = self.unitConversion.convertFromSI(valSelf, self.unitOriginal)
            valOther, unitOther = other.unitConversion.convertFromSI(valOther, other.unitOriginal)
            unit = unitConversion().multiply(unitSelf, unitOther)

            val = valSelf * valOther
            grad = [valOther, valSelf]
            vars = [self, other]

            var = variable(val, unit)
            var._addDependents(vars, grad)
            var._calculateUncertanty()

            return var
        else:
            other = variable(other, '1')
            return self * other

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        if isinstance(other, variable):
            valSelf = self.value
            valOther = other.value
            valSelf, unitSelf = self.unitConversion.convertFromSI(valSelf, self.unitOriginal)
            valOther, unitOther = other.unitConversion.convertFromSI(valOther, other.unitOriginal)
            if unitOther != '1':
                raise ValueError('A measurement with a unit cannot be raise to the power of another measurement with a unit')
            unit = unitConversion().power(unitSelf, valOther)

            val = valSelf ** valOther
            grad = [valOther * valSelf ** (valOther - 1), valSelf ** valOther * np.log(valSelf)]
            vars = [self, other]

            var = variable(val, unit)
            var._addDependents(vars, grad)
            var._calculateUncertanty()

            return var
        else:
            other = variable(other, '1')
            return self ** other

    def __rpow__(self, other):
        if isinstance(other, variable):
            valSelf = self.value
            valOther = other.value
            valSelf, unitSelf = self.unitConversion.convertFromSI(valSelf, self.unitOriginal)
            uncertSelf, _ = self.unitConversion.convertFromSI(self.uncert, self.unitOriginal)
            valOther, unitOther = other.unitConversion.convertFromSI(valOther, other.unitOriginal)
            uncertOther, _ = other.unitConversion.convertFromSI(other.uncert, other.unitOriginal)
            if unitSelf != '1':
                raise ValueError('A measurement with a unit cannot be raise to the power of another measurement with a unit')
            unit = unitConversion().power(unitOther, valSelf)

            val = valOther ** valSelf
            grad = [valSelf * valOther ** (valSelf - 1), valOther ** valSelf * np.log(valOther)]
            vars = [self, other]

            var = variable(val, unit)
            var._addDependents(vars, grad)
            var._calculateUncertanty()

            return var
        else:
            other = variable(other, '1')
            return other ** self

    def __truediv__(self, other):
        if isinstance(other, variable):
            valSelf = self.value
            valOther = other.value
            valSelf, unitSelf = self.unitConversion.convertFromSI(valSelf, self.unitOriginal)
            valOther, unitOther = other.unitConversion.convertFromSI(valOther, other.unitOriginal)
            unit = unitConversion().divide(unitSelf, unitOther)

            val = valSelf / valOther
            grad = [1 / valOther, valSelf / (valOther**2)]
            vars = [self, other]

            var = variable(val, unit)
            var._addDependents(vars, grad)
            var._calculateUncertanty()

            return var
        else:
            other = variable(other, '1')
            return self / other

    def __rtruediv__(self, other):
        if isinstance(other, variable):
            valSelf = self.value
            valOther = other.value
            valSelf, unitSelf = self.unitConversion.convertFromSI(valSelf, self.unitOriginal)
            valOther, unitOther = other.unitConversion.convertFromSI(valOther, other.unitOriginal)
            unit = unitConversion().divide(unitOther, unitSelf)

            val = valOther / valSelf
            grad = [valOther / (valSelf**2), 1 / (valSelf)]
            vars = [self, other]

            var = variable(val, unit)
            var._addDependents(vars, grad)
            var._calculateUncertanty()

            return var
        else:
            other = variable(other, '1')
            return other / self

    def __neg__(self):
        return -1 * self


times = {}


def timeit(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        if func.__name__ in times:
            times[func.__name__][0] += 1
            times[func.__name__][1] += t2 - t1
        else:
            times[func.__name__] = [1, t2 - t1]
        return res
    return wrapper


class System():
    @timeit
    def __init__(self) -> None:
        self.variables = []
        self.subSystems = []
        self._blockEquations = None
        self._blockVariables = None
        self._blockNr = None
        self.invphi = (np.sqrt(5) - 1) / 2
        self.invphi2 = (3 - np.sqrt(5)) / 2

    @timeit
    def addEquations(self, f):
        setattr(self, 'eq', f)

    @timeit
    def _getEquations(self):
        self._listOfEquations = []
        self._listofEquationsSystems = []
        for sys in self.subSystems:
            if hasattr(sys, 'eq'):
                equ = sys.eq(sys)
                if not (isinstance(equ, list) or isinstance(equ, set)):
                    index = self.subSystems.index(sys)
                    name = self.subSystemNames[index]
                    raise ValueError(f'The equations of {name} is not a list-like-object')
                self._listOfEquations.append(sys.eq)
                self._listofEquationsSystems.append(sys)

    @timeit
    def _getSubsystem(self, sys, sysName):
        L = [sys]
        Lnames = [sysName]
        for name, item in sys.__dict__.items():
            if isinstance(item, System):
                if item not in L:
                    L.append(item)
                    Lnames.append(sysName + '.' + name)
                l, lnames = self._getSubsystem(item, name)
                for ll, llname in zip(l, lnames):
                    if ll not in L:
                        L.append(ll)
                        Lnames.append(sysName + '.' + llname)
        sys.name = sysName
        return L, Lnames

    @timeit
    def _getVariables(self):
        self.variableNames = []
        self.variables = []
        # find all variables and subsystems in self
        for sys, sysName in zip(self.subSystems, self.subSystemNames):
            sys.variables = []
            sys.variableNames = []
            for name, item in sys.__dict__.items():
                if isinstance(item, variable):
                    if item not in self.variables:
                        self.variables.append(item)
                        self.variableNames.append(sysName + '.' + name)

        self._notDefinedVariables = [var for var in self.variables if not var.isDefined]

    @timeit
    def _nEquationsEqualsVariables(self):

        # determine number of free variables
        nVar = len(self._notDefinedVariables)
        if nVar == 0:
            raise ValueError('There are no variables')

        # evaluate the equations. This is just to get the number of equations, therefore this is done at zero
        nEq = len(self._equations(blocking=False))
        if nEq == 0:
            raise ValueError('You have not specified any equtions')

        # compare the nubmer of variables to the number of equations
        if nVar != nEq:
            self._variableUsage()
            raise ValueError(f'The number of equations ({nEq}) is not equal to the number of variables ({nVar}). The number of usage for each variable is stated above')

    @timeit
    def _tarjanBlocking(self):
        # create the jacobian
        n = len(self._notDefinedVariables)
        J = np.zeros([n, n])
        originalEq = self._equations(blocking=False)
        for i, v in enumerate(self._notDefinedVariables):
            originalVal = v.value
            if v.value <= v.bounds[1]:
                v.value += 100
                if v.value >= v.bounds[1]:
                    v.value = v.bounds[1]
            else:
                v.value -= 100
                if v.value <= v.bounds[0]:
                    v.value = v.bounds[0]
            eq = self._equations(blocking=False)
            J[:, i] = [1 if eq[j] != originalEq[j] else 0 for j in range(n)]
            v.value = originalVal

        # create a bipartite graph of the connections between the equations and the variables
        g = bipartite.matrix.from_biadjacency_matrix(csr_matrix(J))

        # determine if the equations set is connected
        if nx.is_connected(g):
            self._blockEquations, self._blockVariables = self._blocksFromBipartite(g, J)
        else:
            S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
            self._blockEquations = []
            self._blockVariables = []
            for s in S:
                equations = [i for i, (node, dat) in enumerate(g.nodes(data=True)) if node in s.nodes and dat['bipartite'] == 0]
                jj = J[equations, :]
                n, m = jj.shape
                variables = []
                for i in range(m):
                    if not all(jj[:, i] == 0):
                        variables.append(i)
                jj = jj[:, variables]
                eqs, vars = self._blocksFromBipartite(s, jj)
                for i, elem in enumerate(eqs):
                    eqs[i] = [equations[e] for e in elem]
                for i, elem in enumerate(vars):
                    vars[i] = [variables[v] for v in elem]
                self._blockEquations += eqs
                self._blockVariables += vars

    @timeit
    def _blocksFromBipartite(self, g, J):
        # create a mathcing of the bipartite graph
        try:
            mathcing = bipartite.hopcroft_karp_matching(g)
        except nx.exception.AmbiguousSolution:
            self._variableUsage()
            raise ValueError('The blocking of the equation set could not be made. This might be due to an error in the equation set. The number of uses of each variable is printed above.') from None

        n, m = J.shape
        if n != m:
            raise ValueError('One of the independent parts of the system of equations does not have an equal number of variables and equations')

        L = [item for _, item in mathcing.items() if g.nodes(data=True)[item]['bipartite'] == 0]
        L = np.argsort(L)

        # use the matching to reorder A such that the main diagonal is 1
        J = J[L]

        # create a graph of A
        g = nx.DiGraph()
        for i in range(n):
            g.add_node(i)
        for i in range(n):
            for j in range(n):
                if i != j and J[i, j] != 0:
                    g.add_edge(i, j, label=j)

        # find the strongly connected components of g
        # these corresponds to the schedule of the equations to solve
        C = [list(elem) for elem in nx.strongly_connected_components(g)]

        # map the strong components to equations through the mathcing of the bipartite graph
        blockEquations = [[L[elem] for elem in sublist] for sublist in C]

        # determine the variables used in each step of the schedule of the equations
        blockVariables = [None] * len(blockEquations)
        for i, c in enumerate(C):
            variables = np.unique(sorted([j for row in c for j, elem in enumerate(J[row, :]) if elem != 0])).tolist()
            J[:, variables] = 0
            blockVariables[i] = variables

        return blockEquations, blockVariables

    @timeit
    def _convertVariablesToSI(self):
        for var in self.variables:
            var.convertToSIUnit()

    @timeit
    def _convertVariablesFromSI(self):
        for var in self.variables:
            var.convertToOriginalUnit()

    @timeit
    def solve(self, method=1):

        # find all variables in the system and subsystem
        self.subSystems, self.subSystemNames = self._getSubsystem(self, 'System')

        # get all variables and their names
        self._getVariables()

        # convert all variables to SI
        # this is only necessary if the system is solved more than once
        self._convertVariablesToSI()

        # get all the equations
        self._getEquations()

        # Determine if the number of equations matches the number of variables
        self._nEquationsEqualsVariables()

        # block the equation set
        if method:
            if self._blockNr is None:
                self._tarjanBlocking()
        else:
            self._blockVariables = [[i for i, _ in enumerate(self._notDefinedVariables)]]
            self._blockEquations = [[i for i, _ in enumerate(self._notDefinedVariables)]]
            self._blockNr = 0

        # x0 = np.array([var.value for var in self._notDefinedVariables], dtype=float)
        # self.SGD(x0, self.cost)
        # bounds = [var.bounds for var in self._notDefinedVariables]
        # x0 = [var.value for var in self._notDefinedVariables]
        # sol = minimize(
        #     self.cost,
        #     x0,
        #     bounds=bounds,
        #     jac=grad(self.cost),
        #     # hess=hessian(self.cost),
        #     method='L-BFGS-B'
        # )
        # self._distributeVariables(sol.x)

        # solve
        for self._blockNr in range(len(self._blockEquations)):
            self._solveBlock()

        # convert all the varibles back in to their orogonal unit
        self._convertVariablesFromSI()

    @ timeit
    def _solveBlock(self):

        variableIndexes = self._blockVariables[self._blockNr]
        vars = [v for i, v in enumerate(self._notDefinedVariables) if i in variableIndexes]

        # solve the system
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        sol = least_squares(
            fun=self._evaluateEquations,
            x0=np.array([var.value for var in vars], dtype=float),
            method='dogbox',
            bounds=np.array([var.bounds for var in vars]).transpose().tolist(),
            gtol=None,
            ftol=None,
            max_nfev=1000,
            verbose=0,
            jac=jacobian(self._evaluateEquations),
        )

        # raise warnings based on the solution
        self._raiseWarnings(sol)

        # # distribut the solution
        self._distributeVariables(sol.x)

    @ timeit
    def _raiseWarnings(self, sol):
        # print the solution if it was not successfull
        if not sol.success:
            print(sol)

        # Give a warning if a bound is active
        if sum(sol.active_mask) != 0:
            for elem, var in zip(sol.active_mask, self.variables):
                if elem == 1 or elem == -1:
                    if elem == 1:
                        bound = 'upper'
                        index = 1
                    else:
                        bound = 'lower'
                        index = 0
                    varIndex = self.variables.index(var)
                    varName = self.variableNames[varIndex]
                    boundValue = var.bounds[index]
                    boundValue, _ = unitConversion().convertFromSI(boundValue, var.unit)
                    print(f'WARNING: The {bound} bound ({boundValue} {var.unit}) of the variable {varName} is active. This might mean that the solution has not converged')

    @ timeit
    def _distributeVariables(self, x):
        ite = 0
        for i, v in enumerate(self._notDefinedVariables):
            if i in self._blockVariables[self._blockNr]:
                v.value = x[ite]
                ite += 1

    @ timeit
    def _evaluateEquations(self, x, blocking=True):
        self._distributeVariables(x)
        # return self._equations(blocking)
        return np.array(self._equations(blocking), dtype=float)

    @ timeit
    def _equations(self, blocking=True):
        # evaluate the equations
        if blocking:
            ite = 0
            iteEq = 0
            eqs = [None] * len(self._blockEquations[self._blockNr])
            for eq, sys in zip(self._listOfEquations, self._listofEquationsSystems):
                equ = eq(sys)
                for elem in equ:
                    for i in range(1, len(elem)):
                        if iteEq in self._blockEquations[self._blockNr]:
                            eqs[ite] = (elem[0] - elem[i]).value
                            ite += 1
                        iteEq += 1
        else:
            ite = 0
            eqs = [None] * len(self._notDefinedVariables)
            for eq, sys in zip(self._listOfEquations, self._listofEquationsSystems):
                equ = eq(sys)
                for elem in equ:
                    for i in range(1, len(elem)):
                        eqs[ite] = (elem[0] - elem[i]).value
                        ite += 1

        return eqs

    @ timeit
    def _variableUsage(self):
        # evaluate the equations
        n = len(self.variables)
        m = len(self._notDefinedVariables)
        uses = np.zeros(n)
        originalEq = self._equations(blocking=False)
        for i, var in enumerate(self.variables):
            origiVal = var.value
            if var.value <= var.bounds[1]:
                var.value += 100
                if var.value >= var.bounds[1]:
                    var.value = var.bounds[1]
            else:
                var.value -= 100
                if var.value <= var.bounds[0]:
                    var.value = var.bounds[0]
            curEq = self._equations(blocking=False)
            nEq = sum([1 if curEq[j] != originalEq[j] else 0 for j in range(m)])
            uses[i] = nEq
            var.value = origiVal

        indexes = np.argsort(uses)
        uses = [uses[i] for i in indexes]
        var = [self.variables[i] for i in indexes]
        differentUsages = list(set(uses))
        print('\n')
        for i in range(len(differentUsages)):
            currentVars = [v for v, u in zip(var, uses) if u == differentUsages[i]]
            print(f'The following variables has been used in {int(differentUsages[i])} times')
            for v in currentVars:
                index = self.variables.index(v)
                name = self.variableNames[index]
                print(name)
            print('\n')

    @ timeit
    def printVariables(self):
        self.subSystems, self.subSystemNames = self._getSubsystem(self, self.name)
        self._getVariables()
        maxLength = np.max([len(elem) for elem in self.variableNames])
        for sys in self.subSystems:

            nPrints = 0
            for _, item in sys.__dict__.items():
                if item in self.variables:
                    nPrints += 1
                    index = self.variables.index(item)
                    name = self.variableNames[index]
                    nameLength = len(name)
                    print(name, ' ' * (maxLength - nameLength + 3), item.printValue())
            if nPrints != 0:
                print('')

    @ timeit
    def writeVariablesOnDiagraom(self, existingPDF, font='Helvetica', fontSize=8):
        from PyPDF2 import PdfFileReader, PdfFileWriter
        import io
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        # The page to print all variables on
        pageNr = 1

        # function to get coordinates in the same system as any pdf viewer
        def coord(x, y, height):
            x, y = x * mm, height - y * mm
            return x, y

        # find the name of the existing pdf
        existingPDF = existingPDF.lower()
        if not '.pdf' in existingPDF:
            raise ValueError('You have to supply a .pdf file')
        index = existingPDF.find('.pdf')
        name = existingPDF[0:index]

        # read the page of the existing pdf
        existing_pdf = PdfFileReader(open(name + '.pdf', "rb"))
        n = existing_pdf.getNumPages()

        for i in range(n):
            output = PdfFileWriter()
            page = existing_pdf.getPage(i)
            if i + 1 == pageNr:

                # create a new page with the same size as the existing pdf
                packet = io.BytesIO()
                can = canvas.Canvas(packet)
                can.setFont(font, fontSize)
                # rect = generic.RectangleObject([int(float(elem) * 0.352) for elem in page.mediaBox])
                rect = page.mediaBox
                can.setPageSize(rect)
                height = float(rect[3])

                # write on the new page
                for var in self.variables:
                    if not var.pos is None:
                        can.drawString(*coord(var.pos[0], var.pos[1], height), text=var.printValue())
                can.save()

                # merge the new and the existing pages
                new_pdf = PdfFileReader(packet)
                page.mergePage(new_pdf.getPage(0))

                # add the modified page to the existing pdf
            output.addPage(page)

        # write the file
        with open(name + '_with_variables.pdf', 'wb') as f:
            output.write(f)


def main():
    A = variable(5, 'L/min', uncert=0.1)
    B = variable(2, 'L/min', uncert=5)
    C = variable(3, 'm', uncert=0.5)
    D = A**3
    D.convertToOriginalUnit()
    D.nDigits = 10
    print(D.printValue())


if __name__ == "__main__":
    main()
