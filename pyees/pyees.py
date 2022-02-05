import numpy as np
from scipy.optimize import least_squares
import CoolProp.CoolProp as CP
from PyPDF2 import PdfFileReader, PdfFileWriter
import io
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
try:
    from pyees.unitSystem import unit as unitConversion
except ModuleNotFoundError:
    from unitSystem import unit as unitConversion


def prop(property, paramenterA, valueA, paramenterB, valueB, fluid):
    units = {
        'D': 'kg/m3',
        'V': 'kg/m-s',
        'C': 'J/kg-K',
        'H': 'J/kg'
    }
    if property not in units:
        raise ValueError(f'The property {property} is unkown. The known properties are {[key for key in units]}')
    unit = units[property]
    val = CP.PropsSI(property, paramenterA, valueA, paramenterB, valueB, fluid)
    var = variable(val, unit=unit)
    var.unit = unit
    return var


class variable():
    def __init__(self, value, unit, upperBound=np.inf, lowerBound=-np.inf, nDigits=3, fix=False, pos=None) -> None:
        # unit
        self.unitOriginal = unit
        self.valueOriginal = value
        self.unitConversion = unitConversion()

        # value
        valueSI, unitSI = self.unitConversion.convertToSI(value, self.unitOriginal)
        self.unit = unitSI
        self.value = valueSI

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

    def fix(self):
        self.isDefined = True

    def printValue(self):
        return f'{self.value:.{self.nDigits}f} {self.unit}'

    def updateValue(self, value):
        if isinstance(value, variable):
            if not value.unit == self.unit:
                raise ValueError(f'The units of the updating value {value} does not match the variable {self}')
            self.value = value.value
        else:
            self.value = value

    def convertToOriginalUnit(self):
        self.value, self.unit = self.unitConversion.convertFromSI(self.value, self.unitOriginal)

    def __repr__(self, original=False) -> str:
        if not original:
            value = self.value
            unit = self.unit
        else:
            value = self.valueOriginal
            unit = self.unitOriginal
        return f'variable({value}, {unit})'

    def __add__(self, other):
        if isinstance(other, variable):
            if self.unit != other.unit:
                raise ValueError(f'You tried to add {self.__repr__(original=True)} to  {other.__repr__(original=True)}, but the units does not match')
            val = self.value + other.value
            unit = self.unit
            if self.unitOriginal == other.unitOriginal:
                val, unit = self.unitConversion.convertFromSI(val, unit)
        else:
            val = self.value + other
            unit = self.unit
        return variable(val, unit)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, variable):
            if self.unit != other.unit:
                raise ValueError(f'You tried to subtract {other.__repr__(original=True)} from {self.__repr__(original=True)}, but the units does not match')
            val = self.value - other.value
            unit = self.unit
            if self.unitOriginal == other.unitOriginal:
                val, unit = self.unitConversion.convertFromSI(val, unit)
        else:
            val = self.value - other
            unit = self.unit
        return variable(val, unit)

    def __rsub__(self, other):
        if isinstance(other, variable):
            if self.unit != other.unit:
                raise ValueError(f'You tried to subtract {self.__repr__(original=True)} from {other.__repr__(original=True)}, but the units does not match')
            val = other.value - self.value
            unit = self.unit
            if self.unitOriginal == other.unitOriginal:
                val, unit = self.unitConversion.convertFromSI(val, unit)
        else:
            val = other - self.value
            unit = self.unit
        return variable(val, unit)

    def __mul__(self, other):
        if isinstance(other, variable):
            unit = unitConversion().multiply(self.unit, other.unit)
            val = self.value * other.value
        else:
            unit = self.unit
            val = self.value * other
        return variable(val, unit)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        if isinstance(other, variable):
            raise NotImplemented
        else:
            return variable(self.value ** other, self.unit)

    def __truediv__(self, other):
        if isinstance(other, variable):
            unit = unitConversion().divide(self.unit, other.unit)
            val = self.value / other.value
        else:
            unit = self.unit
            val = self.value / other
        return variable(val, unit)

    def __rtruediv__(self, other):
        if isinstance(other, variable):
            unit = unitConversion().divide(other.unit, self.unit)
            val = other.value / self.value
        else:
            unit = unitConversion().divide('1', self.unit)
            val = other / self.value
        return variable(val, unit)

    def __neg__(self):
        return variable(-self.valueOriginal, self.unitOriginal)


class System():

    def __init__(self) -> None:
        self.variables = []
        self.subSystems = []

    def addEquations(self, f):
        setattr(self, 'eq', f)

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

    def _raiseWarnings(self, sol):
        # print the solution if it was not successfull
        if not sol.success:
            print(sol)

        # Give a warning if a bound is active
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

    def solve(self):

        # find all variables in the system and subsystem
        self.subSystems, self.subSystemNames = self._getSubsystem(self, 'System')

        # get all variables and their names
        self._getVariables()
        self._notDefinedVariables = [var for var in self.variables if not var.isDefined]

        # Determine if the number of equations matches the number of variables
        self._nEquationsEqualsVariables()

        # determine the scale factor for all equations
        self._findScaleFactor()

        # solve the system
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        x = np.array([var.value for var in self._notDefinedVariables])
        lower = [var.bounds[0] for var in self._notDefinedVariables]
        upper = [var.bounds[1] for var in self._notDefinedVariables]
        bounds = [lower, upper]

        sol = least_squares(
            self._evaluateEquations,
            x,
            method='dogbox',
            bounds=bounds,
            gtol=None,
            ftol=None,
            max_nfev=1000,
            verbose=0
        )

        # raise warnings based on the solution
        self._raiseWarnings(sol)

        # distribut the solution
        self._distributeVariables(sol.x)

        # convert all the varibles back in to their orogonal unit
        for var in self.variables:
            var.convertToOriginalUnit()

    def _findScaleFactor(self):
        x = [var.value for var in self._notDefinedVariables]
        eq = self._evaluateEquations(x)
        self.scaleFactor = [np.abs(elem) if elem != 0 else None for elem in eq]

    def _nEquationsEqualsVariables(self):

        # determine number of free variables
        n_var = len(self._notDefinedVariables)

        if n_var == 0:
            raise ValueError('There are no variables')

        # evaluate the equations. This is just to get the number of equations, therefore this is done at zero
        x = [var.value for var in self._notDefinedVariables]
        n_eq = len(self._evaluateEquations(x))

        if n_eq == 0:
            raise ValueError('You have not specified any equtions')

        # compare the nubmer of variables to the number of equations
        if n_var != n_eq:
            raise ValueError(f'The number of equations ({n_eq}) is not equal to the number of variables ({n_var})')

    def _distributeVariables(self, x):
        for xi, var in zip(x, self._notDefinedVariables):
            var.value = xi

    def _evaluateEquations(self, x, scale=True):

        # distribute the variables
        self._distributeVariables(x)

        # evaluate the equations
        eq = []
        for sys in self.subSystems:
            if hasattr(sys, 'eq'):
                equations = sys.eq(sys)
                if not equations is None:
                    for elem in equations:
                        eq += [elem[0] - elem_i for elem_i in elem if elem_i != elem[0]]
        eq = [e.value for e in eq]

        # Normalize all equations. This makes the problem easier to solve numericallys
        if scale:
            if hasattr(self, 'scaleFactor'):
                for i, (e, scale) in enumerate(zip(eq, self.scaleFactor)):
                    if scale is None and e != 0:
                        scale = np.abs(e)
                        self.scaleFactor[i] = scale
                    if not scale is None:
                        eq[i] = e / scale
        return eq

    def printVariables(self):
        self.subSystems, self.subSystemNames = self._getSubsystem(self, self.name)
        self._getVariables()
        maxLength = np.max([len(elem) for elem in self.variableNames])
        for sys in self.subSystems:
            for key, item in sys.__dict__.items():
                if isinstance(item, variable):
                    if item in self.variables:
                        index = self.variables.index(item)
                        name = self.variableNames[index]
                        nameLength = len(name)
                        print(name, ' ' * (maxLength - nameLength + 3), item.printValue())
            print('')

    def writeVariablesOnDiagraom(self, existingPDF, font='Helvetica', fontSize=8):

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
    A = variable(1000, 'L')
    B = variable(0.5, 'min/L')
    C = A * B
    C.convertToOriginalUnit()
    D = C + B
    print(A)


if __name__ == "__main__":
    main()
