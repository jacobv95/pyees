import numpy as np
import scipy.odr as odr
from scipy.optimize import fsolve
import string
import warnings
import matplotlib.axes as axes
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
from typing import List
try:
    from variable import variable, arrayVariable, scalarVariable
except ImportError:
    from pyees.variable import variable, arrayVariable, scalarVariable



    
class _fit():
    def __init__(self, func, x, y, p0 = None, useParameters = None) -> None:

        self._hasRun = False
        if p0 is None:
            p0 = [0] * self._nParameters
        if len(p0) != self._nParameters:
            raise ValueError(f'The input "p0" has to be None or have a length of {self._nParameters}')

        if useParameters is None:
            useParameters = [True] * self._nParameters   
        if len(useParameters) != self._nParameters:
            raise ValueError(f'The input "useParameters" has to have a length of {self._nParameters}')
        
        self.func = func
        self.p0 = p0
        self.useParameters = useParameters

        if not (isinstance(x, arrayVariable) and isinstance(y, arrayVariable)):
            raise ValueError('The inputs has to be variables')

        self.xVal = x.value
        self.yVal = y.value
        self.xUnit = x._unitObject
        self.yUnit = y._unitObject
        self.xUncert = x.uncert
        self.yUncert = y.uncert

        indexesNotToUse = []
        for i in range(len(self.xVal)):
            if np.isnan(self.xVal[i]):
                indexesNotToUse.apd(i)
                continue
            if np.isnan(self.xUncert[i]):
                indexesNotToUse.append(i)
                continue
            if np.isnan(self.yVal[i]):
                indexesNotToUse.append(i)
                continue
            if np.isnan(self.yUncert[i]):
                indexesNotToUse.append(i)
                continue
        
        if indexesNotToUse:
            indexesToUse = [i for i in range(len(self.xVal)) if not i in indexesNotToUse]
            self.xVal = self.xVal[indexesToUse]
            self.xUncert = self.xUncert[indexesToUse]
            self.yVal = self.yVal[indexesToUse]
            self.yUncert = self.yUncert[indexesToUse]

        # uncertanties can not be 0
        self._sx = [elem if elem != 0 else 1e-10 for elem in self.xUncert]
        self._sy = [elem if elem != 0 else 1e-10 for elem in self.yUncert]

        # create the regression
        np.seterr('ignore')
        data = odr.RealData(self.xVal, self.yVal, sx=self._sx, sy=self._sy)
        regression = odr.ODR(data, odr.Model(self._func), beta0=self.p0)
        regression = regression.run()       
        
        ## create a list of coefficients
        self.coefficients = []
    
        units = self.getVariableUnits()
        for i in range(len(regression.beta)):
            var = variable(regression.beta[i], units[i], np.sqrt(regression.cov_beta[i,i]))
            self.coefficients.append(var)
        
        ## add the covariance between the coefficients
        for i in range(len(self.coefficients)):
            for j in range(len(self.coefficients)):
                if i == j:
                    continue
                self.coefficients[i].addCovariance(
                    var = self.coefficients[j],
                    covariance = regression.cov_beta[i,j],
                    unitStr = str(self.coefficients[i]._unitObject * self.coefficients[j]._unitObject)
                )

        # determine r-squared
        delta, epsilon = regression.delta, regression.eps
        residuals = ( np.sign(self.yVal-self._func(regression.beta, self.xVal)) * np.sqrt(delta**2 + epsilon**2) )
        ss_res = sum(residuals**2)
        ss_tot = sum((self.yVal - np.mean(self.yVal))**2)
        if ss_tot != 0:
            self.r_squared = variable(1 - (ss_res / ss_tot))
        else:
            self.r_squared = variable(1)
        
        
        
        dx_star = ( self.xUncert*np.sqrt( ((self.yUncert*delta)**2) /
                ( (self.yUncert*delta)**2 + (self.xUncert*epsilon)**2 ) ) )
        dy_star = ( self.yUncert*np.sqrt( ((self.xUncert*epsilon)**2) /
                ( (self.yUncert*delta)**2 + (self.xUncert*epsilon)**2 ) ) )
        sigma_odr = np.sqrt(dx_star**2 + dy_star**2)
        self._residualY = variable(residuals, self.yUnit, sigma_odr) 
        self._hasRun = True
        
    def getOnlyUsedTerms(self, B):
        for i in range(len(B)):
            if not self.useParameters[i]:
                if self._hasRun:
                    B[i] = variable(self.p0[i], self.getVariableUnits()[i])
                else:
                    B[i] = self.p0[i]
        return B
        
    def __str__(self):
        return self.func_name() + fr', $R^2 = {self.r_squared.value:.5f}$'


    @staticmethod
    def __splitPlotlyKeywordArguments(fig, kwargs):
        addTraceKwargs = {}
        if fig._has_subplots():
            
            if (not 'row' in kwargs or not 'col' in kwargs):
                raise ValueError('The figure is a plotly.graph_object.Figure that has subplots. The keyworkd arguments has to include both "row" and "col" ')
            
            addTraceKwargs['row'] = kwargs['row']
            addTraceKwargs['col'] = kwargs['col']
            kwargs.pop('row')
            kwargs.pop('col')
        return kwargs, addTraceKwargs

    def scatter(self, ax, showUncert=True, **kwargs):

        if all(self.xUncert == 0) and all(self.yUncert == 0):
            showUncert = False


        if isinstance(ax, axes.Axes):
            # scatter
            if showUncert:
                return ax.errorbar(self.xVal, self.yVal, xerr=self.xUncert, yerr=self.yUncert, linestyle='', **kwargs)
            else:
                return ax.scatter(self.xVal, self.yVal, **kwargs)
        elif isinstance(ax, go.Figure):

            kwargs, addTraceKwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
            
            if showUncert:
                ax.add_trace(
                    go.Scatter(
                        x = self.xVal,
                        y = self.yVal,
                        error_x = dict(array = self.xUncert),
                        error_y = dict(array = self.yUncert),
                        mode = 'markers',
                        **kwargs),
                    **addTraceKwargs
                    )
            else:
                ax.add_trace(
                    go.Scatter(x = self.xVal, y = self.yVal, mode = 'markers', **kwargs),
                    **addTraceKwargs
                    )
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')
        
    def scatterNormalizedResiduals(self, ax, **kwargs):
        
        np.seterr('ignore')
        scale = variable(np.array([1 / ((elemX**2 + elemY**2)**(1/2)) for elemX, elemY in zip(self._sx, self._sy)]))
        normRes = scale * self._residualY
        np.seterr('warn')
        
        if isinstance(ax, axes.Axes):
            if not 'label' in kwargs:
                kwargs['label'] = 'Normalized residuals'
            return ax.errorbar(self.xVal, normRes.value, xerr=self.xUncert, yerr=normRes.uncert, **kwargs)
        elif isinstance(ax, go.Figure):

            kwargs, addTraceKwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
            if not 'name' in kwargs:
                kwargs['name'] = 'Normalized residuals'
            ax.add_trace(
                go.Scatter(
                    x = self.xVal,
                    y = normRes.value,
                    error_x = dict(array = self.xUncert),
                    error_y = dict(array = normRes.uncert),
                    mode = 'markers',
                    **kwargs),
                    **addTraceKwargs
                )
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')

    def scatterResiduals(self, ax, **kwargs):
        
        
        if isinstance(ax, axes.Axes):
            if not 'label' in kwargs:
                kwargs['label'] = 'Residuals'
            return ax.errorbar(self.xVal, self._residualY.value, xerr=self.xUncert, yerr=self._residualY.uncert, linestyle='', **kwargs)
        elif isinstance(ax, go.Figure):

            kwargs, addTraceKwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
            if not 'name' in kwargs:
                kwargs['name'] = 'Residuals'
            ax.add_trace(
                go.Scatter(
                    x = self.xVal,
                    y = self._residualY.value,
                    error_x = dict(array = self.xUncert),
                    error_y = dict(array = self._residualY.uncert),
                    mode = 'markers',
                    **kwargs),
                    **addTraceKwargs
                )
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')

    def plotData(self, ax, **kwargs):

        if isinstance(ax, axes.Axes):
            if not 'label' in kwargs:
                kwargs['label'] = 'Data'
            return ax.plot(self.xVal, self.yVal, **kwargs)
        elif isinstance(ax, go.Figure):

            kwargs, addTraceKwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
            if not 'name' in kwargs:
                kwargs['name'] = 'Data'
            ax.add_trace(
                go.Scatter(
                    x = self.xVal,
                    y = self.yVal,
                    mode = 'lines',
                    **kwargs),
                    **addTraceKwargs
                )
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')

    def predict(self, x):
        if not isinstance(x, scalarVariable):
            raise ValueError('The input "x" has to be a variable')
        return self.func(x)
    
    def _predict(self, coeffs, x):
        # if not isinstance(x, scalarVariable):
        #     x = variable(x, self.xUnit)
        return self._func(coeffs, x)

    def plotUncertanty(self, ax, x = None, **kwargs):
        
        if x is None:
            x = variable(np.linspace(np.min(self.xVal), np.max(self.xVal), 100), self.xUnit)
        else:
            if not isinstance(x, arrayVariable):
                raise ValueError('The input "x" has to be a variable')
            
        y = self.predict(x)
        y = list(y.value + y.uncert) + [np.nan] + list(y.value - y.uncert)
        x = list(x.value) + [np.nan] + list(x.value)
        
        if isinstance(ax, axes.Axes):
            return ax.plot(x, y, **kwargs)
        elif isinstance(ax, go.Figure):

            kwargs, addTraceKwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
                                            
            
            ax.add_trace(
                go.Scatter(
                    x = x,
                    y = y,
                    mode = 'lines',
                    **kwargs),
                    **addTraceKwargs
                )
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')

    def plot(self, ax, x=None, **kwargs):

       
        if x is None:
            x = variable(np.linspace(np.min(self.xVal), np.max(self.xVal), 100), self.xUnit)
        else:
            if not isinstance(x, arrayVariable):
                raise ValueError('The input "x" has to be a variable')

        y = self.predict(x).value
        x = x.value
        
        if isinstance(ax, axes.Axes):
            if not 'label' in kwargs:
                kwargs['label'] = self.__str__()
            return ax.plot(x, y, **kwargs)
        elif isinstance(ax, go.Figure):

            kwargs, addTraceKwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
            
            if not 'name' in kwargs:
                kwargs['name'] = self.__str__()
                                
            ax.add_trace(
                go.Scatter(
                    x = x,
                    y = y,
                    mode = 'lines',
                    **kwargs),
                    **addTraceKwargs
                )
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')
          
    def addUnitToLabels(self, ax, **kwargs):
        self.addUnitToXLabel(ax, **kwargs)
        self.addUnitToYLabel(ax, **kwargs)

    def addUnitToXLabel(self, ax, **kwargs):
        if isinstance(ax, axes.Axes):
            xLabel = ax.get_xlabel()
            if xLabel:
                xLabel += ' '
            xLabel += rf'$\left[{self.xUnit.__str__(pretty=True)}\right]$'
            ax.set_xlabel(xLabel)
        elif isinstance(ax, go.Figure):
            
            if ax._has_subplots():
                
                _, kwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
                
                subplot = ax.get_subplot(kwargs['row'], kwargs['col'])
                xLabel = subplot.xaxis.title.text
                if xLabel is None:
                    xLabel = rf'$\left[{self.xUnit.__str__(pretty=True)}\right]$'
                else:
                    xLabel = rf'$\text{{{xLabel}}} \left[{self.xUnit.__str__(pretty=True)}\right]$'
                subplot.xaxis.title = xLabel      
                
            else:
                xLabel = ax.layout.xaxis.title.text
                if xLabel is None:
                    xLabel = rf'$\left[{self.xUnit.__str__(pretty=True)}\right]$'
                else:
                    xLabel = rf'$\text{{{xLabel}}} \left[{self.xUnit.__str__(pretty=True)}\right]$'
                ax.update_xaxes(title = xLabel)            
        
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')

    def addUnitToYLabel(self, ax, **kwargs):
        if isinstance(ax, axes.Axes):
            yLabel = ax.get_ylabel()
            if yLabel:
                yLabel += ' '
            yLabel += rf'$\left[{self.yUnit.__str__(pretty=True)}\right]$'
            ax.set_ylabel(yLabel)
        
        elif isinstance(ax, go.Figure):
            
            if ax._has_subplots():
                _, kwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
                
                subplot = ax.get_subplot(kwargs['row'], kwargs['col'])
                yLabel = subplot.yaxis.title.text
                if yLabel is None:
                    yLabel = rf'$\left[{self.yUnit.__str__(pretty=True)}\right]$'
                else:
                    yLabel = rf'$\text{{{yLabel}}} \left[{self.yUnit.__str__(pretty=True)}\right]$'
                subplot.yaxis.title = yLabel
        
            else:
                yLabel = ax.layout.yaxis.title.text
                if yLabel is None:
                    yLabel = rf'$\left[{self.yUnit.__str__(pretty=True)}\right]$'
                else:
                    yLabel = rf'$\text{{{yLabel}}} \left[{self.yUnit.__str__(pretty=True)}\right]$'
                ax.update_yaxes(title = yLabel)   
        
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')

    def plotUncertantyOfInputs(self, ax, n = 100, **kwargs):
            
        class Ellipse:
            def __init__(self, x, y, w, h):
                self.x = x
                self.y = y
                self.w = w
                self.h = h

                self.lineIn1 = None
                self.lineOut1 = None
                self.lineIn2 = None
                self.lineOut2 = None

        indexes = np.argsort(self.xVal)
        
        ellipses = []
        for i in indexes:
            ellipses.append(
                Ellipse(
                    self.xVal[i],
                    self.yVal[i],
                    2*self.xUncert[i],
                    2*self.yUncert[i]
                )
            )


        # Tangency conditions (discriminant zero)
        def tangency_conditions(vars):
            m, b = vars
            # Quadratic coefficients for intersection with ellipses
            def quad_coeffs(ellipse):
                A = (m**2 / ((ellipse.h/2)**2)) + (1 / ((ellipse.w/2)**2))
                B = (2*m*(b - ellipse.y) / ((ellipse.h/2)**2)) - (2*ellipse.x / ((ellipse.w/2)**2))
                C = ((b - ellipse.y)**2 / ((ellipse.h/2)**2)) + (ellipse.x**2 / ((ellipse.w/2)**2)) - 1
                return A, B, C

            A0, B0, C0 = quad_coeffs(ellipseA)
            A1, B1, C1 = quad_coeffs(ellipseB)

            # Tangency means discriminant is zero
            eq1 = B0**2 - 4*A0*C0
            eq2 = B1**2 - 4*A1*C1

            return [eq1, eq2]


        def find_tangent_intersection(m, b, x0, y0, w, h):
            
            def equations(x):
                y = m * x + b
                return ((x - x0)**2 / w**2) + ((y - y0)**2 / h**2) - 1
            warnings.filterwarnings("ignore")
            x = fsolve(equations, x0)[0]
            warnings.filterwarnings("default")
            
            return [[x, m * x + b]]
            

        def getPointsOnEllipseFurthestFromLine(ellipse, theta):
            x1 = ellipse.x + (ellipse.w / 2) * np.cos(theta + np.pi/2)
            y1 = ellipse.y + (ellipse.h / 2) * np.sin(theta + np.pi/2)
            
            x2 = ellipse.x + (ellipse.w / 2) * np.cos(theta - np.pi/2)
            y2 = ellipse.y + (ellipse.h / 2) * np.sin(theta - np.pi/2)

            return (x1, y1), (x2, y2)    



        def ccw(A,B,C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        # Return true if line segments AB and CD intersect
        def intersect(A,B,C,D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

        def line_intersection(line1, line2):
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div == 0:
                return None, None

            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return x, y


        xCoordinates1 = []
        yCoordinates1 = []

        for i in range(len(ellipses)-1):
            ellipseA = ellipses[i]
            ellipseB = ellipses[i+1]

            ## find the 2 points on each ellipse, where the points are perpendicular to the line between the centers of the two ellipses
            theta = np.atan2((ellipseB.y - ellipseA.y), (ellipseB.x - ellipseA.x))
            (x00, y00), (x01, y01) = getPointsOnEllipseFurthestFromLine(ellipseA, theta)
            (x10, y10), (x11, y11) = getPointsOnEllipseFurthestFromLine(ellipseB, theta)
            
            ## create all 2 lines between the 2 points
            lines = [
                [[x00,y00], [x10,y10]],
                [[x00,y00], [x11,y11]]
            ]


            # create a line from the center of ellipseA to ellipseB. Keep the line, that do not cross this line
            testLine = [[ellipseA.x, ellipseA.y], [ellipseB.x, ellipseB.y]]
            bools = [not intersect(line[0], line[1], testLine[0], testLine[1]) for line in lines]    
            index = np.where(bools)[0][0]
            line = lines[index]

            ## move the two lines such that they are tangent to the two ellipses        
            m = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
            b = line[1][1] - m * line[1][0]
            
            m, b = fsolve(tangency_conditions, [m,b])
            line[0] = find_tangent_intersection(m,b, ellipseA.x, ellipseA.y, ellipseA.w/2, ellipseA.h/2)[0]
            line[1] = find_tangent_intersection(m,b, ellipseB.x, ellipseB.y, ellipseB.w/2, ellipseB.h/2)[0]

            ellipseA.lineOut1 = line
            ellipseB.lineIn1 = line


        for ellipse in ellipses:

            if ellipse.lineIn1 is None or ellipse.lineOut1 is None: continue

            intersects = intersect(ellipse.lineIn1[0], ellipse.lineIn1[1], ellipse.lineOut1[0], ellipse.lineOut1[1])
            
            if not intersects: continue

            intersection = line_intersection(ellipse.lineIn1, ellipse.lineOut1)

            ellipse.lineIn1[1] = intersection
            ellipse.lineOut1[0] = intersection


        def getCoordinatesOfEllipseBetweenTangencyPoints(x, y, w, h, lineIn, lineOut):
            tangencyPointA = lineIn[1]
            tangencyPointB = lineOut[0]

            if tangencyPointA == tangencyPointB: return [], []
            

            theta = np.linspace(0, 2 * np.pi, n)

            # Parametric equations for an ellipse
            ellipse_x = x + (w / 2) * np.cos(theta)
            ellipse_y = y + (h / 2) * np.sin(theta)
            
            diff = [np.sqrt((x - tangencyPointA[0])**2 + (y - tangencyPointA[1])**2) for x, y in zip(ellipse_x, ellipse_y)]
            index1 = np.argmin(diff)
            
            diff = [np.sqrt((x - tangencyPointB[0])**2 + (y - tangencyPointB[1])**2) for x, y in zip(ellipse_x, ellipse_y)]
            index2 = np.argmin(diff)
            
            iMin = np.min([index1, index2])
            iMax = np.max([index1, index2])
            indexes = list(range(iMin, iMax+1))

            i1 = (index1 - 1) % n
            i2 = (index1 + 1) % n
            coord1 = [ellipse_x[i1], ellipse_y[i1]]
            coord2 = [ellipse_x[i2], ellipse_y[i2]]

            d1 = np.sqrt((coord1[0] - lineIn[0][0])**2 + (coord1[1] - lineIn[0][1])**2)
            d2 = np.sqrt((coord2[0] - lineIn[0][0])**2 + (coord2[1] - lineIn[0][1])**2)

            if d1 >= d2:
                indexToTest = i2
            else:
                indexToTest = i1
            

            if indexToTest in indexes:
                index = indexes[-1]
                out = []
                for i in range(n - len(indexes)):
                    out.append(index)
                    if index == n-1:
                        index = 0
                    else:
                        index += 1
                indexes = out

            xs = [ellipse_x[i] for i in indexes]
            ys = [ellipse_y[i] for i in indexes]

            coord1 = [xs[0], ys[0]]
            coord2 = [xs[-1], ys[-1]]

            d1 = np.sqrt((coord1[0] - lineIn[1][0])**2 + (coord1[1] - lineIn[1][1])**2)
            d2 = np.sqrt((coord2[0] - lineIn[1][0])**2 + (coord2[1] - lineIn[1][1])**2)

            if d2 < d1:
                xs = list(reversed(xs))
                ys = list(reversed(ys))
            
            return xs, ys

        for ellipseA in ellipses:
            
            if not ellipseA.lineIn1 is None:
                xCoordinates1.append(ellipseA.lineIn1[0][0])
                yCoordinates1.append(ellipseA.lineIn1[0][1])
                xCoordinates1.append(ellipseA.lineIn1[1][0])
                yCoordinates1.append(ellipseA.lineIn1[1][1])
                
            
            if (not ellipseA.lineIn1 is None) and (not ellipseA.lineOut1 is None):

                xs, ys = getCoordinatesOfEllipseBetweenTangencyPoints(
                    ellipseA.x,
                    ellipseA.y,
                    ellipseA.w,
                    ellipseA.h,
                    ellipseA.lineIn1,
                    ellipseA.lineOut1
                )

                xCoordinates1 += xs
                yCoordinates1 += ys   
                
                
        xCoordinates2 = []
        yCoordinates2 = []

        for i in range(len(ellipses)-1):
            ellipseA = ellipses[i]
            ellipseB = ellipses[i+1]

            ## find the 2 points on each ellipse, where the points are perpendicular to the line between the centers of the two ellipses
            theta = np.atan2((ellipseB.y - ellipseA.y), (ellipseB.x - ellipseA.x))
            (x00, y00), (x01, y01) = getPointsOnEllipseFurthestFromLine(ellipseA, theta)
            (x10, y10), (x11, y11) = getPointsOnEllipseFurthestFromLine(ellipseB, theta)

            ## create all 2 lines between the 2 points
            lines = [
                [[x01,y01], [x10,y10]],
                [[x01,y01], [x11,y11]]
            ]

            # create a line from the center of ellipseA to ellipseB. Keep the line, that do not cross this line
            testLine = [[ellipseA.x, ellipseA.y], [ellipseB.x, ellipseB.y]]
            bools = [not intersect(line[0], line[1], testLine[0], testLine[1]) for line in lines]    
            index = np.where(bools)[0][0]
            line = lines[index]
            
            ## move the two lines such that they are tangent to the two ellipses        
            m = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
            b = line[1][1] - m * line[1][0]
            
            m, b = fsolve(tangency_conditions, [m,b])
            line[0] = find_tangent_intersection(m,b, ellipseA.x, ellipseA.y, ellipseA.w/2, ellipseA.h/2)[0]
            line[1] = find_tangent_intersection(m,b, ellipseB.x, ellipseB.y, ellipseB.w/2, ellipseB.h/2)[0]

            ellipseA.lineOut2 = line
            ellipseB.lineIn2 = line



        for ellipse in ellipses:

            if ellipse.lineIn2 is None or ellipse.lineOut2 is None: continue

            intersects = intersect(ellipse.lineIn2[0], ellipse.lineIn2[1], ellipse.lineOut2[0], ellipse.lineOut2[1])
            
            if not intersects: continue

            intersection = line_intersection(ellipse.lineIn2, ellipse.lineOut2)

            ellipse.lineIn2[1] = intersection
            ellipse.lineOut2[0] = intersection

            
        for ellipseA in ellipses:
            if not ellipseA.lineIn2 is None:
                xCoordinates2.append(ellipseA.lineIn2[0][0])
                yCoordinates2.append(ellipseA.lineIn2[0][1])
                xCoordinates2.append(ellipseA.lineIn2[1][0])
                yCoordinates2.append(ellipseA.lineIn2[1][1])
            
            if (not ellipseA.lineIn2 is None) and (not ellipseA.lineOut2 is None):
                
                xs, ys = getCoordinatesOfEllipseBetweenTangencyPoints(
                        ellipseA.x,
                        ellipseA.y,
                        ellipseA.w,
                        ellipseA.h,
                        ellipseA.lineIn2,
                        ellipseA.lineOut2
                )
                
                xCoordinates2 += xs
                yCoordinates2 += ys            


        ellipses[0].lineIn2 = [ellipses[0].lineOut2[1], ellipses[0].lineOut2[0]]
        xCoordinates3, yCoordinates3 = getCoordinatesOfEllipseBetweenTangencyPoints(
            ellipses[0].x,
            ellipses[0].y,
            ellipses[0].w,
            ellipses[0].h,
            ellipses[0].lineIn2,
            ellipses[0].lineOut1
        )

        ellipses[-1].lineOut2 = [ellipses[-1].lineIn2[1], ellipses[-1].lineIn2[0]]
        xCoordinates4, yCoordinates4 = getCoordinatesOfEllipseBetweenTangencyPoints(
            ellipses[-1].x,
            ellipses[-1].y,
            ellipses[-1].w,
            ellipses[-1].h,
            ellipses[-1].lineIn1,
            ellipses[-1].lineOut2
        )


        xCoordinates2 = list(reversed(xCoordinates2)) 
        yCoordinates2 = list(reversed(yCoordinates2)) 

        xCoordinates = xCoordinates1 + xCoordinates4 + xCoordinates2 + xCoordinates3
        yCoordinates = yCoordinates1 + yCoordinates4 + yCoordinates2 + yCoordinates3

        xy = []
        for i in range(len(xCoordinates)):
            xy.append([xCoordinates[i], yCoordinates[i]])
        
        return ax.add_patch(patches.Polygon(xy, **kwargs))

    def scatterUncertatyAsEllipses(self, ax, **kwargs):
            
        class Ellipse:
            def __init__(self, x, y, w, h):
                self.x = x
                self.y = y
                self.w = w
                self.h = h

                self.lineIn1 = None
                self.lineOut1 = None
                self.lineIn2 = None
                self.lineOut2 = None

        indexes = np.argsort(self.xVal)
        
        ellipses = []
        for i in indexes:
            ellipses.append(
                Ellipse(
                    self.xVal[i],
                    self.yVal[i],
                    2*self.xUncert[i],
                    2*self.yUncert[i]
                )
            )

        out = []
        for ellipse in ellipses:
            theta = np.linspace(0, 2 * np.pi, 100)

            # Parametric equations for an ellipse
            ellipse_x = ellipse.x + (ellipse.w / 2) * np.cos(theta)
            ellipse_y = ellipse.y + (ellipse.h / 2) * np.sin(theta)

            out.append(ax.plot(ellipse_x, ellipse_y, **kwargs))

        return out
         
    def func(self, x):
        return self._func(self.coefficients, x)

    def evaluateFit(self):
        ## TODO evaluate the fit - use the degrees of freedom as in the book
        
        pred = self._predict(self.xVal)
        
        count = 0
        for yi, uyi, pi in zip(self.yVal, self.yUncert, pred):
            
            ## count the number of datapoints were the prediction is within 1 standard deviation
            if yi == pi:
                count += 1
            elif yi > pi:
                if yi - uyi < pi:
                    count += 1
            else:
                if yi + uyi > pi:
                    count += 1
            
            
        ## determine the relative number of datapoints where the prediction is within 1 standard deviation
        r = count / len(self.xVal)
        
        if r < 0.68:
            warnings.warn(f'{(1-r)*100:.2f}% of the datapoints were more than 1 standard deviation away from the regression. This might indicate that the regression is poor', category=Warning, stacklevel=1)
            


class dummy_fit(_fit):
    """Create a dummy fit of the input data. No regression is performed when creating a dummy fit. The dummy fit object is onely used to easily plot the data
    
    Parameters:
    x : variable
        The independent variable
    y : variable
        The dependents variable
    """
    def __init__(self, x : variable,  y : variable):
        
        if not (isinstance(x, arrayVariable) and isinstance(y, arrayVariable)):
            raise ValueError('The inputs has to be variables')

        self.xVal = x.value
        self.yVal = y.value
        self.xUnit = x._unitObject
        self.yUnit = y._unitObject
        self.xUncert = x.uncert
        self.yUncert = y.uncert

        self.r_squared = variable(1)

    def func(self, x):
        if isinstance(x, arrayVariable):
            return variable([1] * len(x))
        return variable(1)
    
    def func_name(self):
        return '1'


class exp_fit(_fit):
    """Create an exponential fit of the input data and easily plot the regression.
    
    f(x) = a * exp(b * x) + c

    Parameters:
    x : variable
        The independent variable
    y : variable
        The dependents variable
    p0 : List(float) = None
        The initial guesses in order (a,b,c). If p0 is set to None, then all parameters are initialized to 0.
    useParameters : List(double) = None
        Wheater or not to use the parameters in order (a,b,c). If an element is set to "False", then the parameter is fixed to the initial guess
    """
    
    def __init__(self, x : variable, y: variable, p0 : List[float] = None, useParameters : List[bool] | None = [True, True, False]):
        self._nParameters = 3
        if x.unit != '1':
            raise ValueError('The variable "x" cannot have a unit')
        _fit.__init__(self, self.func, x, y, p0=p0, useParameters = useParameters)

    def getVariableUnits(self):
        return [self.yUnit, '', self.yUnit]

    def _func(self, B, x):
        a,b,c = self.getOnlyUsedTerms(B)
        return a * np.exp(b * x) + c

    def func_name(self):
        a,b,c = self.coefficients
        return fr'$a\cdot b^(b\cdot x)+d,\quad a={a.__str__(pretty = True)}, \quad b={b.__str__(pretty = True)}, \quad c={c.__str__(pretty = True)}$'


class pow_fit(_fit):
    """Create a power fit of the input data and easily plot the regression.
    
    f(x) = a * x**b + c

    Parameters:
    x : variable
        The independent variable
    y : variable
        The dependents variable
    p0 : List(float) = None
        The initial guesses in order (a,b,c). If p0 is set to None, then all parameters are initialized to 0.
    useParameters : List(double) = None
        Wheater or not to use the parameters in order (a,b,c). If an element is set to "False", then the parameter is fixed to the initial guess
    """
    def __init__(self, x : variable, y: variable, p0 : List[float] = None, useParameters : List[bool] | None = [True,True,False]):
        self._nParameters = 3
        
        if x.unit != '1':
            raise ValueError('The variable "x" cannot have a unit')
        
        _fit.__init__(self, self.func, x, y, p0=p0, useParameters=useParameters)

    def getVariableUnits(self):
        return [self.yUnit, '1', self.yUnit]

    def _func(self, B, x):
        a,b,c = self.getOnlyUsedTerms(B)
        return a * x**b+c

    def func_name(self):
        return fr'$a x^b+c,\quad a={self.coefficients[0].__str__(pretty = True)}, \quad b={self.coefficients[1].__str__(pretty = True)}, \quad b={self.coefficients[1].__str__(pretty = True)}$'


def lin_fit(x : variable, y: variable, p0 : List[float] = None, useParameters : List[bool] | None = None):
    """Create a linear fit of the input data and easily plot the regression.
    
    f(x) = a*x + b

    Parameters:
    x : variable
        The independent variable
    y : variable
        The dependents variable
    p0 : List(float) = None
        The initial guesses in order (a,b). If p0 is set to None, then all parameters are initialized to 0.
    useParameters : List(double) = None
        Wheater or not to use the parameters in order (a,b). If an element is set to "False", then the parameter is fixed to the initial guess
    """
    return pol_fit(x, y, deg=1, p0=p0, useParameters=useParameters)


class pol_fit(_fit):
    """Create a polynomial fit of the input data and easily plot the regression.
    
    f(x) = sum_0^n ( a_i * x**(n-i) )

    Parameters:
    x : variable
        The independent variable
    y : variable
        The dependents variable
    p0 : List(float) = None
        The initial guesses in order (a_n, ..., a_1, a_0). If p0 is set to None, then all parameters are initialized to 0.
    useParameters : List(double) = None
        Wheater or not to use the parameters in order (a_n, ..., a_1, a_0). If an element is set to "False", then the parameter is fixed to the initial guess
    deg: int
        The degree of the polynomial
    """
    def __init__(self, x : variable, y: variable, p0 : List[float] = None, useParameters : List[bool] | None = None, deg :int = 2):
        self._nParameters = deg + 1
        self.deg = deg

        _fit.__init__(self, self.func, x, y, p0=p0, useParameters=useParameters)

    def getVariableUnits(self):
        units = []
        n = self.deg
        for i in range(n + 1):
            if i != n:
                u = self.yUnit / self.xUnit ** (n-i)
            else:
                u = self.yUnit
            units.append(u)
        return units

    def _func(self, B, x):
        B = self.getOnlyUsedTerms(B)
        out = 0
        for i,b in enumerate(B):
            out += b * x**(self.deg - i)
        return out

    def func_name(self):
        out = '$'
        n = self.deg
        for i in range(n + 1):
            exponent = n - i
            if i == 0:
                out += fr'{string.ascii_lowercase[i]}'
            else:
                out += fr'+{string.ascii_lowercase[i]}'
            if exponent != 0:
                out += fr'x'
            if exponent > 1:
                    out += fr'^{exponent}'
       
        for i in range(n + 1):
            out += fr', {string.ascii_lowercase[i]}={self.coefficients[i].__str__(pretty = True)}'
        out += '$'
        return out


class logistic_fit(_fit):
    """Create a logistic fit of the input data and easily plot the regression.
    
    f(x) = L / (1 + exp(-k * (x - x0)))

    Parameters:
    x : variable
        The independent variable
    y : variable
        The dependents variable
    p0 : List(float) = None
        The initial guesses in order (L,k,x0). If p0 is set to None, then all parameters are initialized to 0.
    useParameters : List(double) = None
        Wheater or not to use the parameters in order (L,k,x0). If an element is set to "False", then the parameter is fixed to the initial guess
    """
    def __init__(self, x : variable, y: variable, p0 : List[float] = None, useParameters : List[bool] | None = None):

        self._nParameters = 3

        if x.unit != '1':
            raise ValueError('The variable "x" cannot have a unit')
        _fit.__init__(self, self.func, x, y, p0=p0, useParameters=useParameters)

    def getVariableUnits(self):
        return [self.yUnit,'1','1']
    
    def _func(self, B, x):
        L,k,x0 = self.getOnlyUsedTerms(B)
        return L / (1 + np.exp(-k * (x - x0)))

    def func_name(self):
        L = self.coefficients[0].__str__(pretty=True)
        k = self.coefficients[1].__str__(pretty=True)
        x0 = self.coefficients[2].__str__(pretty=True)

        out = fr'$\\frac{{L}}{{1 + e^{{-k\cdot (x-x_0)}}}}'
        out += fr'\quad L={L}'
        out += fr'\quad k={k}'
        out += fr'\quad x_0={x0}$'
        return out


def crateNewFitClass(func, funcNameFunc, getVariableUnitsFunc, nParameters):
    
    class newFit(_fit):
        
        def __init__(self, x : variable, y: variable, p0 : List[float] | None = None, useParameters : List[bool] | None = None):
            
            self.getVariableUnitsFunc = getVariableUnitsFunc
            self.func = func
            self.func_nameFunc = funcNameFunc
            self._nParameters = nParameters
            
            _fit.__init__(self, self.func, x, y, p0=p0, useParameters = useParameters)

        def _func(self,B,x):
            return self.func(B,x)

        def func_name(self):
            return self.func_nameFunc(self.coefficients)
        
        def getVariableUnits(self):
            return self.getVariableUnitsFunc(self.xUnit, self.yUnit)

    return newFit


class multi_variable_lin_fit(_fit):
    def __init__(self, x : List[variable], y: variable, p0 : List[float] = None, useParameters : List[bool] | None = None):
        
        self._nParameters = len(x) + 1

        self._hasRun = False
        if p0 is None:
            p0 = [0] * self._nParameters
        if len(p0) != self._nParameters:
            raise ValueError(f'The input "p0" has to be None or have a length of {self._nParameters}')

        if useParameters is None:
            useParameters = [True] * self._nParameters   
        if len(useParameters) != self._nParameters:
            raise ValueError(f'The input "useParameters" has to have a length of {self._nParameters}')
        
        self.p0 = p0
        self.useParameters = useParameters

        if not isinstance(x, List):
            raise ValueError('The argument "x" has to be List of variables')

        for elem in x:
            if not isinstance(elem, arrayVariable):
                raise ValueError('The argument "x" has to be a list of variables')

        if not isinstance(y, arrayVariable):
            raise ValueError('The argument "y" has to be variables')


        self.yVal = y.value
        self.yUnit = y._unitObject
        self.yUncert = y.uncert
        self.xVal = [elem.value for elem in x]
        self.xUnit = [elem._unitObject for elem in x]
        self.xUncert = [elem.uncert for elem in x]
        
        


        indexesNotToUse = []
        for i in range(len(self.yVal)):
            if np.isnan(self.yVal[i]):
                indexesNotToUse.apd(i)
                continue
            if np.isnan(self.yUncert[i]):
                indexesNotToUse.append(i)
                continue

            for xValue, xUncert in zip(self.xVal, self.xUncert):

                if np.isnan(xValue[i]):
                    indexesNotToUse.append(i)
                    continue
                if np.isnan(xUncert[i]):
                    indexesNotToUse.append(i)
                    continue
         
        
        if indexesNotToUse:
            indexesToUse = [i for i in range(len(self.yVal)) if not i in indexesNotToUse]
            self.yVal = self.yVal[indexesToUse]
            self.yUncert = self.yUncert[indexesToUse]
            for i in range(len(self.xVal)):
                self.xVal[i] = self.xVal[i][indexesToUse]
                self.xUncert[i] = self.xUncert[i][indexesToUse]

        

        # uncertanties can not be 0
        self._sx = []
        for elem in self.xUncert:
            self._sx.append([e if e != 0 else 1e-10 for e in elem])
        self._sy = [elem if elem != 0 else 1e-10 for elem in self.yUncert]

        # create the regression
        np.seterr('ignore')
        data = odr.RealData(self.xVal, self.yVal, sx = self._sx, sy=self._sy)
        regression = odr.ODR(data, odr.Model(self._ffunc), beta0=self.p0)
        regression = regression.run()   
            

        ## create a list of coefficients
        self.coefficients = []
    
        units = self.getVariableUnits()
        for i in range(len(regression.beta)):
            var = variable(regression.beta[i], units[i], np.sqrt(regression.cov_beta[i,i]))
            self.coefficients.append(var)
        
        ## add the covariance between the coefficients
        for i in range(len(self.coefficients)):
            for j in range(len(self.coefficients)):
                if i == j:
                    continue
                self.coefficients[i].addCovariance(
                    var = self.coefficients[j],
                    covariance = regression.cov_beta[i,j],
                    unitStr = str(self.coefficients[i]._unitObject * self.coefficients[j]._unitObject)
                )

        # determine r-squared
        delta, epsilon = regression.delta, regression.eps
        out = np.zeros(epsilon.shape)
        for i in range(len(delta)):
            out[i] = np.sqrt(sum([delta[i,j]**2]))
        delta = out
        residuals = ( np.sign(self.yVal-self._func(regression.beta, self.xVal)) * np.sqrt(delta**2 + epsilon**2) )
        
        ss_res = sum(residuals**2)
        ss_tot = sum((self.yVal - np.mean(self.yVal))**2)
        if ss_tot != 0:
            self.r_squared = variable(1 - (ss_res / ss_tot))
        else:
            self.r_squared = variable(1)
        
        
        
        out = []
        for i in range(len(self.yVal)):
            o = 0
            for j in range(len(self.xUncert)):
                o += self.xUncert[j][i]**2
            out.append(np.sqrt(o))
        xUncert = out
        
        dx_star = ( xUncert*np.sqrt( ((self.yUncert*delta)**2) /
                ( (self.yUncert*delta)**2 + (xUncert*epsilon)**2 ) ) )
        dy_star = ( self.yUncert*np.sqrt( ((xUncert*epsilon)**2) /
                ( (self.yUncert*delta)**2 + (xUncert*epsilon)**2 ) ) )
        sigma_odr = np.sqrt(dx_star**2 + dy_star**2)
        self._residualY = variable(residuals, self.yUnit, sigma_odr) 
        self._hasRun = True

    def predict(self, x):
        return self._func(self.coefficients, x)

    def getVariableUnits(self):
        out = []
        for i in range(len(self.xUnit)):
            out.append(self.yUnit / self.xUnit[i])
        out.append(self.yUnit)
        return out
    
    def _ffunc(self, B, X):
                
        B = self.getOnlyUsedTerms(B)
        out = 0
        for i in range(len(B)-1):
            ## I DO NOT KNOWN WHY I HAVE TO USE XVAL INSTEAD OF X
            ## BUT THIS WORKS
            out += B[i] * self.xVal[i]
        out += B[-1]

        return out
    
    def _func(self, B, x):
                
        B = self.getOnlyUsedTerms(B)
        out = 0
        for i in range(len(B)-1):
            out += B[i] * x[i]
        out += B[-1]

        return out

    def func_name(self):
        B = [elem.__str__(pretty=True) for elem in self.coefficients]
        
        out = ""
        for i in range(len(B)-1):
            if out:
                out += " + "
            out += fr"a_{i+1}\cdot x_{i+1}"
        if out:
            out += " + "
        out += fr"a_{len(B)}"

        for i in range(len(B)):
            out += fr"\quad a_{i+1}={B[i]}"

        return out

    def scatter(self, ax, index, showUncert=True, **kwargs):

        if all(self.xUncert[index] == 0) and all(self.yUncert == 0):
            showUncert = False


        if isinstance(ax, axes.Axes):
            # scatter
            if showUncert:
                return ax.errorbar(self.xVal[index], self.yVal, xerr=self.xUncert, yerr=self.yUncert, linestyle='', **kwargs)
            else:
                return ax.scatter(self.xVal[index], self.yVal, **kwargs)
        elif isinstance(ax, go.Figure):

            kwargs, addTraceKwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
            
            if showUncert:
                ax.add_trace(
                    go.Scatter(
                        x = self.xVal[index],
                        y = self.yVal,
                        error_x = dict(array = self.xUncert),
                        error_y = dict(array = self.yUncert),
                        mode = 'markers',
                        **kwargs),
                    **addTraceKwargs
                    )
            else:
                ax.add_trace(
                    go.Scatter(x = self.xVal[index], y = self.yVal, mode = 'markers', **kwargs),
                    **addTraceKwargs
                    )
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')
        
    def scatterNormalizedResiduals(self, ax, index, **kwargs):

        np.seterr('ignore')
        scale = variable(np.array([1 / ((elemX**2 + elemY**2)**(1/2)) for elemX, elemY in zip(self._sx[index], self._sy)]))
        normRes = scale * self._residualY
        np.seterr('warn')
        
        if isinstance(ax, axes.Axes):
            if not 'label' in kwargs:
                kwargs['label'] = 'Normalized residuals'
            return ax.errorbar(self.xVal[index], normRes.value, xerr=self.xUncert[index], yerr=normRes.uncert, **kwargs)
        elif isinstance(ax, go.Figure):

            kwargs, addTraceKwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
            if not 'name' in kwargs:
                kwargs['name'] = 'Normalized residuals'
            ax.add_trace(
                go.Scatter(
                    x = self.xVal[index],
                    y = normRes.value,
                    error_x = dict(array = self.xUncert[index]),
                    error_y = dict(array = normRes.uncert),
                    mode = 'markers',
                    **kwargs),
                    **addTraceKwargs
                )
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')

    def scatterResiduals(self, ax, index, **kwargs):
        
        if isinstance(ax, axes.Axes):
            if not 'label' in kwargs:
                kwargs['label'] = 'Residuals'
            return ax.errorbar(self.xVal[index], self._residualY.value, xerr=self.xUncert[index], yerr=self._residualY.uncert, linestyle='', **kwargs)
        elif isinstance(ax, go.Figure):

            kwargs, addTraceKwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
            if not 'name' in kwargs:
                kwargs['name'] = 'Residuals'
            ax.add_trace(
                go.Scatter(
                    x = self.xVal[index],
                    y = self._residualY.value,
                    error_x = dict(array = self.xUncert[index]),
                    error_y = dict(array = self._residualY.uncert),
                    mode = 'markers',
                    **kwargs),
                    **addTraceKwargs
                )
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')

    def plotData(self, ax, index, **kwargs):
        if isinstance(ax, axes.Axes):
            if not 'label' in kwargs:
                kwargs['label'] = 'Data'
            return ax.plot(self.xVal[index], self.yVal, **kwargs)
        elif isinstance(ax, go.Figure):

            kwargs, addTraceKwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
            if not 'name' in kwargs:
                kwargs['name'] = 'Data'
            ax.add_trace(
                go.Scatter(
                    x = self.xVal[index],
                    y = self.yVal,
                    mode = 'lines',
                    **kwargs),
                    **addTraceKwargs
                )
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')

    def plotUncertanty(self, ax, index, x = None, **kwargs):
        if x is None:
            x = []
            for i in range(len(self.xVal)):
                if i != index:
                    x.append(variable([np.mean(self.xVal[i])] * 100, self.xUnit[i]))
                else:
                    x.append(variable(np.linspace(np.min(self.xVal[i]), np.max(self.xVal[i]), 100), self.xUnit[i]))
        else:
            if not isinstance(x, List):
                raise ValueError('The argument "x" has to be List of variables')

            for elem in x:
                if not isinstance(elem, arrayVariable):
                    raise ValueError('The argument "x" has to be a list of variables')
              
        y = self.predict(x)
        y = list(y.value + y.uncert) + [np.nan] + list(y.value - y.uncert)
        x = list(x[index].value) + [np.nan] + list(x[index].value)
        
        if isinstance(ax, axes.Axes):
            return ax.plot(x, y, **kwargs)
        elif isinstance(ax, go.Figure):

            kwargs, addTraceKwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
                                            
            
            ax.add_trace(
                go.Scatter(
                    x = x,
                    y = y,
                    mode = 'lines',
                    **kwargs),
                    **addTraceKwargs
                )
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')

    def plot(self, ax, index, x=None, **kwargs):
        if x is None:
            x = []
            for i in range(len(self.xVal)):
                if i != index:
                    x.append(variable([np.mean(self.xVal[i])] * 100, self.xUnit[i]))
                else:
                    x.append(variable(np.linspace(np.min(self.xVal[i]), np.max(self.xVal[i]), 100), self.xUnit[i]))
        else:
            if not isinstance(x, List):
                raise ValueError('The argument "x" has to be List of variables')

            for elem in x:
                if not isinstance(elem, arrayVariable):
                    raise ValueError('The argument "x" has to be a list of variables')
            
        y = self.predict(x).value
        x = x[index].value
        
        if isinstance(ax, axes.Axes):
            if not 'label' in kwargs:
                kwargs['label'] = self.__str__()
            return ax.plot(x, y, **kwargs)
        elif isinstance(ax, go.Figure):

            kwargs, addTraceKwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
            
            if not 'name' in kwargs:
                kwargs['name'] = self.__str__()
                                
            ax.add_trace(
                go.Scatter(
                    x = x,
                    y = y,
                    mode = 'lines',
                    **kwargs),
                    **addTraceKwargs
                )
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')
                    
    def addUnitToLabels(self, ax, index, **kwargs):
        self.addUnitToXLabel(ax, index, **kwargs)
        self.addUnitToYLabel(ax, **kwargs)

    def addUnitToXLabel(self, ax, index, **kwargs):

        if isinstance(ax, axes.Axes):
            xLabel = ax.get_xlabel()
            if xLabel:
                xLabel += ' '
            xLabel += rf'$\left[{self.xUnit[index].__str__(pretty=True)}\right]$'
            ax.set_xlabel(xLabel)
        elif isinstance(ax, go.Figure):
            
            if ax._has_subplots():
                
                _, kwargs = self.__splitPlotlyKeywordArguments(ax, kwargs)
                
                subplot = ax.get_subplot(kwargs['row'], kwargs['col'])
                xLabel = subplot.xaxis.title.text
                if xLabel is None:
                    xLabel = rf'$\left[{self.xUnit[index].__str__(pretty=True)}\right]$'
                else:
                    xLabel = rf'$\text{{{xLabel}}} \left[{self.xUnit[index].__str__(pretty=True)}\right]$'
                subplot.xaxis.title = xLabel      
                
            else:
                xLabel = ax.layout.xaxis.title.text
                if xLabel is None:
                    xLabel = rf'$\left[{self.xUnit[index].__str__(pretty=True)}\right]$'
                else:
                    xLabel = rf'$\text{{{xLabel}}} \left[{self.xUnit[index].__str__(pretty=True)}\right]$'
                ax.update_xaxes(title = xLabel)            
        
        else:
            raise ValueError('The axes has to be a matplotlib axes or a plotly graphs object')

    def plotUncertantyOfInputs(self, ax, index, n = 100, **kwargs):
                
            xUncert = np.interp(np.linspace(0, len(self.xUncert[index])-1, n), range(len(self.xUncert[index])), self.xUncert[index])
            yUncert = np.interp(np.linspace(0, len(self.yUncert)-1, n), range(len(self.yUncert)), self.yUncert)
            xValue = np.interp(np.linspace(0, len(self.xVal[index])-1, n), range(len(self.xVal[index])), self.xVal[index])
            yValue = np.interp(np.linspace(0, len(self.yVal)-1, n), range(len(self.yVal)), self.yVal)
            
            for i in range(n):
                ax.add_patch(Ellipse((xValue[i], yValue[i]), 2*xUncert[i], 2*yUncert[i], **kwargs))
                if 'label' in kwargs: kwargs['label'] = None

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    flow = variable([9.14, 13.32, 18.17, 24.33, 29.18, 34.13, 39.20, 44.70, 50.10, 55.71, 60.56, 66.17, 70.36, 74.65, 78.39, 81.92, 86.10], 'm3/h')
    dp = variable([79.20, 78.39, 77.43, 75.50, 73.90, 72.29, 70.20, 67.63, 64.42, 61.85, 58.31, 54.62, 51.41, 47.71, 44.50, 40.80, 36.63], 'h')

    flow = variable(flow.value, flow.unit, flow.value * 0.08)
    dp = variable(dp.value, dp.unit, dp.value * 0.10)

    f = pol_fit(flow, dp)

    fig, ax = plt.subplots()
    f.scatter(ax)
    f.plot(ax)
    a = f.plotUncertantyOfInputs(ax, alpha = 0.5)
    f.scatterUncertatyAsEllipses(ax, color = 'red')
    plt.show()
    print(type(a))