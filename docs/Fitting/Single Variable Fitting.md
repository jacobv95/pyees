# Single Variable fitting

The package includes a tool to produce fits. The following fits are known
 - dummy fit            f(x) = 1
 - Linear fit           f(x) = a*x + b
 - Polynomial fit       f(x) = sum_0^n (a_i * x**(n - i))
 - Power fit            f(x) = a * b**x + c
 - Exponential fit      f(x) = a * exp(b * x) + c
 - Logistic fit         f(x) = L / (1 + exp(-k * (x - x0)))


The dummy fit will always return a constant function with a value of 1. This "fit" is used to easily plot data.

```
F_dummy = dummy_fit(x: variable, y: variable)
F_lin = lin_fit(x: variable, y: variable, p0: list[float] | None = None, useParameters : list[bool] | None = None)
F_pol = pol_fit(x: variable, y: variable, p0: list[float] | None = None, useParameters : list[bool] | None = None, deg: int =2)
F_pow = pow_fit(x: variable, y: variable, p0: list[float] | None = None, useParameters : list[bool] | None = [True,True,False])
F_exp = exp_fit(x: variable, y: variable, p0: list[float] | None = None, useParameters : list[bool] | None = [True,True,False])
F_logistic = logistic_fit(x: variable, y: variable, p0: list[float] | None = None, useParameters : list[bool] | None = None)
```
 - x is the x data used to generate the regression
 - y is the y data used to generate the regression
 - p0 is the initial guess of the regression coefficients. The coefficients will be initialized to 0 if p0 is set to None
 - deg is the degree of the polynomial
 - useParameters defines which coefficients to tune. If a coefficient is not tuned, then it will remain as the initial guesses.

## Priting
The fit can be printed. This is done in latex format. First the model is printed, and then the coefficients are printed.

```
print(fit) -> str
```

## Predict
Once a fit has been made this can be used to create a prediction. 

```
fit.predict(x: variable) -> variable
```



## Plot
The fit class has a function to plot

```
fit.plot(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, x: variable = None, **kwargs) -> List[matplotlib.lines.Line2D] | None
```

Parameters:
- ax is the axis or figure to plot on
- x is an arrayVariable of x values used to plot the regression. If None, then 100 points will be plotted within the range of the data used to create the fit.
- **kwargs are key word arguments for matplotlib.pyplot.plot or plotly.graph_objects.Scatter

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a list of a single element. That being the matplotlib.lines.Line2D created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


## Scatter
The fit class has a function to scatter the data used to generate the fit.

```
fit.scatter(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, showUncert: bool = True, **kwargs) -> matplotlib.collections.PathCollection | None
```

Parameters:
- ax is the axis or figure to plot on
- showUncert is a bool. Errorbars are shown if true
- **kwargs are key word arguments for matplotlib.pyplot.scatter  

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a matplotlib.collections.PathCollection objected created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.

## Plot data
The fit class has a function to plot the data used to generate the fit.

```
fit.plotData(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, **kwargs) -> List[matplotlib.lines.Line2D] | None
```

Parameters:
- ax is the axis or figure to plot on
- **kwargs are key word arguments for matplotlib.pyplot.plot  

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a list of a single element. That being the matplotlib.lines.Line2D created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


## Plot uncertanty
The fit class has a function to plot the uncertanty bands of the regression

```
fit.plotUncertanty(ax, x: variable = None, **kwargs) -> List[matplotlib.lines.Line2D] | None
```

Parameters:
- ax is the axis or figure to plot on
- x is an arrayVariable of x values used to plot the regression. If None, then 100 points will be plotted within the range of the data used to create the fit.
- **kwargs are key word arguments for matplotlib.pyplot.plot  

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a list of a single element. That being the matplotlib.lines.Line2D created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


## Scatter residuals
The fit class has a function to scatter the residuals. This is usefull when evaluating the fit

```
fit.scatterResiduals(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, **kwargs) -> matplotlib.collections.PathCollection | None
```

Parameters:
- ax is the axis or figure to plot on
- label is either a bool or a string. If True, the word "Residuals" is printed in the legend
- **kwargs are key word arguments for matplotlib.pyplot.scatter  


Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a matplotlib.collections.PathCollection objected created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


## Scatter normalized residuals
The fit class has a function to scatter the normalized residuals. The fit uses orthogonal distance regression when creating the regression. Here the weights of the data points are scaled with respect to the uncertanty of the data. If some of the data has a larger uncertanty than other, then the method fit.scatterResiduals may results in a plot, where it seems as if the regression does not capture the data. However, using the methods fit.scatterNormalizedResiduals could shown, that the regression has captured the data, if it has deemed some of the datapoints less important than others.

```
fit.scatterNormalizedResiduals(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, **kwargs) -> matplotlib.collections.PathCollection | None
```

Parameters:
- ax is the axis or figure to plot on
- label is either a bool or a string. If True, the word "Normalized residuals" is printed in the legend
- **kwargs are key word arguments for matplotlib.pyplot.scatter  


Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a matplotlib.collections.PathCollection objected created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


## Plot uncertanty of inputs

The fit class has a function to plot the uncertanty area of the inputs. The uncertanty of each input are represented as an ellipse under the hood. The ellipses are combined using tangents. From this a matplotlib.pathces.Polygon is created and plottet to the axes.

```
fit.plotUncertantyOfInputs(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, n = 100: int, **kwargs) -> matplotlib.patches.Polygon | None
```

Parameters:
- ax is the axis or figure to plot on
- n is an integer representing the angle resolution used to plot the uncertanty of the inputs represented as ellipses
- **kwargs are key word arguments for the matplotlib.patches.Polygon 

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a matplotlib.patches.Polygon objected created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.

## Scatter uncertaty as ellipses
The fit class has a function to plot the uncertanty area of the inputs. The uncertanty of each input are represented as an ellipse which is plotted as a matplotlib.lines.Line2D.

```
fit.scatterUncertatyAsEllipses(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, n = 100: int, **kwargs) -> List[matplotlib.lines.Line2D] | None
```

Parameters:
- ax is the axis or figure to plot on
- n is an integer representing the angle resolution used to plot the uncertanty of the inputs represented as ellipses
- **kwargs are key word arguments for the list of matplotlib.lines.Line2D 

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a matplotlib.lines.Line2D objected created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


## Axis labels
A fit object has a function to set the units of the axis. The units are set to the unit of the variables at the instance when the fit object was created. If the units are converted after the fit object was made, then this has no affect on the methods to set the units of the axis.

```
fit.addUnitToLabels(ax)
fit.addUnitToXLabel(ax)
fit.addUnitToYLabel(ax)
```

The unit of the data parsed when initializing the fit object will be appended to the axis labels

## Example
```
import pyees as pe
import matplotlib.pyplot as plt

x = pe.variable([3, 4, 5, 6], 'm', [0.15, 0.3, 0.45, 0.6])
y = pe.variable([10, 20, 30, 40], 'C', [2, 3, 4, 5])

F = pe.lin_fit(x, y)


fig, ax = plt.subplots()
F.scatter(ax)
F.plot(ax)
ax.set_xlabel('Distance')
ax.set_ylabel('Temperature')
F.addUnitToLabels(ax)
ax.legend()
fig.tight_layout()
plt.show()

```

![Fitting example](/docs/examples/fitExample.png)