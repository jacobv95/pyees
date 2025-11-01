# Multi variable fitting

The package includes a tool to produce fits. The following fits are known
 - dummy fit            f(X) = 1
 - Linear fit           f(X) = a_0 + sum_i^n (a_i * X_i) 


The dummy fit will always return a constant function with a value of 1. This "fit" is used to easily plot data.

```
F_dummy = dummy_fit(x: list[variable], y: variable)
F_lin = lin_fit(x: list[variable], y: variable, p0: list[float] | None = None, useParameters : list[bool] | None = None)
```
 - x is the x data used to generate the regression
 - y is the y data used to generate the regression
 - p0 is the initial guess of the regression coefficients. The coefficients will be initialized to 0 if p0 is set to None
 - useParameters defines which coefficients to tune. If a coefficient is not tuned, then it will remain as the initial guesses.

## Priting
The fit can be printed. This is done in latex format. First the model is printed, and then the coefficients are printed.

```
print(fit) -> str
```

## Predict
Once a fit has been made this can be used to create a prediction. 

```
fit.predict(x: list[variable]) -> variable
```



## Plot in plane
The fit class has a function to plot

```
fit.plot(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, index: int, x: list[variable] = None, **kwargs) -> List[matplotlib.lines.Line2D] | None
```

Parameters:
- ax is the axis or figure to plot on
- index is the index of the input variables, which will not be held constant. If x=None, then all variables will be set to the average value of the supplied data during fitting. If x!=None, then all elements of x has to be a scalar variable but x[index], which has to be an array variable.
- x is an arrayVariable of x values used to plot the regression. If None, then 100 points will be plotted within the range of the data used to create the fit.
- **kwargs are key word arguments for matplotlib.pyplot.plot or plotly.graph_objects.Scatter

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a list of a single element. That being the matplotlib.lines.Line2D created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


## Scatter in plane
The fit class has a function to scatter the data used to generate the fit.

```
fit.scatter(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, index: int, showUncert: bool = True, **kwargs) -> matplotlib.collections.PathCollection | None
```

Parameters:
- ax is the axis or figure to plot on
- index is the index of the input variables which will be used as the x-variable on the axis when scattering the data.
- showUncert is a bool. Errorbars are shown if true
- **kwargs are key word arguments for matplotlib.pyplot.scatter  

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a matplotlib.collections.PathCollection objected created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.

## Plot data in plane
The fit class has a function to plot the data used to generate the fit.

```
fit.plotData(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, index: int, **kwargs) -> List[matplotlib.lines.Line2D] | None
```

Parameters:
- ax is the axis or figure to plot on
- index is the index of the input variables which will be used as the x-variable on the axis when plotting the data.
- **kwargs are key word arguments for matplotlib.pyplot.plot  

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a list of a single element. That being the matplotlib.lines.Line2D created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


## Plot uncertanty in plane
The fit class has a function to plot the uncertanty bands of the regression

```
fit.plotUncertanty(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, index: int, x: list[variable] = None, **kwargs) -> List[matplotlib.lines.Line2D] | None
```

Parameters:
- ax is the axis or figure to plot on
- index is the index of the input variables, which will not be held constant. If x=None, then all variables will be set to the average value of the supplied data during fitting. If x!=None, then all elements of x has to be a scalar variable but x[index], which has to be an array variable.
- x is an arrayVariable of x values used to plot the regression. If None, then 100 points will be plotted within the range of the data used to create the fit.
- **kwargs are key word arguments for matplotlib.pyplot.plot  

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a list of a single element. That being the matplotlib.lines.Line2D created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


## Scatter residuals in plane
The fit class has a function to scatter the residuals. This is usefull when evaluating the fit

```
fit.scatterResiduals(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, index: int, **kwargs) -> matplotlib.collections.PathCollection | None
```

Parameters:
- ax is the axis or figure to plot on
- index is the index of the input variables which will be used as the x-variable on the axis when scattering the residuals.
- label is either a bool or a string. If True, the word "Residuals" is printed in the legend
- **kwargs are key word arguments for matplotlib.pyplot.scatter  


Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a matplotlib.collections.PathCollection objected created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


## Scatter normalized residuals in plane
The fit class has a function to scatter the normalized residuals. The fit uses orthogonal distance regression when creating the regression. Here the weights of the data points are scaled with respect to the uncertanty of the data. If some of the data has a larger uncertanty than other, then the method fit.scatterResiduals may results in a plot, where it seems as if the regression does not capture the data. However, using the methods fit.scatterNormalizedResiduals could shown, that the regression has captured the data, if it has deemed some of the datapoints less important than others.

```
fit.scatterNormalizedResiduals(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, index: int, **kwargs) -> matplotlib.collections.PathCollection | None
```

Parameters:
- ax is the axis or figure to plot on
- index is the index of the input variables which will be used as the x-variable on the axis when scattering the normalized residuals.
- label is either a bool or a string. If True, the word "Normalized residuals" is printed in the legend
- **kwargs are key word arguments for matplotlib.pyplot.scatter  


Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a matplotlib.collections.PathCollection objected created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


## Plot uncertanty of inputs in plane

The fit class has a function to plot the uncertanty area of the inputs. The uncertanty of each input are represented as an ellipse under the hood. The ellipses are combined using tangents. From this a matplotlib.pathces.Polygon is created and plottet to the axes.

```
fit.plotUncertantyOfInputs(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, index: int, n = 100: int, **kwargs) -> matplotlib.patches.Polygon | None
```

Parameters:
- ax is the axis or figure to plot on
- index is the index of the input variables which will be used as the x-variable on the axis when plotting the uncertanty of the inputs.
- n is an integer representing the angle resolution used to plot the uncertanty of the inputs represented as ellipses
- **kwargs are key word arguments for the matplotlib.patches.Polygon 

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a matplotlib.patches.Polygon objected created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.

## Scatter uncertaty as ellipses
The fit class has a function to plot the uncertanty area of the inputs. The uncertanty of each input are represented as an ellipse which is plotted as a matplotlib.lines.Line2D.

```
fit.scatterUncertatyAsEllipses(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, index: int, n = 100: int, **kwargs) -> List[matplotlib.lines.Line2D] | None
```

Parameters:
- ax is the axis or figure to plot on
- index is the index of the input variables which will be used as the x-variable on the axis when scattering the uncertanty as ellipses.
- n is an integer representing the angle resolution used to plot the uncertanty of the inputs represented as ellipses
- **kwargs are key word arguments for the list of matplotlib.lines.Line2D 

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a matplotlib.lines.Line2D objected created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.



## Axis labels
A fit object has a function to set the units of the axis. The units are set to the unit of the variables at the instance when the fit object was created. If the units are converted after the fit object was made, then this has no affect on the methods to set the units of the axis.

```
fit.addUnitToLabels3D(ax, **kwargs)
fit.addUnitToXLabel3D(ax, **kwargs)
fit.addUnitToYLabel3D(ax, **kwargs)
fig.addUnitToLabels3D(ax, **kwargs)

fit.addUnitToXLabel(ax, **kwargs)
fit.addUnitToYLabel(ax, **kwargs)
fit.addUnitToLabels(ax, **kwargs)

```

The unit of the data parsed when initializing the fit object will be appended to the axis labels




## Plot in 3d

A multi variable fit object has a function to plot the regression in 3d

```
fit.plot3D(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, **kwargs) -> matplotlib.lines.Line2D | None
```

Parameters:
- ax is the axis or figure to plot on
- **kwargs are key word arguments for matplotlib.pyplot.plot  

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a list of a single element. That being the matplotlib.lines.Line2D created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


 ## Scatter in 3d

A multi variable fit object has a function to scatter the data in 3d

```
fit.scatter3D(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, **kwargs) -> matplotlib.lines.Line2D | None
```

Parameters:
- ax is the axis or figure to scatter on
- **kwargs are key word arguments for matplotlib.pyplot.plot  

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a list of a single element. That being the matplotlib.lines.Line2D created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.


 ## Plot residuals in 3d

A multi variable fit object has a function to plot the residuals in 3D

```
fit.plotResiduals3D(ax: matplotlib.axes.Axes | plotly.graph_objects.Figure, **kwargs) -> matplotlib.lines.Line2D | None
```

Parameters:
- ax is the axis or figure to plot the residuals on
- **kwargs are key word arguments for matplotlib.pyplot.plot  

Returns:
 - If the type of the argument 'ax' is matplotlib.axes.Axes then the method returns a list of a single element. That being the matplotlib.lines.Line2D created on the axis. If the type of the argument 'ax' is plotly.graph_objects.Figure then the method returns None.