
# Fitting
The package includes a tool to produce fits. The following fits are known
 - dummy fit
 - Linear fit
 - Polynomial fit
 - Power fit
 - Exponential fit
 - Logistic fit
 - Logistic fit with a fixed maximum value of 100

The dummy fit will always return a constant function with a value of 1. This fit is primarily used to easily plot the data.

```
F_dummy = dummy_fit(x: variable, y: variable)
F_lin = lin_fit(x: variable, y: variable, p0 = None: list | None)
F_pol = pol_fit(x: variable, y: variable, deg: int = 2, terms = None: list | Nonep0 = None: list | None)
F_pow = pow_fit(x: variable, y: variable, p0 = None: list | None)
F_exp = exp_fit(x: variable, y: variable, p0 = None: list | None)
F_logistic = logistic_fit(x: variable, y: variable, p0 = None: list | None)
F_logistic_100 = logistic_100_fit(x: variable, y: variable, p0 = None: list | None)
```

 - x is the x data used to generate the regression
 - y is the y data used to generate the regression
 - p0 is the initial guess of the regression coefficients. The coefficients will be initialized to 1 if p0 is set to None
 - deg is the degree of the polynomial
 - terms defined the terms of the polynomial to use. if terms is None, then all terms are used. Each element in the list has to be a boolean. The terms are ordered from highest to lowest polynomial degree

## Priting
The fit can be printed. This is done in latex format. First the model is printed, and then the coefficients are printed.

```
print(pol_fit) -> str
```

## Predict
Once a fit has been made this can be used to create a prediction. The input can either be a float or a variable. If the float is a variable it is assued to have the same unit as the x-data used to generate the fit-object.

```
pol_fit.predict(x: float | variable) -> variable
```

## Predict differential
Once a fit has been made this can be used to create a prediction of the differential. The input can either be a float or a variable. If the float is a variable it is assued to have the same unit as the x-data used to generate the fit-object.

```
pol_fit.predictDifferential(x: float | variable) -> variable
```



## Plot
The fit class has a function to plot

```
fit.plot(ax, label=True, x=None, **kwargs)
```

- ax is the axis object from matplotlib
- label is either a bool or a string. If True, the regression function will be written in the legend
- x is a numpy array of x values used to plot the regression. If None, then 100 points will be plotted
- **kwargs are key word arguments for matplotlib.pyplot.plot  

### Plot differential
The fit class has a function to plot the differential

```
fit.plotDifferential(ax, label=True, x=None, **kwargs)
```

- ax is the axis object from matplotlib
- label is either a bool or a string. If True, the regression function will be written in the legend
- x is a numpy array of x values used to plot the regression. If None, then 100 points will be plotted
- **kwargs are key word arguments for matplotlib.pyplot.plot  

### Scatter
The fit class has a function to scatter the data used to generate the fit.

```
fit.scatter(ax, label=True, showUncert=True, **kwargs)
```

- ax is the axis object from matplotlib
- label is either a bool or a string. If True, the word "Data" is printed in the legend
- showUncert is a bool. Errorbars are shown if true
- **kwargs are key word arguments for matplotlib.pyplot.scatter  

### plotData
The fit class has a function to plot the data used to generate the fit.

```
fit.plotData(ax, label=True, **kwargs)
```

- ax is the axis object from matplotlib
- label is either a bool or a string. If True, the word "Data" is printed in the legend
- **kwargs are key word arguments for matplotlib.pyplot.scatter  

### Axis labels
The fit class has a function to set the units of the axis

```
F = lin_fit(x: variable,y: variable)
F.addUnitToLabels(ax)
F.addUnitToXLabel(ax)
F.addUnitToYLabel(ax)
```

The unit of the data parsed when initializing the fit object will be appended to the axis labels

### Example
```
from dataUncert import *
import matplotlib.pyplot as plt

x = variable([3, 4, 5, 6], 'm', [0.15, 0.3, 0.45, 0.6])
y = variable([10, 20, 30, 40], 'C', [2, 3, 4, 5])

F = lin_fit(x, y)


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