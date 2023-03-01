# pyees
Python package to perform data processing with uncertanties.

## Example
We would like to calculate <img src="https://render.githubusercontent.com/render/math?math=C=A\cdot B"> given to measurements <img src="https://render.githubusercontent.com/render/math?math=A=12.3"> and <img src="https://render.githubusercontent.com/render/math?math=B=35.1"> both with uncertanties <img src="https://render.githubusercontent.com/render/math?math=\sigma_A=2.6"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma_B=8.9">. The value of <img src="https://render.githubusercontent.com/render/math?math=C"> is simply computed as <img src="https://render.githubusercontent.com/render/math?math=C=12.3\cdot 35.1 = 431.73">. The uncertanty of <img src="https://render.githubusercontent.com/render/math?math=C"> is determined using the following equation

<img src="https://render.githubusercontent.com/render/math?math=\sigma_C = \sqrt{  \left(\frac{\partial C}{\partial A} \sigma_A\right)^2 %2B \left(\frac{\partial C}{\partial B} \sigma_B\right)^2 %2B 2\frac{\partial C}{\partial A}\frac{\partial C}{\partial B}\sigma_{AB}}">

Here <img src="https://render.githubusercontent.com/render/math?math=\sigma_{AB}"> is the covariance of the measurements of <img src="https://render.githubusercontent.com/render/math?math=A"> and <img src="https://render.githubusercontent.com/render/math?math=B">. We will assumed that this is zero for now. The equation above is evaluated as follows:

<img src="https://render.githubusercontent.com/render/math?math=\sigma_C = \sqrt{  \left(B \sigma_A\right)^2 %2B \left(A\sigma_B\right)^2 } = \sqrt{  \left(35.1 \cdot 2.6\right)^2 %2B \left(12.3 \cdot 8.9\right)^2 } = \sqrt{(91.26)^2 %2B (109.47)^2}=142.52">

## Scope of this package
These computations quickly becomes very difficult for more complicated equations than the one used in this example. This packages is designed to easily perform such computations. Furthermore a few features is added to the packages
 - Read data with uncertanty from an .xls or .xlsx file
 - Print measurements with the correct number of significant digits based on the uncertanty
 - Plot data with errorbars
 - Perform regression where the regression constants are affected by the uncertanty of the data

## Documentation
The documentation is split in to 5 parts:
 - 1 variables
 - 2 Constants
 - 3 Importing data
 - 4 Fitting
 - 5 Prop

See the folder "docs" for the documentation

# How to install
Just run ```pip install dataUncert```