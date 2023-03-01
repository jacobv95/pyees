import numpy as np
import scipy.odr as odr


def lin(B, x):
    b = B[0]
    return b + 0 * x


def odrWrapper(description, x, y, sx, sy):
    data = odr.RealData(x, y, sx, sy)
    regression = odr.ODR(data, odr.Model(lin), beta0=[1])
    regression = regression.run()
    popt = regression.beta
    cov_beta = np.sqrt(np.diag(regression.cov_beta))
    sd_beta = regression.sd_beta
    print(description, popt, sd_beta, cov_beta)


# constants
b = 50
n = 10000
noiseScale = 10
uncert = 1
np.random.seed(0)

# no noise no uncertanty
x = np.linspace(0, 100, n)
y = np.ones(n) * b
sx = [1e-10] * n    # very smalle value as the uncertanty can not be zero
sy = [1e-10] * n    # very smalle value as the uncertanty can not be zero
odrWrapper('No noise no uncertanty:  ', x, y, sx, sy)

# noise but no uncertanty
x = np.linspace(0, 100, n)
y = np.ones(n) * b
y += noiseScale * (2 * np.random.rand(n) - 1)
sx = [1e-10] * n
sy = [1e-10] * n
odrWrapper('Noise but no uncertanty: ', x, y, sx, sy)


# no noise but uncertanty
x = np.linspace(0, 100, n)
y = np.ones(n) * b
sx = [1e-10] * n
sy = [uncert] * n
odrWrapper('No noise but uncertanty: ', x, y, sx, sy)

# noise and uncertanty
x = np.linspace(0, 100, n)
y = np.ones(n) * b
y += noiseScale * (2 * np.random.rand(n) - 1)
sx = [1e-10] * n
sy = [1] * n
odrWrapper('Noise and uncertanty:    ', x, y, sx, sy)
