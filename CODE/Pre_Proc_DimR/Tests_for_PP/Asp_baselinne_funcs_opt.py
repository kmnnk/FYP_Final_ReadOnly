import h5py
import matplotlib.pyplot as plt
import numpy
import scipy.io
import os
from PIL import Image
import numpy as np
from numpy import asarray
from scipy.signal import savgol_filter
from scipy.signal import detrend
import pandas as pd
from scipy.optimize import curve_fit
# from statsmodels.tsa.tsatools import detrend
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pywt
from math import log10, sqrt
import time
import pysptools
from pysptools import distance
import math
import pandas as pd
import openpyxl

#*********************************************************************************************************************

#BASELINE CORRECTION WITH POLYNOMIAL FITTING
def polym_fitting_baseline(deg, x_wavelengths, spectra_y):
    x = numpy.concatenate(x_wavelengths)
    coefficients = np.polyfit(x, spectra_y, deg)
    baseline = np.polyval(coefficients, x)
    corrected_data = spectra_y - baseline

    plt.plot(x_wavelengths, corrected_data, lw = 0.5)
    plt.show()


def als_baseline(y, smoothness, asymm, niter=100):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + smoothness * D.dot(D.transpose())
        est_baseline = spsolve(Z, w*y)
        w = asymm * (y > est_baseline) + (1-asymm) * (y < est_baseline)

    baselined_spec = y - est_baseline
    return baselined_spec

'''
# Estimation of the baseline:
estimated_baselined = als_baseline(map_t3[:,100,100], l, p)
#estimated_baselined1 = baseline_als(detrended_2, l, p)

# Baseline subtraction:
baselined_spectrum = map_t3[:,100,100] - estimated_baselined
#baselined_spectrum1 = detrended_2 - estimated_baselined1

# How does it look like?
# plt.plot(x_c, map_t3[:,100,100], lw =0.5)
plt.plot(x_c, baselined_spectrum, lw = 0.5) #produces similar baseline to detrend fucntion...but the peaks are stronger and smaller peaks less pronouced
plt.show()
'''
