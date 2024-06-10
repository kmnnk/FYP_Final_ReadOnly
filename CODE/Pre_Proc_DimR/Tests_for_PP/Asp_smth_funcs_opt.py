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
def SAVGOL_smoothing(noisy_spec, int_arr, wd_l, p_order):
    start = time.time()
    smth_data = savgol_filter(noisy_spec, window_length=wd_l, polyorder=p_order)
    end = time.time()

    smth_noise = int_arr - smth_data
    smthed_noise_pwr = np.mean(np.abs(smth_noise) ** 2)
    computation_time = end - start

    return smthed_noise_pwr, computation_time

def WT_smoothing(noisy_spec, int_arr, wavelet, threshold):
    '''
    Inputs:
    - wavelet (Choose a Daubechies wavelet (e.g., 'db2'))
    - threshold
    '''
    start = time.time()
    coeffs = pywt.wavedec(noisy_spec , wavelet)   # Perform wavelet transform
    coeffs_smoothed = [pywt.threshold(c, threshold, mode='soft') for c in coeffs] # smooth by incl/excl coeffs based on threshold
    smth_data = pywt.waverec(coeffs_smoothed, wavelet) # Recreate smth data using the chosen coeffs
    end = time.time()

    smth_noise = int_arr - smth_data
    smthed_noise_pwr = np.mean(np.abs(smth_noise) ** 2)
    computation_time = end - start

    return smthed_noise_pwr, computation_time

def SNR_diff(orig_sig_pwr, gen_noise_pwr, smth_noise_pwr):
    noised_snr = 10 * np.log10(orig_sig_pwr / gen_noise_pwr)
    smoothed_snr = 10 * np.log10(orig_sig_pwr / smth_noise_pwr)
    snr_diff = smoothed_snr - noised_snr
    rel_SNR = (snr_diff / noised_snr) * 100

    return rel_SNR, snr_diff

def plot_SAVGOL(noisy_spec, intensity_arr, true_signal_power, noise_power):
    poly_order_snrs_rel = []
    poly_order_snrs_ab = []
    poly_order_compute_time = []
    for poly_order in range(1, 9):
        smth_noise_snrs_rel = []
        smth_noise_snrs_ab = []
        compute_time = []
        for window_l in range(9, 61, 2):
            if poly_order >= window_l:
                continue
            smth_noise_pwr, comp_time = SAVGOL_smoothing(noisy_spec, intensity_arr, window_l, poly_order)
            rel_snr, ab_snr = SNR_diff(true_signal_power, noise_power, smth_noise_pwr)
            smth_noise_snrs_rel.append(rel_snr)
            smth_noise_snrs_ab.append(ab_snr)
            compute_time.append(comp_time)
        poly_order_snrs_rel.append(smth_noise_snrs_rel)
        poly_order_snrs_ab.append(smth_noise_snrs_ab)
        poly_order_compute_time.append(compute_time)

    #compute the time on average for savgol to run
    time_per_item = []
    for item in poly_order_compute_time:
        avg_item_time = sum(item)/len(item)
        time_per_item.append(avg_item_time)
    avg_time_overall = sum(time_per_item)/len(time_per_item)
    print("Computation time savgol: ", avg_time_overall)

    #identify the biggest rel SNR and ab SNR
    maxrelSNR_per_item = []
    for item in poly_order_snrs_rel:
        maxSNR_item = max(item)
        maxrelSNR_per_item.append(maxSNR_item)
    max_rel_SNR_overall = max(maxrelSNR_per_item)
    print("Max rel SNR Savgol: ", max_rel_SNR_overall)

    maxabSNR_per_item = []
    for item in poly_order_snrs_ab:
        maxSNR_item = max(item)
        maxabSNR_per_item.append(maxSNR_item)
    max_ab_SNR_overall = max(maxabSNR_per_item)
    print("Max ab SNR Savgol: ", max_ab_SNR_overall)

    '''
    #create plots
    fig1, (ax1, ax2) = plt.subplots(2, 1)
    fig1.suptitle('Savgol Filter Smoothing')
    legend_labels = ["1", "2", "3", "4", "5", "6", "7", "8"]

    for index in range(len(poly_order_snrs_rel)):
        ax1.plot(range(9, 105, 2), poly_order_snrs_rel[index], label=legend_labels[index])
    ax1.set_title('Relative SNR change')
    ax1.set_ylabel('SNR % change')
    ax1.legend()

    for index1 in range(len(poly_order_snrs_ab)):
        ax2.plot(range(9, 105, 2), poly_order_snrs_ab[index1], label=legend_labels[index1])
    ax2.set_title('Absolute SNR change')
    ax2.set_xlabel('Window length')
    ax2.set_ylabel('SNR change')
    ax2.legend()

    plt.show()
    '''
    

def plot_WT(noisy_spec, intensity_arr, true_signal_power, noise_power):
    poly_order_snrs_rel = []
    poly_order_snrs_ab = []
    poly_order_compute_time = []
    for wavelet in range(1, 25):
        wavelet = "db{}".format(wavelet)
        smth_noise_snrs_rel = []
        smth_noise_snrs_ab = []
        compute_time = []
        for threshold in range(1, 10001, 500): # Have to figure out range initially by looking at effects on a plot
            smth_noise_pwr, comp_time = WT_smoothing(noisy_spec, intensity_arr, wavelet, threshold)
            rel_snr, ab_snr = SNR_diff(true_signal_power, noise_power, smth_noise_pwr)
            smth_noise_snrs_rel.append(rel_snr)
            smth_noise_snrs_ab.append(ab_snr)
            compute_time.append(comp_time)
        poly_order_snrs_rel.append(smth_noise_snrs_rel)
        poly_order_snrs_ab.append(smth_noise_snrs_ab)
        poly_order_compute_time.append(compute_time)

    #compute the time on average for WT to run
    time_per_item = []
    for item in poly_order_compute_time:
        avg_item_time = sum(item) / len(item)
        time_per_item.append(avg_item_time)
    avg_time_overall = sum(time_per_item) / len(time_per_item)
    print("Computation time WT: ", avg_time_overall)

    # identify the biggest rel SNR and ab SNR
    maxrelSNR_per_item = []
    for item in poly_order_snrs_rel:
        maxSNR_item = max(item)
        maxrelSNR_per_item.append(maxSNR_item)
    max_rel_SNR_overall = max(maxrelSNR_per_item)
    print("Max rel SNR WT: ", max_rel_SNR_overall)

    maxabSNR_per_item = []
    for item in poly_order_snrs_ab:
        maxSNR_item = max(item)
        maxabSNR_per_item.append(maxSNR_item)
    max_ab_SNR_overall = max(maxabSNR_per_item)
    print("Max ab SNR WT: ", max_ab_SNR_overall)

    '''
    #create plots
    fig1, (ax1, ax2) = plt.subplots(2, 1)
    fig1.suptitle('Wavelet Transform Filter Smoothing')

    for inner_rel_list in poly_order_snrs_rel:
        ax1.plot(range(1, 10001, 500), inner_rel_list)
    ax1.set_title('Relative SNR change')
    ax1.set_ylabel('SNR % change')
    ax1.legend()

    for inner_ab_list in poly_order_snrs_ab:
        ax2.plot(range(1, 10001, 500), inner_ab_list)
    ax2.set_title('Absolute SNR change')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('SNR change')
    ax2.legend()

    plt.show()
    '''







