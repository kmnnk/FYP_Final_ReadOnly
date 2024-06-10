import h5py
import matplotlib.pyplot as plt
# import scipy.io
import os
from PIL import Image
import cv2
import numpy as np
from numpy import asarray
from scipy.signal import savgol_filter
from scipy.signal import detrend
# from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
# from scipy.io import loadmat
# from scipy.ndimage import gaussian_filter1d
# from scipy.ndimage import median_filter
import re
import csv
import pandas as pd

def save_spectral_data(array_spec_data, filename, file_p):
    # Ensure the directory exists
    os.makedirs(file_p, exist_ok=True)

    # Save the NumPy array as a .npy file
    np.save(os.path.join(file_p, filename), array_spec_data)

def rd_MATims_store_bcc_and_spectra(file_path, list_all_images, list_all_spectra, list_xc):
    # Open the MATLAB v7.3 file
    try: 
        with h5py.File(file_path, 'r') as file:
            data_bcc = np.array(file['bcc'], dtype=np.uint8)
            list_all_images.append(data_bcc)
            
            x_c = file['x_c'][:] #x_c is the wavenumbers collected in a specific spectrum (so on the x axis)
            list_xc.append(x_c) #appended to list in order of the files looked at so should match the order of the bcc and spectra arrays

            variables = list(file.keys())
            pattern = r'map\w*'
            matches = [name for name in variables if re.match(pattern, name)]

            correct_spec = []
            for var in matches:
                if re.match('^[mapt0-9_]*$', var) and file[var].shape[-2:] == data_bcc.shape[-2:]:  # Check if the first two dimensions match
                    correct_spec.append(var)
            
            print(correct_spec)

            for var in correct_spec:
                list_all_spectra.append(file[var][()])

            # print(file[var].shape[:2])
            # print(data_bcc.shape[:2])

    except Exception as e:
        print(f"An error occurred with file: {file_path}")
        print(f"Error details: {e}")
    
    # return matches, variables, correct_spec


def rd_ims_from_folder(folder_path):
    list_all_bcc_images = []
    list_all_spectra = []
    list_xc = []
    filenames = []  # to know which order they are being inputted into the bcc and spectra arrays - if need to check

    # List all files in the specified folder
    for filename in os.listdir(folder_path):
        filenames.append(filename)
        if filename.endswith('.mat'):  # Check if the file is a MATLAB file
            file_path = os.path.join(folder_path, filenames[-1])
            # matches, variables, correct_spec = rd_MATims_store_bcc_and_spectra(file_path, list_all_bcc_images, list_all_spectra)
            rd_MATims_store_bcc_and_spectra(file_path, list_all_bcc_images, list_all_spectra, list_xc)

    # Convert the list of lists to a NumPy array
    # maximum x and y dimensions of the image list
    max_x = 200
    max_y = 200

    # Create an empty array to store the images
    array_spec_data = np.empty((len(list_all_spectra), 1024, max_x, max_y))
    array_bcc_data = np.empty((len(list_all_bcc_images), max_x, max_y))

    # Iterate over the image list and assign each image to the big array
    for i, image in enumerate(list_all_spectra):
        x, y = image.shape[1], image.shape[2]
        array_spec_data[i, :, :x, :y] = image[:, :min(x, max_x), :min(y, max_y)]

    for i, b_image in enumerate(list_all_bcc_images):
        x, y = b_image.shape
        array_bcc_data[i, :min(x, max_x), :min(y, max_y)] = b_image[:min(x, max_x), :min(y, max_y)]

    return array_bcc_data, array_spec_data, list_xc, filenames

folder_path = '/rds/general/user/nmk120/home/RawD/Batch1'
# bcc, spec, matches, variables, correct_spec = rd_ims_from_folder(folder_path) #if using this make sure to  put extra vars back into the return statement above
bcc, spec, xc, filenames = rd_ims_from_folder(folder_path)

# directory_smth = '/rds/general/user/nmk120/home/RawD_savednpy'
# save_spectral_data(spec, 'raw1.npy', directory_smth)


print('Batch1_shape_bcc_imported:', bcc.shape)
print('Batch1_shape_spec_imported:', spec.shape)
print(len(spec))
print(len(bcc))


#FILTER DEAD PIX FROM BCC DATA
def filter_dead_pix_bcc(array_bcc_data):
    array_bcc_data = array_bcc_data.astype(np.float32)
    # filtered_im = []
    blurred_img_array = np.empty_like(array_bcc_data)

    for im in range(array_bcc_data.shape[0]):
        im_med1 = cv2.medianBlur(array_bcc_data[im], 3) 
        im_med2 = cv2.medianBlur(im_med1, 3)
        im_med3 = cv2.medianBlur(im_med2, 3)

        blurred_img_array[im] = im_med3
    
    return blurred_img_array
    
array_filtered_im = filter_dead_pix_bcc(bcc)
print('Batch1_shape_arraybbc_filtered:', array_filtered_im.shape)

#SAVING FILTERED BCC DATA
#saving smoothed bcc ground truth data 
def save_bccimages_as_npy(array_filtered_im, file_p):
    # make sure directory exists
    os.makedirs(file_p, exist_ok=True)

    # file name
    filename = 'all_bcc_images_second.npy'

    # Save the NumPy array as a .npy file
    np.save(os.path.join(file_p, filename), array_filtered_im)

directory_bcc = '/rds/general/user/nmk120/home/BCC1'
save_bccimages_as_npy(array_filtered_im, directory_bcc) 


#1 - SMOOTHING
def smooth_spectra(spectra, wl, polyO):
    num_images, num_rows, num_cols, wavelengths = spectra.shape
    
    # Reshape the spectra array for efficient processing
    reshaped_spectra = spectra.reshape((num_images * num_rows * num_cols, wavelengths))

    # Apply savgol_filter to the entire array
    smoothed_spectra = savgol_filter(reshaped_spectra, window_length=wl, polyorder=polyO, axis=1)

    # Reshape the smoothed array back to the original shape
    array_smoothed_spectra = smoothed_spectra.reshape((num_images, num_rows, num_cols, wavelengths))

    return array_smoothed_spectra
    
smoothed_sg1 = smooth_spectra(spec, 23, 2)
print('Batch1_shape_spec_smth:', smoothed_sg1.shape)
directory_smth = '/rds/general/user/nmk120/home/wl23/FullPP1'
save_spectral_data(smoothed_sg1, 'smth1.npy', directory_smth)

#2 - DETRENDING
def detrending(spectra):
    num_images, wavelengths, num_rows, num_cols = spectra.shape
    detrended_spectra = []

    for image in range(num_images):
        detrended_image = []
        for row in range(num_rows):
            detrended_row = []
            for col in range(num_cols):
                detrended_spec = detrend(spectra[image, :, row, col]) # said unexpected keyword 'type=linear' so took out as the default is linear anyway;for arrays, but cant use as the conversion of whole list to an array above didnt work, so now working with a list of arrays
                detrended_row.append(np.array(detrended_spec))
 
            detrended_image.append(np.array(detrended_row))

        detrended_spectra.append(np.array(detrended_image))

   
    # Make the detrended_spectra list into an array
    max_x = max(image.shape[0] for image in detrended_spectra)
    max_y = max(image.shape[1] for image in detrended_spectra)

    # Create an empty array to store the images
    array_specdetr = np.empty((len(detrended_spectra), max_x, max_y, 1024))

    # Iterate over the image list and assign each image to the big array
    for i, image in enumerate(detrended_spectra):
        x, y = image.shape[0], image.shape[1]
        array_specdetr[i, :x, :y, :] = image
    
    return array_specdetr

detrended_spec = detrending(smoothed_sg1)

print('Batch1_shape_smth_detr:', detrended_spec.shape)
directory_smth = '/rds/general/user/nmk120/home/wl23/FullPP1'
save_spectral_data(detrended_spec, 'smth_detr.npy', directory_smth)


#3 - BASELINE CORRECTION
def ALSbaseline_correction (spectra, smoothness, asymm, niter=6):
    num_images, num_rows, num_cols, wavelengths = spectra.shape
    baselined_spectra = []

    L = wavelengths
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)

    for image in range(num_images):
        im_basel =[]
        for row in range(num_rows):
            row_basel = []
            for col in range(num_cols):
                current_spec = spectra[image, row, col, :]

                for i in range(niter):
                    
                    W = sparse.spdiags(w, 0, L, L)
                    Z = W + smoothness * D.dot(D.transpose())
                    est_baseline = spsolve(Z, w*current_spec)
                    w = asymm * (current_spec > est_baseline) + (1-asymm) * (current_spec < est_baseline)

                baselined_spec = current_spec - est_baseline
                row_basel.append(baselined_spec)
            im_basel.append(row_basel)
        baselined_spectra.append(im_basel)

    array_baselined_spectra = np.array(baselined_spectra, dtype=object)

    return array_baselined_spectra

baselined_spec = ALSbaseline_correction(detrended_spec, 100000, 0.05)
print('Batch1_shape_smth_detr_basel:', baselined_spec.shape)
directory_smth = '/rds/general/user/nmk120/home/wl23/FullPP1'
save_spectral_data(baselined_spec, 'smth_detr_basel.npy', directory_smth)

#NORMALISATION
def normalisation(spectra, epsilon = 1e-10):
    
    min_vals = np.min(spectra, axis=3, keepdims=True)
    max_vals = np.max(spectra, axis=3, keepdims=True)
    
    scaled_data = np.divide(np.subtract(spectra, min_vals), max_vals - min_vals + epsilon) 
    
    return scaled_data

normalised = normalisation(baselined_spec)
print('Batch1_shape_smth_detr_basel_norm:', normalised.shape)
directory_smth = '/rds/general/user/nmk120/home/wl23/FullPP1'
save_spectral_data(normalised, 'normalised_fullPP.npy', directory_smth)

# smth_2 = smooth_spectra(normalised, 23, 2)
# print('Batch1_shape_smth_detr_basel_norm_smth2:', smth_2.shape)
# directory_smth = '/rds/general/user/nmk120/home/wl23/FullPP1'
# save_spectral_data(smth_2, 'normalised_smth2fullPP.npy', directory_smth)
# print('done')
