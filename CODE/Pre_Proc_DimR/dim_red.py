from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# import lda
import numpy as np
import matplotlib.pyplot as plt


#load data
data = np.load('/rds/general/user/nmk120/home/wl23/combined_spec_array.npy', allow_pickle=True)
# labels = np.load('/rds/general/user/nmk120/home/BCC_all/combined_bcc_array.npy', allow_pickle=True)

# # Assuming `data` is a 4D array with shape (58, 200, 200, 1024)
# # and `labels` is a 3D array with shape (58, 200, 200)

# # Reshape data and labels into 2D arrays
reshaped_data = data.reshape((-1, data.shape[3])) #(23520000, 1024)
# reshaped_labels = labels.reshape(-1) #(232520000,)


# ########## LDA ############################
# # Initialize the LDA model
# lda = LDA(n_components=1)  # adjust number of components
# # lda = lda.LDA(n_components=1)

# # Fit the LDA model to your data and labels
# lda.fit(reshaped_data, reshaped_labels)
# print(lda.classes_)

# # Transform your data using the fitted LDA model
# transformed_data = lda.transform(reshaped_data)
# # transformed_data = lda.doc_topic_(reshaped_data)

# # Reshape the transformed data back to the original shape
# reshaped_transformed_data = transformed_data.reshape((58, 200, 200, lda.n_components))

# np.save('/rds/general/user/nmk120/home/wl23/LDAdimred_alldata.npy', reshaped_transformed_data)
#############################################


######### PCA ################################
# from sklearn.decomposition import PCA

# pca = PCA()
# pca.fit(reshaped_data)

# cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
# n_components = np.where(cumulative_variance_ratio >= 0.93)[0][0] + 1

# # Calculate the differences between consecutive elements
# diff = np.diff(np.cumsum(pca.explained_variance_ratio_))
# # Define a threshold - might need adjusting
# threshold = 0.0007
# # Find point where the difference drops below the threshold
# elbow_point = np.where(diff < threshold)[0][0] + 1
# print('Elbow Point:', elbow_point)
# print('Cumulative Variance at Elbow Point:', cumulative_variance_ratio[elbow_point])

# # Plot the explained variance ratio
# plt.figure(figsize=(10, 7))
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# # Plot the elbow point
# plt.scatter(elbow_point, cumulative_variance_ratio[elbow_point], color='red')
# plt.text(elbow_point, cumulative_variance_ratio[elbow_point], f'  Elbow Point: {elbow_point}, Variance: {cumulative_variance_ratio[elbow_point]:.2f}')

# plt.savefig('/rds/general/user/nmk120/home/Dim_Red/PCAvariance.png')


# Initialize the PCA model
pca = PCA(n_components=25)  # adjust number of components

# Fit the PCA model to your data
pca.fit(reshaped_data)

# Transform your data using the fitted PCA model
transformed_data = pca.transform(reshaped_data)
reshaped_transformed_data = transformed_data.reshape((58, 200, 200, pca.n_components))


np.save('/rds/general/user/nmk120/home/wl23/PCAdimred_alldata.npy', reshaped_transformed_data)

##############################################

############# PCA analysis ##################
# from sklearn.decomposition import PCA

# pca = PCA(n_components=25)
# X_pca = pca.fit_transform(reshaped_data)

# # The explained variance ratio is stored in the explained_variance_ratio_ attribute
# explained_variance_ratio = pca.explained_variance_ratio_

# # You can print it out
# for i, exp_var in enumerate(explained_variance_ratio):
#     print(f"PC{i+1}: {exp_var}")

# # Open a file in write mode
# with open('/rds/general/user/nmk120/home/Dim_Red/explained_variance_ratio.txt', 'w') as f:
#     # Write the explained variance ratio of each PC to the file
#     for i, exp_var in enumerate(explained_variance_ratio):
#         f.write(f"PC{i+1}: {exp_var}\n")