import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,  roc_curve, auc

def focal_loss(alpha, gamma):
    alpha = tf.constant(alpha, dtype=tf.float32)
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = tf.squeeze(tf.gather(alpha, tf.cast(y_true, tf.int32)))
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_mean(alpha_t * tf.pow(1. - pt_1, gamma) * tf.math.log(tf.clip_by_value(pt_1, 1e-8, 1.0))) \
               -tf.reduce_mean((1 - alpha_t) * tf.pow(pt_0, gamma) * tf.math.log(tf.clip_by_value(1. - pt_0, 1e-8, 1.0)))
    return focal_loss_fixed

# Load the model
model = load_model('/rds/general/user/nmk120/home/Mod_HypP_3D/64_aug/model_trial_35_fold_5.h5', custom_objects={'focal_loss_fixed': focal_loss(alpha=[0.35855980062288656, 0.892506805895908], gamma=3.797118419588512)})
# model = load_model('/rds/general/user/nmk120/home/Mod_HypP/LDA1/model_trial_339_fold_3_act_400t.h5', custom_objects={'focal_loss_fixed': focal_loss(alpha=[0.8495532908555112, 0.8251531184182448], gamma=2.441217612119302)})


# Load the unseen data
# data = np.load('/rds/general/user/nmk120/home/wl23/LDAdimred_alldata.npy', allow_pickle=True)
# data = np.load('/rds/general/user/nmk120/home/wl23/combined_spec_array.npy', allow_pickle=True)
# labels = np.load('/rds/general/user/nmk120/home/BCC_all/combined_bcc_array.npy', allow_pickle=True)

data = np.load('/rds/general/user/nmk120/home/wl23/all_spec_data_aug_rot.npy', allow_pickle=True)
labels = np.load('/rds/general/user/nmk120/home/wl23/all_spec_labels_aug_rot.npy', allow_pickle=True)


X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels, test_size=0.1, random_state=42) #split 10% fo test to kep some totally unseen until the best model is made, the remaining 90% can be sent to be used in the kfold (small test size, to allow the most as poss to be sent to kfold)
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')


# # Make predictions
y_pred = model.predict(X_test)

# Plotting 
fig, axs = plt.subplots(2, 3, figsize=(12, 12))

for i in range(6):
    ax = axs[i//3, i%3]  # Select the current axes
    im = ax.imshow(y_test[i], cmap='gray')  
    pred = ax.imshow(y_pred[i], cmap='hot', alpha=0.6)

# Add a colorbar to the right of the figure
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(pred, cax=cbar_ax, orientation='vertical')

# fig.savefig('/rds/general/user/nmk120/home/Mod_HypP/LDA1/mod_im_output1_newThr.png')
fig.savefig('/rds/general/user/nmk120/home/Mod_HypP_3D/64_aug/mod_im_output1_n.png')


# Convert probabilities to binary predictions (0 or 1)
y_pred_binary = (y_pred > 0.41).astype(int)

#Comparison of images , using the actual label and the predicted after binary classification with threshold
plt.figure(figsize=(18, 12))

for i in range(3):
    for j in range(2):
        index = 2*i + j

        # Plot the actual output
        plt.subplot(3, 4, 4*i + 2*j + 1)
        plt.imshow(y_test[index], cmap='gray')  
        plt.title(f'Actual Output {index+1}')

        # Plot the predicted output
        plt.subplot(3, 4, 4*i + 2*j + 2)
        plt.imshow(y_pred_binary[index], cmap='gray')
        plt.title(f'Predicted Output {index+1}')

plt.tight_layout()
plt.savefig('/rds/general/user/nmk120/home/Mod_HypP_3D/64_aug/mod_im_output2_n.png')
# plt.savefig('/rds/general/user/nmk120/home/Mod_HypP/LDA1/mod_im_output2_newThr.png')

print('done')

