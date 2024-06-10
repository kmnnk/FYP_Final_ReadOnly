import sys
print(sys.executable)
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Conv2DTranspose, Flatten, Dense, UpSampling2D, Reshape, Dropout, BatchNormalization, Activation, Lambda, concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight, compute_sample_weight
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
import kerastuner

#load data
data = np.load('/rds/general/user/nmk120/home/wl23/combined_spec_array.npy', allow_pickle=True)
# data = np.load('/rds/general/user/nmk120/home/wl23/PCAdimred_alldata.npy', allow_pickle=True)
# data = np.load('/rds/general/user/nmk120/home/wl23/LDAdimred_alldata.npy', allow_pickle=True)
labels = np.load('/rds/general/user/nmk120/home/BCC_all/combined_bcc_array.npy', allow_pickle=True)

#data shape/type
print("Data and labels type and shape imported:", data.dtype, data.shape, labels.dtype, labels.shape)

#making sure all data of correct form
data = data.astype('float32')
labels = labels.astype('float32')

#data split iinto 70% train, 15% val and 15% test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Get the list of all available GPUs ##################################################
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    print("GPUs detected. TensorFlow will automatically choose an available GPU.")
else:
    print("No GPUs detected. Running on CPU.")
###################################################################################

#weighted focal loss function for class imbalance - higher value of gamma more the model focuses on miclassified examples, alpha[weight0, wieght1]
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

# Define metrics functions
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

#small model function
def build_model(hp):
    inputs = Input((200, 200, 1024))
    x = Conv2D(1, (1, 1))(inputs)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv2), conv1], axis=-1)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

    out = Conv2D(1, (1, 1), activation='sigmoid')(conv3)

    model = Model(inputs=inputs, outputs=out)
    alpha1 = hp.Choice('alpha1', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    alpha2 = hp.Choice('alpha2', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    model.compile(optimizer='adam', 
                  loss = focal_loss(alpha=[alpha1, alpha2], gamma=hp.Float('gamma', 0.0, 5.0, step=0.5)), 
                  metrics=['AUC', 'accuracy', precision, recall])

    return model

tuner = RandomSearch(
    build_model,
    objective=kerastuner.Objective('val_auc', direction='max'),
    max_trials=200, 
    executions_per_trial=3,
    directory='unet_hyperparam_search',
    project_name='unet'
)

tuner.search_space_summary()

#early stopping added
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_train, y_train, batch_size=16,epochs=25, validation_data=(X_val, y_val), callbacks=[early_stopping])

#getting best returns from search
tuner.results_summary()
best_models = tuner.get_best_models(num_models=3)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

#writing model perf and best params
with open('/rds/general/user/nmk120/home/Mod_HypP/1/RandS_S_mod_hyperParam_metrics.txt', 'w') as f:
    best_models = tuner.get_best_models(num_models=3)
    for i, model in enumerate(best_models):
        f.write(f"Model rank: {i+1}\n")
        alpha1 = best_hps.get('alpha1')
        alpha2 = best_hps.get('alpha2')
        gamma = best_hps.get('gamma')
        f.write(f'Alpha1: {alpha1}\n')
        f.write(f'Alpha2: {alpha2}\n')
        f.write(f'Gamma: {gamma}\n')
        predictions = model.predict(X_test)  
        predictions_binary = (predictions > 0.45).astype(int)
        report = classification_report(y_test.flatten(), predictions_binary.flatten())  
        cm = confusion_matrix(y_test.flatten(), predictions_binary.flatten()) 
        f.write(f'Classification Report:\n {report}\n')
        f.write(f'Confusion Matrix:\n {cm}\n')
        f.write("\n")

    best_model = tuner.get_best_models(num_models=1)[0]
    alpha1 = best_hps.get('alpha1')
    alpha2 = best_hps.get('alpha2')
    gamma = best_hps.get('gamma')
    f.write('Best Model Metrics:\n')
    f.write(f'Alpha1: {alpha1}\n')
    f.write(f'Alpha2: {alpha2}\n')
    f.write(f'Gamma: {gamma}\n')
    predictions = best_model.predict(X_test) 
    predictions_binary = (predictions > 0.45).astype(int)
    report = classification_report(y_test.flatten(), predictions_binary.flatten())  
    cm = confusion_matrix(y_test.flatten(), predictions_binary.flatten())  
    fpr, tpr, thresholds = roc_curve(y_test.flatten(), predictions_binary.flatten())
    roc_auc = auc(fpr, tpr) 
    f.write(f'Classification Report:\n {report}\n')
    f.write(f'Confusion Matrix:\n {cm}\n')
    f.write(f'AUC: {roc_auc}\n')