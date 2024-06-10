#Note - Change the filepath when saving according to what is required

import sys
print(sys.executable)
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#imports for code #######################################################
import gc
import numpy as np
from tensorflow.keras.utils import Sequence 
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import time
from tensorflow.keras.activations import gelu
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LeakyReLU, AveragePooling3D, Conv2D, Conv3D, MaxPooling3D, Flatten, Dense, Conv2DTranspose, Flatten, Dense, UpSampling2D, Reshape, Dropout, BatchNormalization, Activation, Lambda, concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight, compute_sample_weight
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch, GridSearch
import kerastuner
from keras.regularizers import l2
import optuna
from optuna_integration.keras import KerasPruningCallback
from optuna.visualization import plot_optimization_history

optuna.logging.enable_default_handler() # allow logging errors

#load files in - these are for the augmented larger dataset
data = np.load('/rds/general/user/nmk120/home/wl23/all_spec_data_aug_rot.npy', allow_pickle=True)
labels = np.load('/rds/general/user/nmk120/home/wl23/all_spec_labels_aug_rot.npy', allow_pickle=True)
data = np.expand_dims(data, axis=-1) 

#check data type and shape
print("Data and labels type and shape imported:", data.dtype, data.shape, labels.dtype, labels.shape)

#ensure data of type that can be used by models
data = data.astype('float32')
labels = labels.astype('float32')

#split data into hold-out test set and data to go into cross validation for further splitting 
X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels, test_size=0.1, random_state=42) 

###################################################################################
#data generator implemented using: https://stackoverflow.com/questions/62916904/failed-copying-input-tensor-from-cpu-to-gpu-in-order-to-run-gatherve-dst-tensor
class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
###################################################################################

# Get the list of all available GPUs ##############################################
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    print("GPUs detected. TensorFlow will automatically choose an available GPU.")
else:
    print("No GPUs detected. Running on CPU.")
###################################################################################

#weighted focal loss function - higher value of gamma more the model focuses on miclassified examples, alpha[weight0, wieght1] ###############
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


#build model function that varies based on the paramters selected by optuna search
def build_model(alpha1, alpha2, gamma, optimizer, n_layers, filter_num, conv_size, pool_size, dropout_rate, l2_f, use_bn, activation, dimred_act, kernel_initialiser, act_alpha=None, act_alpha_DR=None):
    if activation == 'leaky_relu':
        activation = LeakyReLU(alpha=act_alpha) #alpha value for activation function if leaky relu selected
        
    if dimred_act == 'leaky_relu':
        dimred_act = LeakyReLU(alpha=act_alpha_DR) #alpha value for activation function if leaky relu selected for 1x1 dimensionality reduction convolution
    
    inputs = Input((200, 200, 1024, 1))  #change according to the input size, ie. for PCA the depth will be different
    print(inputs.shape)
    #chnage the depth dimension based on what is required
    x = Conv3D(1, (1, 1, 1023), activation = dimred_act, kernel_initializer=kernel_initialiser)(inputs) #no. for depth kernel size calc using formula: Out = (W−F+2P)/S+1,where W is the input volume size, F is the kernel size, S is the stride, and P is the padding. but with stride=1 and valid padding (becuase want to red dims and also the edges of spec from visul inspection of spectra hold minimal info), formula becomes: Out = W - F + 1
    # 2 = 1024 - F + 1 --> F = 1023
    print(inputs.shape)

    convs = []
    pools = [x]

# Encoder
    for i in range(n_layers):
        x = Conv3D(filter_num * 2**i, (conv_size, conv_size, conv_size), activation=activation, padding='same', kernel_regularizer=l2(l2_f), kernel_initializer=kernel_initialiser)(x)  # Change to Conv3D
        if use_bn:
            x = BatchNormalization()(x)
        x = Conv3D(filter_num * 2**i, (conv_size, conv_size, conv_size), activation=activation, padding='same', kernel_regularizer=l2(l2_f), kernel_initializer=kernel_initialiser)(x)
        if use_bn:
            x = BatchNormalization()(x)
        convs.append(x)
        x = MaxPooling3D(pool_size=(pool_size, pool_size, 1))(x)  # Change to MaxPooling3D
        x = Dropout(dropout_rate)(x) 
        pools.append(x)

    # Middle
    x = Conv3D(filter_num * 2**n_layers, (conv_size, conv_size, conv_size), activation=activation, padding='same', kernel_regularizer=l2(l2_f), kernel_initializer=kernel_initialiser)(x)  # Change to Conv3D
    if use_bn:
        x = BatchNormalization()(x)
    x = Conv3D(filter_num * 2**n_layers, (conv_size, conv_size, conv_size), activation=activation, padding='same', kernel_regularizer=l2(l2_f), kernel_initializer=kernel_initialiser)(x)  # Change to Conv3D
    if use_bn:
        x = BatchNormalization()(x)

    x = AveragePooling3D(pool_size=(1, 1, x.shape[-2]))(x)
    x = K.squeeze(x, axis=3)

    convs = [AveragePooling3D(pool_size=(1, 1, conv.shape[-2]))(conv) for conv in convs]
    convs = [K.squeeze(conv, axis=3) for conv in convs]

    # Decoder
    for i in range(n_layers-1, -1, -1):
        x = Conv2DTranspose(filter_num * 2**i, (conv_size, conv_size), strides=(pool_size, pool_size), padding='same', kernel_initializer=kernel_initialiser)(x)  # Change to Conv2DTranspose
        if use_bn:
            x = BatchNormalization()(x)
        x = concatenate([x, convs[i]])
        x = Conv2D(filter_num * 2**i, (conv_size, conv_size), activation=activation, padding='same', kernel_regularizer=l2(l2_f), kernel_initializer=kernel_initialiser)(x)  # Change to Conv2D
        if use_bn:
            x = BatchNormalization()(x)
        x = Conv2D(filter_num * 2**i, (conv_size, conv_size), activation=activation, padding='same', kernel_regularizer=l2(l2_f), kernel_initializer=kernel_initialiser)(x)
        if use_bn:
            x = BatchNormalization()(x)

    out = Conv2D(1, (1, 1), activation='sigmoid')(x) #sigmoid actiivation since binary classification task

    model = Model(inputs=inputs, outputs=out)

    model.compile(optimizer=optimizer, 
                  loss=focal_loss(alpha=[alpha1, alpha2], gamma=gamma), 
                  metrics=['AUC', 'accuracy'])

    return model

#defined for saving values from cross val
best_auc = -np.inf
best_model_path = "" 
best_trial_number = -1
best_fold_number = -1
best_fold_per_trial = {}


n_splits = 6 # this allows for equal number of val and test, as if a normal split - so becomes approx, 34 traini, 9 val and 9 scoring (and 6 to test right at the end with the best model)
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
def objective(trial):
    K.clear_session()
    gc.collect()

    global best_auc
    global best_model_path
    global best_trial_number
    global best_fold_number
    alpha1 = trial.suggest_float('alpha1', 0.0, 1.0)
    alpha2 = trial.suggest_float('alpha2', 0.0, 1.0)
    gamma = trial.suggest_float('gamma', 0.0, 5.0)
    batch_size = trial.suggest_int('batch_size', 1, 29) #upto half the batch size to avoid OOM
    n_layers = trial.suggest_int('n_layers', 1, 3) #cant do 4 coz of the error on concatenate
    use_bn = trial.suggest_categorical('use_bn', [True, False])
    filter_num = trial.suggest_categorical('filter_num', [8, 16, 32, 64])
    conv_size = trial.suggest_categorical('conv_size', [3, 5, 7])
    pool_size = trial.suggest_categorical('pool_size', [2]) #cant do 4 here either due to the concateenate 
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    l2_f = trial.suggest_float('l2_f', 0.0, 1.0)
    dimred_act = trial.suggest_categorical('dimred_act', [None, 'relu', 'leaky_relu', 'gelu', 'elu', 'selu']) #for the dimension reduction, to see whether a aactivation is benficil or not
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'gelu', 'elu', 'selu']) #avaliable built in relu functions, this doesnt have an option of none becuase a typcal unet structure has activtion functiion nd this is wht makes convolutions be able to pick out prominent features
    kernel_initialiser = trial.suggest_categorical('kernel_initialiser', ['he_normal', 'he_uniform', None])

    if activation == 'leaky_relu':
        act_alpha = trial.suggest_float('act_alpha', 0.001, 0.3) # 0.3 as biggest as this is the automatic value assigned and 0.001 as an arbitary very small value
    elif activation == 'gelu':
        activation = gelu
        act_alpha = None
    else:
        act_alpha = None

    if dimred_act == 'leaky_relu':
        act_alpha_DR = trial.suggest_float('act_alpha_DR', 0.001, 0.3)
    elif dimred_act == 'gelu':
        dimred_act = gelu
        act_alpha_DR = None
    else:
        act_alpha_DR = None

    # optimizer = Adam(learning_rate=lr)

    early_stopping = EarlyStopping(monitor='val_auc', patience=5)

    scores = []
    best_auc_for_this_trial = -np.inf
    best_fold_number_for_this_trial = -1

    for fold_number, (train_index, test_index) in enumerate(kfold.split(X_train_val)):
        #around 1 fold assignd to x_scores (52/5 = 9 images), remaining will be in x_train_val = 43 images these then are split into train and val
        X_train1, X_score = X_train_val[train_index], X_train_val[test_index] 
        y_train1, y_score = y_train_val[train_index], y_train_val[test_index]

        #Splitting into training and validation sets
        X_train_kv, X_val_kv, y_train_kv, y_val_kv = train_test_split(X_train1, y_train1, test_size=0.2, random_state=42) #so, 37 or 38 in image for train and 9 or 10 val
        
        #using data generator to retrieve the data according to the batch size to limit OOM
        train_gen = DataGenerator(X_train_kv, y_train_kv, batch_size)
        val_gen = DataGenerator(X_val_kv, y_val_kv, batch_size)

        # Build the model for this fold, using the function above with the selected parameters from the search 
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        optimizer = Adam(learning_rate=lr)

        # Build the model
        model = build_model(alpha1, alpha2, gamma, optimizer, n_layers, filter_num, conv_size, pool_size, dropout_rate, l2_f, use_bn, activation, dimred_act, kernel_initialiser, act_alpha, act_alpha_DR)

        #creating the best model per fold 
        checkpoint = ModelCheckpoint(f'/rds/general/user/nmk120/home/Mod_HypP_3D/2_aug/model_trial_{trial.number}_fold_{fold_number}.h5', monitor='val_auc', verbose=1, save_best_only=True, mode='max')

        batch_size = int(batch_size)
        history = model.fit(train_gen, batch_size=batch_size, epochs=100, validation_data=(val_gen), callbacks=[early_stopping, KerasPruningCallback(trial, 'val_auc'), checkpoint])

        score = model.evaluate(X_score, y_score, verbose=0)
        scores.append(score[1])  # Append the auc

        if max(history.history['val_auc']) > best_auc_for_this_trial:
            best_auc_for_this_trial = max(history.history['val_auc'])
            best_fold_number_for_this_trial = fold_number

        #session cleariing at end to clear all memory from that fold, mitigate OOM
        K.clear_session()
        gc.collect()
    
    best_fold_per_trial[trial.number] = best_fold_number_for_this_trial

    if best_auc_for_this_trial > best_auc:
        best_auc = best_auc_for_this_trial
        best_trial_number = trial.number  
        best_fold_number = best_fold_number_for_this_trial

    return np.mean(scores) #median for robustness against outliers

# Create a study object
study = optuna.create_study(direction='maximize')
print("created study")

# Start the hyperparameter tuning
study.optimize(objective, n_trials=200)
print("started HP tuning")

print("Before saving image")
fig = plot_optimization_history(study)
print("After generating plot, before saving image")
fig.write_image("/rds/general/user/nmk120/home/Mod_HypP_3D/2_aug/optunaS_optimization_history_200t_fl_701515_NEW.png")
print("After saving image")

best_trial_number = study.best_trial.number

print(f"Best Trial Number: {best_trial_number}, Best Fold Number: {best_fold_number}")
alpha1 = study.best_trial.params['alpha1']
alpha2 = study.best_trial.params['alpha2']
gamma = study.best_trial.params['gamma']
best_fold_in_best_trial = best_fold_per_trial[best_trial_number]

# Load the saved best model
load_best_model = load_model(f'/rds/general/user/nmk120/home/Mod_HypP_3D/2_aug/model_trial_{best_trial_number}_fold_{best_fold_in_best_trial}.h5', custom_objects={'focal_loss_fixed': focal_loss(alpha=[alpha1, alpha2], gamma=gamma)})

# Get the best model parameters
lr = study.best_trial.params['lr']
optimizer = Adam(learning_rate=lr)
batch_size = int(study.best_trial.params['batch_size'])
n_layers = study.best_trial.params['n_layers']
# use_pooling = study.best_trial.params['use_pooling']
use_bn = study.best_trial.params['use_bn']
filter_num = study.best_trial.params['filter_num']
conv_size = study.best_trial.params['conv_size']
pool_size = study.best_trial.params['pool_size']
dropout_rate = study.best_trial.params['dropout_rate']
l2_f = study.best_trial.params['l2_f']
activation = study.best_trial.params['activation']
dimred_act = study.best_trial.params['dimred_act']
kernel_initialiser = study.best_trial.params['kernel_initialiser']
act_alpha = study.best_trial.params.get('act_alpha', None) # .get method as like in a dictionary, which will retun none if no value is found
act_alpha_DR = study.best_trial.params.get('act_alpha_DR', None)
best_avg_auc = study.best_value


for fold_number, (train_index, test_index) in enumerate(kfold.split(X_train_val)):
    if fold_number == best_fold_number:
        X_train1, X_score = X_train_val[train_index], X_train_val[test_index]
        y_train1, y_score = y_train_val[train_index], y_train_val[test_index] #not using scores as not evaluating this model here, evaluating later on using unseen test data
        break

X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.2, random_state=42) #to build the best model at the end make train and val splits - doesnt interfere with the kfold (as this is done sep to kfold)

train_gen1 = DataGenerator(X_train, y_train, batch_size)
val_gen1 = DataGenerator(X_val, y_val, batch_size)

# Build the best model by retraining using best metrics 
best_model = build_model(alpha1, alpha2, gamma, optimizer, n_layers, filter_num, conv_size, pool_size, dropout_rate, l2_f, use_bn, activation, dimred_act, kernel_initialiser, act_alpha, act_alpha_DR)
early_stopping = EarlyStopping(monitor='val_auc', patience=5)
# Train the best model retrained
history = best_model.fit(train_gen1, batch_size=batch_size, epochs=100, validation_data=(val_gen1), callbacks=[early_stopping])

plt.figure(figsize=(18, 6))
# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('Model Area under the ROC curve (AUC)')
plt.ylabel('Area under the ROC curve (AUC)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('/rds/general/user/nmk120/home/Mod_HypP_3D/2_aug/retrained_BM_perf_200t.png')

best_med_trial_number = study.best_trial.number
# Get the fold number that produced the highest median AUC in all trials from the dict
best_fold_in_best_trial = best_fold_per_trial[best_med_trial_number]

class DataGeneratorTest(Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x

# Instantiate the data generator
test_gen = DataGeneratorTest(X_test, batch_size=1)

# Define the list of thresholds
thresholds = [0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,  0.6]

# Open the file to write the results
with open('/rds/general/user/nmk120/home/Mod_HypP_3D/2_aug/optuna_mod_hyperParam_metrics_200t_fl_701515_NEW.txt', 'w') as f:
    # Write the best hyperparameters
    f.write('Best Model Metrics:\n')
    f.write(f'Alpha1: {alpha1}\n')
    f.write(f'Alpha2: {alpha2}\n')
    f.write(f'Gamma: {gamma}\n')
    f.write(f'Learning Rate: {lr}\n')
    f.write(f'Batch Size: {batch_size}\n')
    f.write(f'Number of Layers: {n_layers}\n')
    f.write(f'Use BN: {use_bn}\n')
    f.write(f'Dropout rate: {dropout_rate}\n')    
    f.write(f'Conv size: {conv_size}\n')
    f.write(f'Pool size: {pool_size}\n')
    f.write(f'l2 factor: {l2_f}\n')
    f.write(f'Kernel weight initialiser: {kernel_initialiser}\n')
    f.write(f'Activation function for unet layers: {activation}\n')
    f.write(f'Activation alpha value for leaky relu of unet layers: {act_alpha}\n')
    f.write(f'Activation function for dimension reduction layer: {dimred_act}\n')
    f.write(f'Activation alpha value for leaky relu of dim red layer: {act_alpha_DR}\n')
    f.write(f"The best median auc from trials value is: {best_avg_auc}, trial number: {best_med_trial_number}, fold number: {best_fold_in_best_trial}\n") #get the auc from th logs
    f.write(f"Best Trial Number: {best_trial_number}, Best Fold Number: {best_fold_number}, Best AUC: {best_auc}\n") #check
    for i in range(n_layers):
        f.write(f'Layer {i}: {filter_num * 2**i} filters')

    # f.write(f'Use Pooling: {use_pooling}\n')
    # f.write(f'Use Dropout: {use_dropout}\n')

    # Loop over the thresholds
    for threshold in thresholds:
        predictions = []
        for batch in test_gen:
            batch_predictions = best_model.predict(batch)
            predictions.extend(batch_predictions)
        predictions = np.array(predictions)
        predictions_binary = (predictions > threshold).astype(int)

        # Make predictions with the best model reloaded
        predictions_loaded_bm = []
        for batch in test_gen:
            batch_predictions = load_best_model.predict(batch)
            predictions_loaded_bm.extend(batch_predictions)
        predictions_loaded_bm = np.array(predictions_loaded_bm) 
        predictions_binary_loaded_bm = (predictions_loaded_bm > threshold).astype(int)


        # Evaluate the best model retrained
        report = classification_report(y_test.flatten(), predictions_binary.flatten())  
        cm = confusion_matrix(y_test.flatten(), predictions_binary.flatten()) 

        # Evaluate the loaded 'best' model
        report_lbm = classification_report(y_test.flatten(), predictions_binary_loaded_bm.flatten())  
        cm_lbm = confusion_matrix(y_test.flatten(), predictions_binary_loaded_bm.flatten()) 


        fpr, tpr, thresholds = roc_curve(y_test.flatten(), predictions_binary.flatten())
        roc_auc = auc(fpr, tpr) 

        fpr_lbm, tpr_lbm, thresholds = roc_curve(y_test.flatten(), predictions_binary_loaded_bm.flatten())
        roc_auc_lbm = auc(fpr_lbm, tpr_lbm) 

        # Write the results for the current threshold
        f.write(f'\nThreshold: {threshold}\n')
        f.write(f'Classification Report BM retrained:\n {report}\n')
        f.write(f'Confusion Matrix BM retrained:\n {cm}\n')
        f.write(f'AUC BM retrained: {roc_auc}\n')
        #results for reloaded best model
        f.write(f'Classification Report BM loaded:\n {report_lbm}\n')
        f.write(f'Confusion Matrix BM loaded:\n {cm_lbm}\n')
        f.write(f'AUC BM loaded: {roc_auc_lbm}\n')
