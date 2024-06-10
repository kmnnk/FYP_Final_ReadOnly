import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from scipy.ndimage import rotate


#Load the data
# data = np.load('/rds/general/user/nmk120/home/wl23/combined_spec_array.npy', allow_pickle=True)
# labels = np.load('/rds/general/user/nmk120/home/BCC_all/combined_bcc_array.npy', allow_pickle=True)

data = np.load('/rds/general/user/nmk120/home/wl23/PCAdimred_alldata.npy', allow_pickle=True) #change to LDA when needed
labels = np.load('/rds/general/user/nmk120/home/BCC_all/combined_bcc_array.npy', allow_pickle=True)


########################### aug data 

def flip_rot_images(images, labels, horizontal=True, vertical=True):
    augmented_images = []
    augmented_labels = []
    images = images.astype('float32')
    labels = labels.astype('float32')

    for image, label in zip(images, labels):
        augmented_images.append(image)
        augmented_labels.append(label)

        if horizontal:
            augmented_images.append(np.fliplr(image))
            augmented_labels.append(label)

        if vertical:
            augmented_images.append(np.flipud(image))
            augmented_labels.append(label)

        if horizontal and vertical:
            augmented_images.append(np.flipud(np.fliplr(image)))
            augmented_labels.append(label)

        # Rotation by 90 degrees in the xy-plane
        augmented_images.append(np.rot90(image, k=1, axes=(0, 1)))
        augmented_labels.append(np.rot90(label, k=1, axes=(0, 1)))

        # Rotation by 270 degrees in the xy-plane
        augmented_images.append(np.rot90(image, k=3, axes=(0, 1)))
        augmented_labels.append(np.rot90(label, k=3, axes=(0, 1)))

    # Convert lists to numpy arrays 
    all_images = np.array(augmented_images)
    all_labels = np.array(augmented_labels)

    # Shuffle the images and labels
    all_images, all_labels = shuffle(all_images, all_labels)

    return all_images, all_labels

all_data, all_labels = flip_rot_images(data, labels)

print('labels', np.unique(all_labels)) 
print('are any Nan in labels', np.isnan(all_labels).any()) 
print('are any Nan in labels', np.isnan(all_data).any()) 

# np.save('/rds/general/user/nmk120/home/wl23/all_spec_data_aug_rot.npy', all_data)
# np.save('/rds/general/user/nmk120/home/wl23/all_spec_labels_aug_rot.npy', all_labels)

#chnage to LDA when needed
np.save('/rds/general/user/nmk120/home/wl23/PCAdata_aug_rot.npy', all_data)
np.save('/rds/general/user/nmk120/home/wl23/PCAlabels_aug_rot.npy', all_labels)

print("done creating aug dataset")


