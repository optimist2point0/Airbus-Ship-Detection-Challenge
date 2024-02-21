import argparse
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


import tensorflow as tf
import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers.legacy import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping


from skimage.morphology import label
from skimage.io import imread
from skimage.transform import resize


# Data generator for images
class Generator(tf.keras.utils.Sequence):

    def __init__(self, folder, filenames, data_frame=None, batch_size=32, image_size=(256, 256), shuffle=True,
                 predict=False, augment=False):
        self.folder = folder
        self.filenames = filenames
        self.data_frame = data_frame
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.predict = predict
        self.augment = augment
        self.on_epoch_end()

    def __load__(self, filename):
        # load jpg file as numpy array
        img = imread(os.path.join(self.folder, filename))
        # create empty mask
        msk = masks_as_image(self.data_frame.query('ImageId==@filename')['EncodedPixels'])

        img = resize(img.astype(np.float32), self.image_size, mode='reflect') / 255.0
        msk = resize(msk.astype(np.float32), self.image_size, mode='reflect')

        # Image Augmentation
        if self.augment:
            d = np.random.choice([{'zx': 1,
                                   'zy': 1,
                                   'flip_horizontal': False,
                                   'flip_vertical': True},
                                  {'zx': 0.9,
                                   'zy': 0.9,
                                   'flip_horizontal': False,
                                   'flip_vertical': False},
                                  None,
                                  {'zx': 1,
                                   'zy': 1,
                                   'flip_horizontal': True,
                                   'flip_vertical': True},
                                  {'theta': 0,
                                   'zx': 1.1,
                                   'zy': 1.1,
                                   'flip_horizontal': True,
                                   'flip_vertical': False},
                                  ])
            if d:
                img = ImageDataGenerator().apply_transform(img, d)
                msk = ImageDataGenerator().apply_transform(msk, d)

        return img, msk

    def __loadpredict__(self, filename):
        # load jpg file as numpy array
        img = imread(os.path.join(self.folder, filename))
        # resize both image and mask
        img = resize(img.astype(np.float32), self.image_size, mode='reflect') / 255.0
        return img

    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)

    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)


# defs for decode and encode our masks of ships (Yeap, i took it from kaggle)
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    # if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


# def to split images into the classes by color channels mean
def channel_class(x):
    for i in range(6):
        if x < (i + 1) * 50:
            return i
        else:
            pass


def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    # Conv2D then ReLU activation
    x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = tf.keras.layers.MaxPool2D(2)(f)
    p = tf.keras.layers.Dropout(0.3)(p)
    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = tf.keras.layers.concatenate([x, conv_features])
    # dropout
    x = tf.keras.layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x


# dice coefficient implementation
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


# loss for model 0.2*binary_crossentropy+ 0.8*(1 - dice_coef)
def bce_dice_coef_loss(y_true, y_pred, smooth=1):
    return 0.2*binary_crossentropy(y_true, y_pred) + 0.8*(1 - dice_coef(y_true, y_pred, smooth))


# TP rate metric
def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--folder_image_path', type=str, help='Path to the folder containing training images')
    parser.add_argument('--train_labels_path', type=str, help='Path to the training labels CSV file')
    parser.add_argument('--result_model_path', type=str, help='Path to save the trained model weights')
    parser.add_argument('--nb_epoch', type=int, help='Number of epoch model to study')
    args = parser.parse_args()

    FOLDER_IMAGE_PATH = args.folder_image_path
    TRAIN_LABELS_PATH = args.train_labels_path
    RESULT_MODEL_PATH = args.result_model_path
    NB_EPOCH = args.nb_epoch

    RANDOM_SEED = 42
    VALID_SIZE = 0.1
    random.seed(RANDOM_SEED)

    # PREPARE DATA
    # Info for images and their labels
    # masks of ships per image
    df = pd.read_csv(TRAIN_LABELS_PATH)

    # count number of ships as new dataframe
    num_of_ships = df.groupby('ImageId')['EncodedPixels'].count().rename("Num_of_ships").reset_index()

    # select images with no target and with targets for different splitting
    imgs_0 = num_of_ships[num_of_ships['Num_of_ships'] == 0].copy().reset_index(drop=True)
    imgs_1 = num_of_ships[num_of_ships['Num_of_ships'] > 0].copy().reset_index(drop=True)

    # split images with ships with stratifying by number of ships
    train_filenames, valid_filenames = train_test_split(
        imgs_1['ImageId'].values,
        test_size=VALID_SIZE,
        stratify=imgs_1['Num_of_ships'],
        random_state=RANDOM_SEED
    )

    # count mean value of all channels(RGB)
    channels_mean = []
    for img_name in imgs_0['ImageId'].values:
        try:
            img = imread(os.path.join(FOLDER_IMAGE_PATH, img_name))
            channels_mean.append(np.mean(img))
        except:
            channels_mean.append(0)

    imgs_0['channels_mean'] = channels_mean
    # round mean values
    imgs_0 = imgs_0.copy().round()
    imgs_0['class'] = imgs_0['channels_mean'].apply(lambda x: channel_class(x))

    # split images with no ships with stratifying by class of color gamma to undersample
    _, train_filenames_0, _, classes_0 = train_test_split(
        imgs_0['ImageId'].values, imgs_0['class'].values,
        test_size=VALID_SIZE,
        stratify=imgs_0['class'],
        random_state=RANDOM_SEED
    )
    train_filenames_0, valid_filenames_0 = train_test_split(
        train_filenames_0,
        test_size=VALID_SIZE,
        stratify=classes_0,
        random_state=RANDOM_SEED
    )

    # concatenate images with ships and without and shuffle
    random.seed(RANDOM_SEED)

    train_filenames = np.concatenate((train_filenames, train_filenames_0))
    random.shuffle(train_filenames)

    valid_filenames = np.concatenate((valid_filenames, valid_filenames_0))
    random.shuffle(valid_filenames)

    print(f"Train shape: {train_filenames.shape}\nValid shape: {valid_filenames.shape}")

    # Train and valid data generators
    train_gen = Generator(FOLDER_IMAGE_PATH, train_filenames, df, augment=True)
    valid_gen = Generator(FOLDER_IMAGE_PATH, valid_filenames, df)

    inputs = tf.keras.layers.Input((256, 256, 3))

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 32)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 64)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 128)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 256)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 512)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 256)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 128)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 64)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 32)
    # outputs
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u9)
    unet_model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name="U-Net")

    # Model compile
    unet_model.compile(optimizer=Adam(1e-4),
                       loss=bce_dice_coef_loss,
                       metrics=[dice_coef, 'binary_accuracy', true_positive_rate])

    checkpoint = ModelCheckpoint(os.path.join(RESULT_MODEL_PATH, 'u_net_weights_best.h5'), monitor='val_dice_coef', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=True)
    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=5)
    callbacks_list = [checkpoint, early]

    # Training model
    loss_history = [unet_model.fit(train_gen,
                                   epochs=NB_EPOCH,
                                   validation_data=valid_gen,
                                   callbacks=callbacks_list
                                   )]

