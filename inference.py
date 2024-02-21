import argparse
import requests

import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from skimage.morphology import label, binary_opening, disk
from skimage.io import imread
from skimage.transform import resize


def download_file_from_google_drive(url, destination):
    session = requests.Session()

    response = session.get(url, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


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


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Predict and create a submission CSV for image classification.')
    parser.add_argument('--test_folder_path', type=str, help='Path to the folder containing test images')
    parser.add_argument('--result_csv_path', type=str, default='submission.csv', help='Path to save the submission CSV file')

    args = parser.parse_args()

    test_folder_path = args.test_folder_path
    result_csv_path = args.result_csv_path

    # Load model weights
    destination = 'u_net_weights_best.h5'
    url = 'https://drive.google.com/uc?id=1ESRh2e5EUytrwxuRt_2HoTAiD-IZnJzB'
    download_file_from_google_drive(url, destination)

    # model init
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
    unet_model.load_weights('u_net_weights_best.h5')

    # load and shuffle filenames
    test_filenames = os.listdir(test_folder_path)
    # Test generator
    test_gen = Generator(folder=test_folder_path, filenames=test_filenames, data_frame=None, batch_size=1000, image_size=(256, 256),
                         shuffle=False, predict=True, augment=False)

    # Predict and save
    out_pred_rows = []
    for imgs, filenames in test_gen:
        # predict batch of images
        preds = unet_model.predict(imgs)
        # loop through batch
        for pred, filename in zip(preds, filenames):
            # resize predicted mask
            pred = resize(pred, (768, 768), mode='reflect')

            pred = binary_opening(pred > 0.5, np.expand_dims(disk(2), -1))
            cur_rles = multi_rle_encode(pred)

            if len(cur_rles) > 0:
                for c_rle in cur_rles:
                    out_pred_rows += [{'ImageId': filename, 'EncodedPixels': c_rle}]
            else:
                out_pred_rows += [{'ImageId': filename, 'EncodedPixels': None}]

    # save dictionary as csv file
    submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
    submission_df.to_csv(result_csv_path, index=False)
