import cv2
import os
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import segmentation_models as sm


class LoadData:
    def __init__(self,
                 train_img_path,
                 train_mask_path,
                 val_img_path,
                 val_mask_path,
                 backbone,
                 batch_size,
                 n_classes,
                 input_shape
                 ):
        self.n_classes = None
        self.resize_shape = None
        self.resize_images = None
        self.train_img_path = train_img_path
        self.train_mask_path = train_mask_path
        self.val_img_path = val_img_path
        self.val_mask_path = val_mask_path
        self.backbone = backbone
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.input_shape = input_shape

    def train_generator(self, train_img_path, train_mask_path):
        img_data_gen_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect'
        )

        image_datagen = ImageDataGenerator(**img_data_gen_args)
        mask_datagen = ImageDataGenerator(**img_data_gen_args)

        image_generator = image_datagen.flow_from_directory(
            train_img_path,
            class_mode=None,
            batch_size=self.batch_size,
            target_size=(self.input_shape[0], self.input_shape[1]),
            seed=100
        )

        mask_generator = mask_datagen.flow_from_directory(
            train_mask_path,
            class_mode=None,
            color_mode='grayscale',
            batch_size=self.batch_size,
            target_size=(self.input_shape[0], self.input_shape[1]),
            seed=100
        )

        train_generator = zip(image_generator, mask_generator)

        for (img, mask) in train_generator:
            img, mask = self.preprocess_data(img, mask, self.n_classes)
            yield img, mask

    def preprocess_data(self, img, mask, num_class):
        img = img.astype('float32') / 255
        preprocess_input = sm.get_preprocessing(self.backbone)
        img = preprocess_input(img)
        mask = to_categorical(mask, num_class)
        return img, mask

    def create_generators(self):
        train_img_gen = self.train_generator(self.train_img_path, self.train_mask_path)
        val_img_gen = self.train_generator(self.val_img_path, self.val_mask_path)
        return train_img_gen, val_img_gen
