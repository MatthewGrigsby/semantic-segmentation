import os
import cv2
import numpy as np
from keras.utils import to_categorical
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import torch


class LoadData:
    def __init__(self,
                 train_img_path,
                 train_mask_path,
                 val_img_path,
                 val_mask_path,
                 backbone,
                 batch_size,
                 n_classes,
                 input_shape, 
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
        self.preprocess_input = smp.encoders.get_preprocessing_fn(self.backbone)

    def train_generator(self, img_path, mask_path, batch_size, n_workers):
        dataset = Dataset(
            img_path, 
            mask_path, 
            augmentation=get_training_augmentation(input_shape=self.input_shape), 
            preprocessing=get_preprocessing(self.preprocess_input),
            classes=[str(x) for x in range(self.n_classes)],
        )
        return MultiEpochsDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    def create_generators(self):
        train_img_gen = self.train_generator(self.train_img_path, self.train_mask_path, batch_size=self.batch_size, n_workers=8)
        val_img_gen = self.train_generator(self.val_img_path, self.val_mask_path, batch_size=1, n_workers=2)
        return train_img_gen, val_img_gen



class Dataset(BaseDataset):
    """
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
        ):
        self.img_ids = os.listdir(images_dir)
        self.msk_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.img_ids]
        self.masks_fps = [os.path.join(masks_dir, msk_id) for msk_id in self.msk_ids]
        
        # convert str names to class values on masks
        self.class_values = [classes.index(cls.lower()) for cls in classes]
        
        self.classes = classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(self.masks_fps[i])
        mask = cv2.imread(self.masks_fps[i], 0)
        # print(mask.shape)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('int')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return (
            torch.from_numpy(image.astype('float32')),#.permute(2, 0, 1), 
            torch.from_numpy(mask.astype('int64'))# .permute(2, 0, 1)
        )
        
    def __len__(self):
        return len(self.img_ids)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_training_augmentation(input_shape):
    h = input_shape[0]
    w = input_shape[1]
    train_transform = [
        ######################################
        ### Augmentation options version 1 ###
        ######################################
        # albu.OneOf([
        #     albu.RandomSizedCrop(min_max_height=(h//2, h//1.2), height=h, width=w, p=0.5),
        #     albu.Rotate(p=0.5),
        # ], p=1),    
        # albu.OneOf([
        #     albu.HorizontalFlip(p=0.5),
        #     albu.VerticalFlip(p=0.5), 
        # ], p=1),               
        # albu.RandomRotate90(p=0.5),

        ######################################
        ### Augmentation options version 2 ###
        ######################################  
        albu.OneOf([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5), 
            albu.RandomRotate90(p=0.5),
        ], p=1),               
        albu.Rotate(p=0.4), 




        # albu.Transpose(p=0.5),
        # albu.OneOf([
        #     albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        #     albu.GridDistortion(p=0.5),
        #     albu.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
        # ], p=0.8),
        # albu.OneOf([
        #     albu.RandomGamma(p=0.5),
        #     albu.CLAHE(p=0.5),
        # ], p=1),
        # albu.RandomBrightnessContrast(p=0.8),    
    ]
    return albu.Compose(train_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

