import cv2
import os
import torch
from torch_model import *
from torch_load_data import *

########################
### Data directories ###
########################
PATCH_SIZE = 1024
TRAIN_IMAGE_DIRECTORY_PATH = '../../data/data_for_training_and_testing_n' + str(PATCH_SIZE) + '/train_images/train/'
TRAIN_MASK_DIRECTORY_PATH = '../../data/data_for_training_and_testing_n' + str(PATCH_SIZE) + '/train_masks/train/'
VAL_IMAGE_DIRECTORY_PATH = '../../data/data_for_training_and_testing_n' + str(PATCH_SIZE) + '/val_images/val/'
VAL_MASK_DIRECTORY_PATH = '../../data/data_for_training_and_testing_n' + str(PATCH_SIZE) + '/val_masks/val/'

##################
### Model info ###
##################
N_CLASSES = 6
BATCH_SIZE = 3
ARCHITECTURE = 'fpn'
ACTIVATION = 'softmax'
WEIGHTS = 'imagenet'
BACKBONE = 'mit_b3'
DEVICE = 'cuda'
LEARNING_RATE = 0.0001
EPOCHS = 350
BURN_IN = 150

MODEL_PATH = f'../../models/{ARCHITECTURE}_{BACKBONE}_{WEIGHTS}_{BATCH_SIZE}b_{N_CLASSES}c_{PATCH_SIZE}p_v3.pth'

def main():
    #######################
    ### Get input shape ###
    #######################
    file_location = TRAIN_IMAGE_DIRECTORY_PATH
    file_path = file_location + os.listdir(file_location)[0]
    input_shape = cv2.imread(file_path, 1).shape
    print(torch.cuda.get_device_name(0))
    print('Input shape: ', input_shape)
    print('Batch size: ', BATCH_SIZE)
    print('Backbone: ', BACKBONE)
    print(f'Burn in: {BURN_IN} epochs')

    ###################
    ### Load images ###
    ###################
    sampler = LoadData(
        train_img_path=TRAIN_IMAGE_DIRECTORY_PATH,
        train_mask_path=TRAIN_MASK_DIRECTORY_PATH,
        val_img_path=VAL_IMAGE_DIRECTORY_PATH,
        val_mask_path=VAL_MASK_DIRECTORY_PATH,
        backbone=BACKBONE,
        batch_size=BATCH_SIZE,
        n_classes=N_CLASSES,
        input_shape=input_shape,
    )
    train_img_gen, val_img_gen = sampler.create_generators()

    ###################
    ### Build model ###
    ###################
    num_train_imgs = len(os.listdir(TRAIN_IMAGE_DIRECTORY_PATH))
    num_val_images = len(os.listdir(VAL_IMAGE_DIRECTORY_PATH))
    print(f'Found {num_train_imgs} training images and {num_val_images} val images')
    model = MultiSegmentModelPytorch(
        backbone=BACKBONE,
        n_classes=N_CLASSES,
        activation=ACTIVATION,
        weights=WEIGHTS,
        batch_size=BATCH_SIZE,
        model_path=MODEL_PATH,
        early_stopping = EarlyStopping(patience=20, min_delta=0),
        burn_in = BURN_IN
    )
    model.compile()
    print(model.model.eval())
    print('Total parameters: ', sum(p.numel() for p in model.model.parameters()))
    print('Trainable parameters: ', sum(p.numel() for p in model.model.parameters() if p.requires_grad))

    ###################
    ### Train model ###
    ###################
    total_loss = utils.losses.JaccardLoss()
    # total_loss = utils.losses.DiceLoss()
    # total_loss = smp.losses.TverskyLoss(mode='multiclass') # ; total_loss.__name__ = 'focal_loss'
    # total_loss = smp.losses.DiceLoss(mode='multiclass')
    metrics = [utils.metrics.IoU(threshold=0.5),]
    optimizer = torch.optim.Adam([dict(params=model.model.parameters(), lr=LEARNING_RATE),])
    model.fit(
        train_img_gen=train_img_gen, 
        val_img_gen=val_img_gen, 
        epochs=EPOCHS, 
        optim=optimizer,
        loss_function=total_loss,
        metrics=metrics,
        device=DEVICE
    )


if __name__ == '__main__':
    main()
