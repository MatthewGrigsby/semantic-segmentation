import tensorflow as tf
from tf_model import *
from tf_load_data import *
import cv2
import os

print(tf.config.list_physical_devices('GPU'))


# Data directories
PATCH_SIZE = 1024
TRAIN_IMAGE_DIRECTORY_PATH = '../../data/data_for_training_and_testing_n' + str(PATCH_SIZE) + '/train_images'
TRAIN_MASK_DIRECTORY_PATH = '../../data/data_for_training_and_testing_n' + str(PATCH_SIZE) + '/train_masks'
VAL_IMAGE_DIRECTORY_PATH = '../../data/data_for_training_and_testing_n' + str(PATCH_SIZE) + '/val_images'
VAL_MASK_DIRECTORY_PATH = '../../data/data_for_training_and_testing_n' + str(PATCH_SIZE) + '/val_masks'

# Model info
N_CLASSES = 6
BATCH_SIZE = 5
ACTIVATION = 'softmax'
WEIGHTS = 'imagenet'
BACKBONE = 'resnet50'
MODEL_PATH = f'../models/unet_{BACKBONE}_{WEIGHTS}_{BATCH_SIZE}batch_{N_CLASSES}classes_p{PATCH_SIZE}_v1.hdf5'
LEARNING_RATE = 0.0001
EPOCHS = 250

early_stopping = DelayedEarlyStopping(
    burn_in=100,
    monitor='val_iou_score',
    min_delta=0,
    patience=20,
    verbose=1,
    mode='max',
    restore_best_weights=True
)


def main():
    # get input shape first
    file_location = TRAIN_IMAGE_DIRECTORY_PATH + '/train/'
    file_path = file_location + os.listdir(file_location)[0]
    input_shape = cv2.imread(file_path, 1).shape
    print('Input shape: ', input_shape)
    
    # load images
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

    ### Build model
    num_train_imgs = len(os.listdir(TRAIN_IMAGE_DIRECTORY_PATH + '/train/'))
    num_val_images = len(os.listdir(VAL_IMAGE_DIRECTORY_PATH + '/val/'))
    steps_per_epoch = num_train_imgs // BATCH_SIZE
    val_steps_per_epoch = num_val_images // BATCH_SIZE
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    metrics = [sm.metrics.iou_score]
    total_loss = sm.losses.categorical_focal_jaccard_loss

    model = MultiSegmentModelTensorflow(
        backbone=BACKBONE,
        n_classes=N_CLASSES,
        activation=ACTIVATION,
        weights=WEIGHTS,
        batch_size=BATCH_SIZE,
        model_path=MODEL_PATH,
        steps_per_epoch=steps_per_epoch,
        val_steps_per_epoch=val_steps_per_epoch
    )

    model.compile(
        input_shape=input_shape,
        optim=optimizer,
        loss_function=total_loss,
        metrics=metrics, 
        # encoder_freeze=True
    )

    # train model
    model.fit(
        train_img_gen=train_img_gen, 
        val_img_gen=val_img_gen, 
        epochs=EPOCHS, 
        earlystopping=early_stopping
    )
    model.save()
    print('Model saved to: ', MODEL_PATH)


if __name__ == '__main__':
    main()
