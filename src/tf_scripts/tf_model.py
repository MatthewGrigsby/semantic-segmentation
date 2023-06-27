import segmentation_models as sm
import tensorflow as tf
import numpy as np


class MultiSegmentModelTensorflow:
    def __init__(self,
                 backbone,
                 n_classes,
                 activation,
                 weights,
                 batch_size,
                 steps_per_epoch,
                 val_steps_per_epoch,
                 model_path,
                 ):
        self.epochs = None
        self.history = None
        self.model = None
        self.train_img_gen = None
        self.val_img_gen = None
        self.val_steps_per_epoch = val_steps_per_epoch
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.backbone = backbone
        self.n_classes = n_classes
        self.activation = activation
        self.weights = weights
        self.model_path = model_path

    def compile(self, input_shape, optim, loss_function, metrics, encoder_freeze=False):
        self.model = sm.Unet(
            self.backbone,
            input_shape=input_shape,
            encoder_weights=self.weights,
            classes=self.n_classes,
            activation=self.activation,
            encoder_freeze=encoder_freeze
        )
        self.model.compile(optim, loss_function, metrics=metrics, run_eagerly=True)
        print(self.model.summary())

    def fit(self, train_img_gen, val_img_gen, epochs, earlystopping):
        print('Batch size: ', self.batch_size)
        self.history = self.model.fit(
            train_img_gen,
            steps_per_epoch=self.steps_per_epoch,
            batch_size=self.batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=val_img_gen,
            validation_steps=self.val_steps_per_epoch, 
            callbacks = [earlystopping]
        )
        self.train_img_gen = train_img_gen
        self.val_img_gen = val_img_gen
        self.epochs = epochs

    def save(self):
        self.model.save(self.model_path)
        np.save(self.model_path[:-5] + '_history.npy', self.history.history)


class DelayedEarlyStopping(tf.keras.callbacks.EarlyStopping):
        def __init__(self, burn_in, **kwargs):
            super(DelayedEarlyStopping, self).__init__(**kwargs)
            self.burn_in = burn_in

        def on_epoch_end(self, epoch, logs=None):
            if epoch >= self.burn_in:
                super().on_epoch_end(epoch, logs)
            else:
                super().on_train_begin(logs=None)