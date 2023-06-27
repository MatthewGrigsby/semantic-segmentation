import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as utils
import torch
import numpy as np
import pandas as pd
# from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from torch_train import *


class MultiSegmentModelPytorch:
    def __init__(self,
                 backbone,
                 n_classes,
                 activation,
                 weights,
                 batch_size,
                 model_path,
                 early_stopping,
                 burn_in
                 ):
        self.epochs = None
        self.history = None
        self.model = None
        self.train_img_gen = None
        self.val_img_gen = None
        self.batch_size = batch_size
        self.backbone = backbone
        self.n_classes = n_classes
        self.activation = activation
        self.weights = weights
        self.model_path = model_path
        self.burn_in = burn_in
        self.early_stopping = early_stopping

    def compile(self):
        self.model = smp.MAnet(
            encoder_name=self.backbone, 
            encoder_weights=self.weights, 
            classes=self.n_classes, 
            activation=self.activation,
        )
        

    def fit(self, 
            train_img_gen, 
            val_img_gen, 
            epochs, 
            optim, 
            loss_function, 
            metrics, 
            device, 
        ):
        self.train_img_gen = train_img_gen
        self.val_img_gen = val_img_gen
        self.epochs = epochs
        print('Batch size: ', self.batch_size)
        train_epoch = TrainEpoch(
            self.model, 
            loss=loss_function, 
            metrics=metrics, 
            optimizer=optim,
            device=device,
            verbose=True,
            scaler=torch.cuda.amp.GradScaler(enabled=True)
        )
        valid_epoch = ValidEpoch(
            self.model, 
            loss=loss_function, 
            metrics=metrics, 
            device=device,
            verbose=True,
        )

        max_score = np.inf
        COLUMNS = ['train_' + loss_function.__name__, 'train_iou', 'val_' + loss_function.__name__, 'val_iou']
        history = pd.DataFrame(columns=COLUMNS)
        for i in range(0, epochs):
            print('\nEpoch: {}'.format(i))
            
            # train model
            train_logs = train_epoch.run(train_img_gen)
            valid_logs = valid_epoch.run(val_img_gen)

            # save epoch history 
            tmp = pd.DataFrame([list(train_logs.values()) + list(valid_logs.values())], columns=COLUMNS, index=[i])
            history = pd.concat([history, tmp])

            if i >= self.burn_in:
                current_metric = train_logs[loss_function.__name__]
                # check if updated model is best yet
                if  max_score >= current_metric:
                    max_score = current_metric
                    torch.save(self.model, self.model_path)
                    history.to_csv(self.model_path[:-4] + '_history.csv', index_label='epoch')
                    print(f'Model saved to {self.model_path} on epoch {i}')

                # early stopping
                if self.early_stopping.early_stop(current_metric):
                    print(f"Early stopping triggered! Patience = {self.early_stopping.patience}")
                    break
        


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
