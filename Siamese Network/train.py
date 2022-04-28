import cv2
import os
import random
import pandas as pd
import numpy as np
import timm
from PIL import Image
from tqdm import tqdm
from config import CFG
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

class SiameseDataset(Dataset):

  def __init__(self, image_df, transform=None): #image_df contains image_id1, image_id2, label
    self.image_df = image_df
    self.transform = transform

  def __len__(self):
    return self.image_df.shape[0]

  def __getitem__(self, index):
    image1 = cv2.imread(self.image_df.image1.iloc[index])
    image2 = cv2.imread(self.image_df.image2.iloc[index])
    if self.transform is not None:
        image1 = self.transform(image=image1)['image']
        image2 = self.transform(image=image2)['image']
    label = self.image_df.label.iloc[index]
    return (image1, image2, label)

class SiameseNetwork(nn.Module):

  def __init__(self, model_name):
    super(SiameseNetwork, self).__init__()
    self.embed_model = timm.create_model(model_name, pretrained=True)
    feats = self.embed_model.classifier.in_features
    self.embed_model.classifier = nn.Linear(feats, CFG.embed_dims)

  def forward(self, image1, image2):
    img1_embeds = self.embed_model(image1)
    img2_embeds = self.embed_model(image2)
    return img1_embeds, img2_embeds

class SiameseNetworkDirver(pl.LightningModule):

    def __init__(self, model, criterion, lr):
        super(SiameseNetworkDirver, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, images1, images2):
        return self.model(image1, image2)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CFG.lr)
        return self.optimizer

    def training_step(self, batch, batch_idx):
        image1, image2, labels = batch[0], batch[1], batch[2]
        img1_embeds, img2_embeds = self.model(image1, image2)
        loss = self.criterion(img1_embeds, img2_embeds, labels)
        logs = { 'train_loss': loss, 'lr': self.optimizer.param_groups[0]['lr'] }
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image1, image2, labels = batch[0], batch[1], batch[2]
        img1_embeds, img2_embeds = self.model(image1, image2)
        loss = self.criterion(img1_embeds, img2_embeds, labels)
        logs = { 'val_loss': loss }
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        image1, image2 = batch[0], batch[1]
        img1_embeds, img2_embeds = self.model(image1, image2)
        return img1_embeds, img2_embeds

class SimilarityLoss(nn.Module):

  def __init__(self, margin=2.0):
    super(SimilarityLoss, self).__init__()
    self.margin = margin

  def forward(self, img1_embeds, img2_embeds, label):
    euclidean_distance = F.pairwise_distance(img1_embeds, img2_embeds, keepdim=True)
    loss = torch.mean((label*torch.pow(euclidean_distance, 2) + 
                      (1 - label)*torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)))
    return loss

def get_distance(predictions):
    distance = []
    img1_embeds, img2_embeds = [], []
    for batch in predictions:
        img1_embeds = batch[0]
        img2_embeds = batch[1]
        euclidean_distance = F.pairwise_distance(img1_embeds, img2_embeds, keepdim=True)
        distance.append(euclidean_distance)
        
    preds_dec = []
    for batch in distance:
        preds_dec += batch.squeeze(1).tolist()
        
    preds = [1 if pred < 1 else 0 for pred in preds_dec]

    return preds_dec, preds

if __name__ == "__main__":
    data = create_dataset(CFG.train_path, CFG.similarity_factor, CFG.dissimilarity_factor)
    dataframe = pd.DataFrame(data, columns=['image1', 'image2', 'label'])

    labels = dataframe.label.values
    dataframe.drop('label', inplace=True, axis=1)
    train_data, val_data, train_labels, val_labels = train_test_split(dataframe, labels, 
                                                                  stratify=labels, 
                                                                  test_size=CFG.val_size, 
                                                                  random_state=CFG.seed)
    train_data['label'] = train_labels
    val_data['label'] = val_labels

    transform = train_transform_object(CFG.image_size)
    train_dataset = SiameseDataset(train_data, transform)
    val_dataset = SiameseDataset(val_data, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False)

    logger = CSVLogger(save_dir=CFG.save_dir, name=CFG.model_name+'_logs')
    logger.log_hyperparams(CFG.__dict__)

    checkpoint_callback = ModelCheckpoint(monitor=CFG.monitor,
                                          save_top_k=1,
                                          save_last=True,
                                          save_weights_only=True,
                                          filename='{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}',
                                          verbose=False,
                                          mode='min')

    early_stop_callback = EarlyStopping(monitor=CFG.monitor, 
                                        patience=CFG.patience, 
                                        verbose=False, 
                                        mode="min")

    trainer = Trainer(
        max_epochs=CFG.epochs,
        gpus=[0],
        accumulate_grad_batches=CFG.accumulate,
        callbacks=[checkpoint_callback, early_stop_callback], 
        logger=logger,
        weights_summary='top',
    )

    model = SiameseNetwork(CFG.model_name)
    criterion = SimilarityLoss(margin=CFG.loss_margin)
    driver = SiameseNetworkDirver(model, criterion, CFG.lr)

    trainer.fit(driver, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    show_metrics(trainer)

    data = create_dataset(CFG.test_path, CFG.test_similarity_factor, CFG.test_dissimilarity_factor)
    test_data = pd.DataFrame(data, columns=['image1', 'image2', 'label'])

    test_dataset = SiameseDataset(test_data, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)

    predictions = trainer.predict(dataloaders=test_dataloader)

    distances, pred_labels = get_distance(predictions)
    true_labels = test_data.label.values

    print(round(accuracy_score(true_labels, pred_labels), 2))
    print(classification_report(true_labels, pred_labels))

    test_data['predictions'] = distances
    test_data.to_csv('test_set_with_predictions.csv', index=False)
