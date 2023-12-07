import os
import argparse
from config import *
import torch
from dataloaders.crnn_dataloader import *
from dataloaders.edgecnn_dataloader import *
from trainers.cnn_trainer import CNNTrainer
from trainers.gnn_trainer import GNNTrainer
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    # Argparser
    parser = argparse.ArgumentParser(description='Train Script for RNA REACTIVITY')
    parser.add_argument('-m', '--model_name', help='Choose from existing models, ex:crnn, edgecnn')
    parser.add_argument('-e', '--num_epoch', help='Number of Epochs')

    # TODO: ADD MORE arguments as we go
    args = parser.parse_args()
    
    # Based on model , Load the Datasets
    # model_name = "edgecnn"
    # num_epoch = 1
    if args.model_name == 'crnn':
        model = CNNTrainer("crnn")
        train_val_dataset = ParquetCRNNDataset(P_TRAIN_PARQUET_QUICK)
        test_dataset = InferenceParquetCRNNDataset(P_TEST_PARQUET)
        log = CRNN_LOG
        chk_pnt = CRNN_CHK_PNT
    
    elif args.model_name == 'edgecnn':
        model = GNNTrainer("edgecnn", edge_distance=4)
        train_val_dataset = SimpleGraphDataset(P_TRAIN_PARQUET_QUICK, edge_distance=4)
        test_dataset = InferenceGraphDataset(P_TEST_PARQUET, edge_distance=4)
        log = EDGECNN_LOG
        chk_pnt = EDGECNN_CHK_PNT
    

    # Randomly split the data into train and validation datasets
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_val_dataset, [0.7, 0.3], generator1)
    

    # DATALOADER
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)  
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    # Chose the Trainer and the model you want to train


    #Check point and logger
    checkpoint_callback = ModelCheckpoint(
                dirpath=chk_pnt,
                monitor='val_loss',
                save_top_k=1,
                filename='best-{epoch}-{val_loss:.2f}'
            )

    logger = TensorBoardLogger(save_dir=log, name=f"{args.num_epoch}epoch_{args.model_name}", version=1)
    #logger = TensorBoardLogger(save_dir=log, name=f"1epoch_edgecnn", version=1)



    # fit the model
    pl_trainer = Trainer(max_epochs = 1, accelerator='gpu', devices=1, precision="16-mixed", default_root_dir=chk_pnt)
    pl_trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    #Predictions
    y_pred = pl_trainer.predict(model=model, dataloaders=test_dataloader)
    y_pred = pl_trainer.predict(model=model, dataloaders=test_dataloader)

    #Save the predictions
    return y_pred

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Train Script for RNA REACTIVITY')
    # args = parser.parse_args()
    main()
# main()
