import argparse
from config import *
import torch
import tqdm
from dataloaders.edgecnn_dataloader import *
from trainers.gnn_trainer import GNNTrainer
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import *


def main():
    # Argparser
    parser = argparse.ArgumentParser(description='Train Script for RNA REACTIVITY')
    parser.add_argument('-m', '--model_name', help='Choose from existing models, ex:edgecnn', required=True)
    parser.add_argument('-e', '--num_epoch', help='Number of Epochs', default=50)
    parser.add_argument('-v', '--version', help='Your model version number for logging and saving')
    parser.add_argument('-b', '--batch_size', help='Batch size for train-test', default=64)
    parser.add_argument('-p', '--precision', help='Train Precision Type', default='bf16-mixed')
    parser.add_argument('-d', '--dataset_type', help='DMS or 2A3', required=True)
    parser.add_argument('--predict_ckpt', help='Model ckpt for prediction', default=None)
    ## TODO: add dataset 2a3-dms type



    # # TODO: ADD MORE arguments as we go
    args = parser.parse_args()

    if args.dataset_type == '2a3':
        log = EDGECNN_2A3_LOG
        chk_pnt = EDGECNN_2A3_CHK_PNT
        train_parquet = P_TRAIN_2A3_PARQUET
        preds = EDGECNN_2A3_PRED
    
    elif args.dataset_type == 'dms':
        log = EDGECNN_DMS_LOG
        chk_pnt = EDGECNN_DMS_CHK_PNT
        train_parquet = P_TRAIN_DMS_PARQUET
        preds = EDGECNN_DMS_PRED

    
    if args.model_name == 'edgecnn':
        train_val_dataset = SimpleGraphDataset(train_parquet, edge_distance=4)
        test_dataset = InferenceGraphDataset(P_TEST_PARQUET, edge_distance=4)
        model = GNNTrainer("edgecnn", num_features=train_val_dataset.num_features, num_channels=4, batch_size=int(args.batch_size))
        # log = EDGECNN_2A3_LOG
        # chk_pnt = EDGECNN_2A3_CHK_PNT
    
    elif args.model_name == 'graphormer':
        train_val_dataset = SimpleGraphDataset(train_parquet, edge_distance=4)
        test_dataset = InferenceGraphDataset(P_TEST_PARQUET, edge_distance=4)
        model = GNNTrainer("graphormer", num_features=train_val_dataset.num_features, num_channels=4)
        # log = GRAPHORMER_LOG
        # chk_pnt = GRAPHORMER_CHK_PNT
    
    # set 'high' or 'highest' for better performance but lower training speed
    torch.set_float32_matmul_precision('high')
    # Randomly split the data into train and validation datasets
    generator1 = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_val_dataset, [0.7, 0.3], generator1)
    
    if args.predict_ckpt is None:
        # DATALOADER
        train_dataloader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)  
        val_dataloader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
        #Checkpoint and logger
        logger = TensorBoardLogger(save_dir=log, name=f"{args.num_epoch}_epoch_{args.model_name}", version=args.version)
        checkpoint_callback = ModelCheckpoint(
                    dirpath=f"{chk_pnt}/version_{args.version}",
                    monitor='val_loss',
                    save_top_k=1,
                    filename='best-{epoch}-{val_loss:.2f}'
                )
        
        # Chose the Trainer and the model you want to train
        
        
        pl_trainer = Trainer(
        max_epochs = int(args.num_epoch), 
        accelerator='gpu', 
        devices=1, 
        precision=args.precision, 
        default_root_dir=chk_pnt, 
        callbacks=checkpoint_callback, 
        logger=logger
    )
        
        # fit the model
        # Uncomment for Mac
        # pl_trainer = Trainer(max_epochs = 1, accelerator='cpu', devices=1, default_root_dir=EDGECNN_CHK_PNT)
        pl_trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
    elif args.predict_ckpt is not None:
        logger = TensorBoardLogger(save_dir=log, name=f"{args.num_epoch}_epoch_{args.model_name}", version=args.version)
        test_dataloader = DataLoader(test_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
        pl_trainer = Trainer(
        accelerator='gpu', 
        devices=1,
        precision=args.precision,
        logger=logger
    )
        model = GNNTrainer.load_from_checkpoint(args.predict_ckpt)  
    
    #Predictions
    y_pred = pl_trainer.predict(model=model, dataloaders=test_dataloader)
    y_pred = torch.cat(y_pred)
    print(len(y_pred))

    #Save the predictions
    torch.save(y_pred, f"{preds}/predictions_{args.version}.pt")

if __name__ == "__main__":
    main()
    make_submission_file(
       '.\experiments\predictions\edgecnn\dms\predictions_1.pt',
       '.\experiments\predictions\edgecnn\2a3\predictions_1.pt'
    )
