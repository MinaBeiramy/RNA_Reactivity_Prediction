import argparse
from config import *
import torch
from dataloaders.crnn_dataloader import *
from trainers.cnn_trainer import CNNTrainer
from torch.utils.data import random_split, DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle

def main():
    # Argparser
    parser = argparse.ArgumentParser(description='Train Script for RNA REACTIVITY')
    parser.add_argument('-m', '--model_name', help='Choose from existing models, ex:crnn', required=True)
    parser.add_argument('-e', '--num_epoch', help='Number of Epochs', default=50)
    parser.add_argument('-v', '--version', help='Your model version number for logging and saving')
    parser.add_argument('-b', '--batch_size', help='Batch size for train-test', default=16)
    parser.add_argument('-p', '--precision', help='Train Precision Type', default='16-mixed')
    parser.add_argument('-d', '--dataset_type', help='DMS or 2A3', required=True)
    parser.add_argument('--predict_ckpt', help='Model ckpt for prediction', default=None)

    # # TODO: ADD MORE arguments as we go
    args = parser.parse_args()

    if args.dataset_type == '2a3':
        log = CRNN_2A3_LOG
        chk_pnt = CRNN_2A3_CHK_PNT
        train_parquet = P_TRAIN_2A3_PARQUET
        preds = CRNN_2A3_PRED
    
    elif args.dataset_type == 'dms':
        log = CRNN_DMS_LOG
        chk_pnt = CRNN_DMS_CHK_PNT
        train_parquet = P_TRAIN_DMS_PARQUET
        preds = CRNN_DMS_PRED

    
    # # Based on model , Load the Dataset

    if args.model_name == 'crnn':
        model = CNNTrainer("crnn")
        train_val_dataset = ParquetCRNNDataset(train_parquet)
        test_dataset = InferenceParquetCRNNDataset(P_TEST_PARQUET)
       
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

        # Chose the Trainer and the model you want to train


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
        # pl_trainer = Trainer(max_epochs = 1, accelerator='cpu', devices=1, default_root_dir=CRNN_CHK_PNT)
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
        model = CNNTrainer.load_from_checkpoint(args.predict_ckpt)  
    
    #Predictions
    
    batches = pl_trainer.predict(model=model, dataloaders=test_dataloader)
    y_pred = [i[0] for i in batches]
    seq_lens = [i[1] for i in batches]

    y_pred = torch.cat(y_pred)
    seq_lens = torch.cat(seq_lens)
    #print(len(y_pred))

    #Save the predictions
    torch.save(y_pred, f"{preds}/predictions_{args.version}.pt")
    torch.save(seq_lens, f"{preds}/seq_lens_{args.version}.pt")


if __name__ == "__main__":
    main()
    # python crnn_script.py -m crnn -e 1 -v 1 -d dms -b 64 -p 16-mixed
    # python crnn_script.py -m crnn -e 1 -v 1 -d dms --predict_ckpt "experiments\checkpoints\crnn\dms\version_1\best-epoch=0-val_loss=0.02.ckpt" -b 64 -p 16-mixed
