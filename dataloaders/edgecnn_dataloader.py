import re
import torch
import numpy as np
import polars as pl
from torch_geometric.data import Data, Dataset
## pytorch geometric .loader for loading graph data in batch
from sklearn.preprocessing import OneHotEncoder

class SimpleGraphDataset(Dataset):
    def __init__(self, parquet_name, edge_distance=5, root=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # 📄 Set the Parquet file name
        self.parquet_name = parquet_name
        # 📏 Set the edge distance for generating the adjacency matrix
        self.edge_distance = edge_distance
        # 🧮 Initialize the one-hot encoder for node features
        self.node_encoder = OneHotEncoder(sparse_output=False, max_categories=5)
        # 🧮 Fit the one-hot encoder to possible values (A, G, U, C)
        self.node_encoder.fit(np.array(['A', 'G', 'U', 'C']).reshape(-1, 1))
        # 📊 Load the Parquet dataframe
        self.df = pl.read_parquet(self.parquet_name)

        # 🧬 Get reactivity column names using regular expression
        reactivity_match = re.compile('(reactivity_[0-9])')
        reactivity_names = [col for col in self.df.columns if reactivity_match.match(col)]
        # 📊 Select only the reactivity columns
        self.reactivity_df = self.df.select(reactivity_names)
        # 📊 Select the 'sequence' column
        self.sequence_df = self.df.select(["sequence", "seq_len"])

    def parse_row(self, idx):
        # 📊 Read the row at the given index
        sequence_row = self.sequence_df.row(idx)[0]
        seq_len_row = self.sequence_df.row(idx)[1]
        reactivity_row = self.reactivity_df.row(idx)
        # 🧬 Get the sequence string and convert it to an array
        sequence = np.array(list(sequence_row)).reshape(-1, 1)
        # 🧬 Encode the sequence array using the one-hot encoder
        encoded_sequence = self.node_encoder.transform(sequence)

        # 📊 Get the edge index using nearest adjacency function
        edges_np = self.nearest_adjacency(seq_len_row, False)
        # 📏 Convert the edge index to a torch tensor
        edge_index = torch.tensor(edges_np, dtype=torch.long)
        # 🧬 Get reactivity targets for nodes
        reactivity = np.array(reactivity_row, dtype=np.float32)[0:seq_len_row]
        # 🔒 Create valid masks for nodes
        valid_mask = np.argwhere(~np.isnan(reactivity)).reshape(-1)
        torch_valid_mask = torch.tensor(valid_mask, dtype=torch.long)
        # 🧬 Replace nan values for reactivity with 0.0 (not super important as they get masked)
        reactivity = np.nan_to_num(reactivity, copy=False, nan=0.0)
        # 📊 Define node features as the one-hot encoded sequence
        node_features = torch.Tensor(encoded_sequence)
        # 🎯 Define targets
        targets = torch.Tensor(reactivity)
        # 📊 Create a PyTorch Data object
        data = Data(x=node_features, edge_index=edge_index, y=targets, valid_mask=torch_valid_mask)
        return data
        #return node_features, edge_index, targets, torch_valid_mask

    def len(self):
        # 📏 Return the length of the dataset
        return len(self.df)

    def get(self, idx):
        # 📊 Get and parse data for the specified index
        data = self.parse_row(idx)
        return data
    
    def nearest_adjacency(self, sequence_length , loops):
        base = np.arange(sequence_length)
        connections = []
        n=self.edge_distance
        
        for i in range(-n, n + 1):
            if i == 0 and not loops:
                continue
            elif i == 0 and loops:
                stack = np.vstack([base, base])
                connections.append(stack)
                continue

            # 🔄 Wrap around the sequence for circular connections
            neighbours = base.take(range(i, sequence_length + i), mode='wrap')
            stack = np.vstack([base, neighbours])

            # Separate connections for positive and negative offsets
            if i < 0:
                connections.append(stack[:, -i:])
            elif i > 0:
                connections.append(stack[:, :-i])

        # Combine connections horizontally
        return np.hstack(connections)

class InferenceGraphDataset(Dataset):
    def __init__(self, parquet_name, edge_distance=2, root=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # 📄 Set the Parquet file name
        self.parquet_name = parquet_name
        # 📏 Set the edge distance for generating the adjacency matrix
        self.edge_distance = edge_distance
        # 🧮 Initialize the one-hot encoder for node features
        self.node_encoder = OneHotEncoder(sparse_output=False, max_categories=4)
        # 🧮 Fit the one-hot encoder to possible values (A, G, U, C)
        self.node_encoder.fit(np.array(['A', 'G', 'U', 'C']).reshape(-1, 1))
        # 📊 Load the Parquet dataframe
        self.df = pl.read_parquet(self.parquet_name)
        # 📊 Select the 'sequence' and 'id_min' columns
        self.sequence_df = self.df.select(["sequence", "seq_len"])
        self.id_min_df = self.df.select("id_min")

    def parse_row(self, idx):
        # 📊 Read the row at the given index
        sequence_row = self.sequence_df.row(idx)[0]
        seq_len_row = self.sequence_df.row(idx)[1]
        id_min = self.id_min_df.row(idx)[0]

        # 🧬 Get the sequence string and convert it to an array
        sequence = np.array(list(sequence_row)).reshape(-1, 1)
        # 🧬 Encode the sequence array using the one-hot encoder
        encoded_sequence = self.node_encoder.transform(sequence)
        # 📏 Get the sequence length
        sequence_length = len(sequence)
        # 📊 Get the edge index using nearest adjacency function
        edges_np = self.nearest_adjacency(seq_len_row, False)
        # 📏 Convert the edge index to a torch tensor
        edge_index = torch.tensor(edges_np, dtype=torch.long)

        # 📊 Define node features as the one-hot encoded sequence
        node_features = torch.Tensor(encoded_sequence)
        ids = torch.arange(id_min, id_min+seq_len_row, 1)

        data = Data(x=node_features, edge_index=edge_index, ids=ids)
        return data
        #return node_features, edge_index, ids


    def len(self):
        # 📏 Return the length of the dataset
        return len(self.df)

    def get(self, idx):
        # 📊 Get and parse data for the specified index
        data = self.parse_row(idx)
        return data
    
    def nearest_adjacency(self, sequence_length, loops):
        base = np.arange(sequence_length)
        connections = []
        n = self.edge_distance
        for i in range(-n, n + 1):
            if i == 0 and not loops:
                continue
            elif i == 0 and loops:
                stack = np.vstack([base, base])
                connections.append(stack)
                continue

            # 🔄 Wrap around the sequence for circular connections
            neighbours = base.take(range(i, sequence_length + i), mode='wrap')
            stack = np.vstack([base, neighbours])

            # Separate connections for positive and negative offsets
            if i < 0:
                connections.append(stack[:, -i:])
            elif i > 0:
                connections.append(stack[:, :-i])

        # Combine connections horizontally
        return np.hstack(connections)



# import os
# import argparse
# from ..config import *
# import torch
# from dataloaders.crnn_dataloader import *
# from trainers.cnn_trainer import CNNTrainer
# from trainers.gnn_trainer import GNNTrainer
# from torch.utils.data import random_split, DataLoader
# from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint

# model = GNNTrainer("edgecnn", edge_distance=4)
# train_val_dataset = SimpleGraphDataset(P_TRAIN_PARQUET_QUICK, edge_distance=4)
# test_dataset = InferenceGraphDataset(P_TEST_PARQUET, edge_distance=4)
# log = EDGECNN_LOG
# chk_pnt = EDGECNN_CHK_PNT

# generator1 = torch.Generator().manual_seed(42)
# train_dataset, val_dataset = random_split(train_val_dataset, [0.7, 0.3], generator1)
    
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)  
# val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)
# test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

# # Chose the Trainer and the model you want to train


# #Check point and logger
# checkpoint_callback = ModelCheckpoint(
#             dirpath=chk_pnt,
#             monitor='val_loss',
#             save_top_k=1,
#             filename='best-{epoch}-{val_loss:.2f}'
#         )

# logger = TensorBoardLogger(save_dir=log, name=f"{args.num_epoch}epoch_{args.model_name}", version=1)
# #logger = TensorBoardLogger(save_dir=log, name=f"1epoch_edgecnn", version=1)



# # fit the model
# pl_trainer = Trainer(max_epochs = 1, accelerator='gpu', devices=1, precision="16-mixed", default_root_dir=chk_pnt)
# pl_trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)