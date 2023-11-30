import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import ConstantPad1d, ConstantPad2d
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import polars as pl
import os
import re
from torchvision.transforms import ToTensor, Compose, Resize, transforms
from typing import Any
from config import *


class CNNDataset(Dataset):
    def __init__(
            self, 
            train_test_flag:str='train', 
            data_csv=P_TRAIN_CSV, 
            target_csv=P_TARGETS_CSV,
            bpp_csv= P_BPP_CSV,
            sub_directories=ETERNA_BBP_SUB_DIRECTORIES, 
            root=None,  
            matrix_dir=ETERNA_PKG_BPP
        ):
        #super().__init__(root)
        #  Set the csv file name
        self.transform = transforms.Compose([
            ToTensor(),
        ])
        self.train_test_flag = train_test_flag
        self.max_len = 224
        self.data_csv = data_csv
        self.df = pd.read_csv(self.data_csv)
        self.sequence_df = self.df["sequence"]
        self.matrix_dir = matrix_dir
        self.sub_directories = sub_directories
        self.matrix_df = bpp_csv


        if self.train_test_flag == 'train':
            self.targets_csv = target_csv
            self.target_df = pd.read_csv(self.targets_csv)
            self.reactivity_shape = 207

            self.sequence_id_df = self.target_df["sequence_id"]

            ## Drop the sequences not available in our original dataset( The ones we do not possess the ractivity for)
            #unavailables =  list(set(self.matrix_df['sequence_id'].to_list()) - set(self.targets['sequence_id'].to_list()))
            self.matrix_df['av'] = self.matrix_df['sequence_id'].isin(self.target_df["sequence_id"])
            self.matrix_df = self.matrix_df[self.matrix_df['av'] == True].drop(columns=['av'])
        
        elif self.train_test_flag == 'test':

            self.sequence_id_df = self.df["sequence_id"]

            ## Drop the sequences not available in our original dataset( The ones we do not possess the ractivity for)
            #unavailables =  list(set(self.matrix_df['sequence_id'].to_list()) - set(self.targets['sequence_id'].to_list()))
            self.matrix_df['av'] = self.matrix_df['sequence_id'].isin(self.df["sequence_id"])
            self.matrix_df = self.matrix_df[self.matrix_df['av'] == True].drop(columns=['av'])
            print('seq id: ', self.sequence_id_df.shape)
            print('mamtrix: ', self.matrix_df.shape)

            print(self.df.columns)



    def __getitem__(self, idx) -> Any:

        if self.train_test_flag == 'train':
            self.sequence_id = self.target_df.iloc[idx, -1]
            #getting len of sequence to pad the sequence accordingly
            self.target_pad = int((self.max_len - self.reactivity_shape) / 2)

            # Print
            # print(self.target_pad)
            # print(self.target_df.iloc[idx, :-1].values.astype('float32').shape)
            
            if self.target_pad % 2 == 1:
                target_pad_opperation = ConstantPad1d((self.target_pad, self.target_pad +1), 0.0)
            elif self.target_pad % 2 == 0:
                target_pad_opperation = ConstantPad1d(self.target_pad, 0.0)

            
            self.reactivity = torch.Tensor(self.target_df.iloc[idx, :-1].values.astype('float32')).flatten()

            self.reactivity = target_pad_opperation(self.reactivity)
            #print('Shape of reactivity: ', self.reactivity.shape)

            #self.matrix_path = self.matrix_df[self.matrix_df['sequence_id'] == str(self.sequence_id)]['path'][0]
            self.matrix_path = self.matrix_df.loc[self.matrix_df['sequence_id'] == str(self.sequence_id), 'path'].values[0]
            self.bpp = torch.Tensor(self.load_bpp_to_nparray(self.matrix_path).astype('float32'))
            self.bpp = torch.unsqueeze(self.bpp, 0)

            # Print
            # print('after function shape:', self.bpp.shape)
            # print('Shape of matrix: ', self.bpp.shape)

            return self.bpp, self.reactivity
        
        elif self.train_test_flag == 'test':
            self.sequence_id = self.df.iloc[idx, 1]
            print('I am the seq id: ', self.sequence_id)
            print('I am the path: ', self.matrix_df.loc[self.matrix_df['sequence_id'] == str(self.sequence_id), 'path'])

            self.matrix_path = self.matrix_df.loc[self.matrix_df['sequence_id'] == str(self.sequence_id), 'path'].values[0]
            self.bpp = torch.Tensor(self.load_bpp_to_nparray(self.matrix_path).astype('float32'))
            self.bpp = torch.unsqueeze(self.bpp, 0)


            return self.bpp


    def __len__(self) -> int:
        if self.train_test_flag == 'train':
            return len(self.target_df)
        
        elif self.train_test_flag == 'test':
            return len(self.matrix_df)

    
    def load_bpp_to_nparray(self, bpp_file):

        # Load the data from the file
        data = np.loadtxt(bpp_file)

        # Create an empty array filled with zeros
        filled_array = np.zeros((self.max_len, self.max_len))

        # Fill the values from the loaded data into the empty array
        for row, col, value in data:
            filled_array[int(row) - 1, int(col) - 1] = value
        
        return filled_array



class SimpleParquetDataset(Dataset):
    def __init__(self, parquet_name, train_test_flag:str='train', root=None, transform=None, pre_transform=None, pre_filter=None):

        # Set the Parquet file name
        self.parquet_name = parquet_name
        self.train_test_flag = train_test_flag

        # Load the Parquet dataframe
        self.df = pl.read_parquet(self.parquet_name)

        if self.train_test_flag == 'train':
        # Get reactivity column names using regular expression
            reactivity_match = re.compile('(reactivity_[0-9])')
            reactivity_names = [col for col in self.df.columns if reactivity_match.match(col)]
            # Select only the reactivity columns
            self.reactivity_df = self.df.select(reactivity_names)
        
        # Select the 'sequence' column
        self.sequence_df = self.df.select(["sequence","path"])
        self.max_len = 224 # max len of seq is 207, therefore a fixed len will of 224 will be chosen for bpp matrix

    def __getitem__(self, idx):
        # Read the row at the given index
        sequence_row = self.sequence_df.row(idx)[0]
        bpp_path_row = self.sequence_df.row(idx)[1]

        sequence = np.array(list(sequence_row[0])).reshape(-1, 1)
        sequence_length = len(sequence)

        # Replace nan values for reactivity with 0.0 (not super important as they get masked)
        data = self.load_bpp_to_nparray(bpp_path_row).astype('float32')
        data = torch.Tensor(data)
        data = torch.unsqueeze(data, 0)
        
        # Define data and targets
        if self.train_test_flag =='train':
            reactivity_row = self.reactivity_df.row(idx)
            reactivity = np.array(reactivity_row, dtype=np.float32)[0:sequence_length]
            reactivity = np.nan_to_num(reactivity, copy=False, nan=0.0)
            targets = torch.Tensor(reactivity)

            return data, targets
        
        elif self.train_test_flag == 'test':
            return data

    def __len__(self):
        # 📏 Return the length of the dataset
        return len(self.sequence_df)

    def load_bpp_to_nparray(self, bpp_file):

        # Load the data from the file
        data = np.loadtxt(bpp_file)

        # Create an empty array filled with zeros
        filled_array = np.zeros((self.max_len, self.max_len))

        # Fill the values from the loaded data into the empty array
        for row, col, value in data:
            filled_array[int(row) - 1, int(col) - 1] = value
        
        return filled_array

    def get(self, idx):
        # 📊 Get and parse data for the specified index
        data = self.parse_row(idx)
        return data
    