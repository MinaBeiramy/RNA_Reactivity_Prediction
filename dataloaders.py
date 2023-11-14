import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import ConstantPad1d, ConstantPad2d
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import os
from torchvision.transforms import ToTensor, Compose, Resize, transforms
from typing import Any
from config import ETERNA_BBP_SUB_DIRECTORIES, P_TRAIN_CSV, P_TARGETS_CSV, ETERNA_PKG_BPP


class CNNDataset(Dataset):
    def __init__(self, data_csv=P_TRAIN_CSV, target_csv=P_TARGETS_CSV, sub_directories=ETERNA_BBP_SUB_DIRECTORIES, root=None,  matrix_dir=ETERNA_PKG_BPP):
        #super().__init__(root)
        #  Set the csv file name
        self.transform = transforms.Compose([
            ToTensor(),
        ])
        self.max_len = 224
        self.reactivity_shape = 207
        self.train_data = data_csv
        self.targets = target_csv
        self.matrix_dir = matrix_dir
        self.sub_directories = sub_directories

        self.df = pd.read_csv(self.train_data)
        self.target_df = pd.read_csv(self.targets)

        self.sequence_df = self.df["sequence"]
        self.sequence_id_df = self.target_df["sequence_id"]

        self.matrix_df = self.list_files_in_directory()
        ## Drop the sequences not available in our original dataset( The ones we do not possess the ractivity for)
        #unavailables =  list(set(self.matrix_df['sequence_id'].to_list()) - set(self.targets['sequence_id'].to_list()))
        self.matrix_df['av'] = self.matrix_df['sequence_id'].isin(self.target_df["sequence_id"])
        self.matrix_df = self.matrix_df[self.matrix_df['av'] == True].drop(columns=['av'])



    def __getitem__(self, idx) -> Any:
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

        
        self.reactivity = torch.tensor(self.target_df.iloc[idx, :-1].values.astype('float32')).flatten()

        self.reactivity = target_pad_opperation(self.reactivity)
        #print('Shape of reactivity: ', self.reactivity.shape)

        #self.matrix_path = self.matrix_df[self.matrix_df['sequence_id'] == str(self.sequence_id)]['path'][0]
        self.matrix_path = self.matrix_df.loc[self.matrix_df['sequence_id'] == str(self.sequence_id), 'path'].values[0]
        self.bpp = torch.tensor(self.load_bpp_to_nparray(self.matrix_path).astype('float32'))

        # Print
        # print('after function shape:', self.bpp.shape)
        # print('Shape of matrix: ', self.bpp.shape)

        return self.bpp, self.reactivity

    def __len__(self) -> int:
        return len(self.target_df)
    
    def list_files_in_directory(self):

        file_paths = []
        file_names = []
        for root, dirs, files in os.walk(ETERNA_PKG_BPP):
            for file in files:
                if file.endswith('.txt'):
                    path = os.path.join(root, file)
                    normalised = os.path.normpath(path)
                    file_paths.append(normalised)
                    # getting the name of sequence 
                    file_names.append(file[:-4])

        matrix_df = pd.DataFrame(columns=['sequence_id', 'path'])
        matrix_df['sequence_id'] = file_names
        matrix_df['path'] = file_paths
        matrix_df.drop_duplicates(subset=['sequence_id'], inplace=True)
        return matrix_df
    
    def load_bpp_to_nparray(self, bpp_file):

        # Load the data from the file
        data = np.loadtxt(bpp_file)

        # Create an empty array filled with zeros
        filled_array = np.zeros((self.max_len, self.max_len))

        # Fill the values from the loaded data into the empty array
        for row, col, value in data:
            filled_array[int(row) - 1, int(col) - 1] = value
        
        return filled_array
