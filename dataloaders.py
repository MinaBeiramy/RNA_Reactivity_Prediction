from torch.utils.data import Dataset, DataLoader
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
        self.train_data = data_csv
        self.targets = target_csv
        self.matrix_dir = matrix_dir
        self.sub_directories = sub_directories

        self.df = pd.read_csv(self.train_data)
        self.target_df = pd.read_csv(self.targets).fillna(0.0)

        self.sequence_df = self.df["sequence"]
        self.sequence_id_df = self.target_df["sequence_id"]

        self.matrix_df = self.list_files_in_directory()
        ## Drop the sequences not available in our original dataset( The ones we do not possess the ractivity for)
        #unavailables =  list(set(self.matrix_df['sequence_id'].to_list()) - set(self.targets['sequence_id'].to_list()))
        self.matrix_df['av'] = self.matrix_df['sequence_id'].isin(self.target_df["sequence_id"])
        self.matrix_df = self.matrix_df[self.matrix_df['av'] == True].drop(columns=['av'])
        print(self.target_df["sequence_id"].shape, '\n', self.matrix_df.shape)
        print(self.matrix_df.head())



    def __getitem__(self, idx) -> Any:
        self.sequence_id = self.target_df.iloc[idx, -1]
        print(self.sequence_id, type(self.sequence_id))
        self.reactivity = self.target_df.iloc[idx, :-1].values

        #self.matrix_path = self.matrix_df[self.matrix_df['sequence_id'] == str(self.sequence_id)]['path'][0]
        self.matrix_path = self.matrix_df.loc[self.matrix_df['sequence_id'] == str(self.sequence_id), 'path'].values[0]
        print(self.matrix_path[0], type(self.matrix_path[0]))
        self.bpp_np = self.load_bpp_to_nparray(self.matrix_path)
        print('before:', self.bpp_np)

        data_transform=transforms.Compose([
            ToTensor(), Resize((224,224))
        ])
        self.bpp = data_transform(self.bpp_np)
        print('After Transform', self.bpp)

        target_transform = transforms.Compose([
            ToTensor(),
            Resize( (224,-1))
        ])

        self.reactivity = target_transform(self.reactivity)
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

        # Determine the shape of the array
        max_row = int(data[:, 0].max())
        max_col = int(data[:, 1].max())

        # Create an empty array filled with zeros
        filled_array = np.zeros((max_row, max_col))

        # Fill the values from the loaded data into the empty array
        for row, col, value in data:
            filled_array[int(row) - 1, int(col) - 1] = value
        
        return filled_array
