
import os
import re
import numpy as np
import pandas as pd
from config import  *
import polars as pl  # 📊 Polars for data manipulation

#TODO make a script to download the data


def list_files_in_directory(read_path:str=ETERNA_PKG_BPP, save_path:str=P_BPP_CSV):

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
    #matrix_df.drop_duplicates(subset=['sequence_id'], inplace=True)
    matrix_df.to_csv(save_path)
    print('********* BPP path files have been successfully saved in /DATA/preprocessed/p_bpp.csv**********')
    #return matrix_df


def setup_directories():
  
    # Directory 
    directory = "DATA"
    
    # Parent Directory path 
    parent_dir = os.getcwd()
    
    # Path 
    path = os.path.join(parent_dir, directory) 
    if not os.path.exists(path):
    
        os.mkdir(path) 
        print("Directory '% s' created" % directory) 
        
        # Directory 
        directory = "preprocessed"

        # Path 
        path = os.path.join(path, directory) 
        os.mkdir(path)
        print("Directory '% s' created" % directory) 

def to_parquet(read_path:str, save_path:str, bpp_path:str=None, dataset_type='2a3'):
    # 📊 Read CSV data using Polars
    dummy_df = pl.scan_csv(read_path)

    if bpp_path:
        bpp_df = pl.scan_csv(bpp_path)

    # 🔍 Define a new schema mapping for specific columns
    new_schema = {}
    for key, value in dummy_df.schema.items():
        if key.startswith("reactivity"):
            new_schema[key] = pl.Float32  # 📊 Convert 'reactivity' columns to Float32
        else:            
            new_schema[key] = value

    df = pl.scan_csv(read_path, schema=new_schema)
    if 'SN_filter' in df.columns:
        df = df.filter(pl.col('SN_filter') == 1) 
        #df = df.filter(pl.col('experiment_type') == 'DMS_MaP')
        #df = df.filter(pl.col('experiment_type') == '2A3_MaP')

    #df = df.unique(subset=["sequence_id"])
    df = df.with_columns(seq_len = pl.col("sequence").str.len_bytes().alias("seq_lengths"))
    df = df.join(bpp_df, on='sequence_id', how='left')
    

    # 💾 Write data to Parquet format with specified settings
    df.sink_parquet(
        save_path,
        compression='uncompressed',  # No compression for easy access
        row_group_size=10,  # Adjust row group size as needed
    )

def separate_data(csv_path:str=TRAIN_CSV, test_train_flag:str='train'):
    ## Get Clean Data For Main train.csv file
    ## Storing Reactivity columns as Targets and the Rests as Train Data
    ## Creating a Separate file for sequences in Train.csv, incase we only want to train on sequences and not the other attributes
    ## Filter the dataframe by 'SN_filter' column where the value is 1.0PP

    if test_train_flag == 'train':
        train_data = pd.read_csv(csv_path)
        print("The Shape of dataset before dropping the duplicates are: ", train_data.shape)
        train_data.drop_duplicates(subset=['sequence_id'], inplace=True)
        print("The Shape of dataset after dropping the duplicates are: ", train_data.shape)
        train_data = train_data[train_data["SN_filter"] == 1.0]
        print("The Shape of dataset after dropping SN filters !=1 is: ", train_data.shape)
        train_data['seq_len'] = train_data["sequence"].apply(len)
        reactivity_pattern = re.compile('(reactivity_[0-9])')
        reactivity_col_names = [col for col in train_data.columns if(reactivity_pattern.match(col))]
        targets = train_data[reactivity_col_names].fillna(0)
        targets['sequence_id'] = train_data['sequence_id']
        targets[reactivity_col_names].to_numpy(dtype=np.float32)
        train_data = train_data.drop(columns=reactivity_col_names)
        train_data.to_csv(P_TRAIN_CSV)
        targets.to_csv(P_TARGETS_CSV)
        print('shape of train data is: ', train_data.shape)
        print('shape of target is : ',  targets.shape)
        print('********** TARGETS and TRAIN DATA are seperated SUCESSFULLY.*******************')
        print('---------------Check /DATA/preprocessed for the new files.-------------------')

    elif test_train_flag == 'test':
        test_data = pd.read_csv(csv_path)
        print("The Shape of dataset before dropping the duplicates are: ", test_data.shape)
        test_data.drop_duplicates(subset=['sequence_id'], inplace=True)
        print("The Shape of dataset after dropping the duplicates are: ", test_data.shape)
        test_data['seq_len'] = test_data["sequence"].apply(len)
        test_data.drop(columns=['id_min', 'id_max'], axis=1, inplace=True)
        test_data.to_csv(P_TEST_CSV)
        print('shape of test data is: ', test_data.shape)
        print('********** TEST DATA are seperated SUCESSFULLY.*******************')
        print('---------------Check /DATA/preprocessed for the new files.-------------------')



if __name__ == "__main__":
    ## Uncomment steps if necessary
    #step 1:
    #
    #setup_directories()
    #list_files_in_directory()
    to_parquet(read_path=TRAIN_CSV, save_path=P_TRAIN_PARQUET, bpp_path=P_BPP_CSV)
    #to_parquet(read_path=TEST_CSV, save_path=P_TEST_PARQUET, bpp_path=P_BPP_CSV)
    #to_parquet(read_path=TRAIN_CSV, save_path=P_TRAIN_PARQUET_QUICK, bpp_path=P_BPP_CSV)