
import re
import numpy as np
import pandas as pd
from config import TRAIN_CSV, P_TARGETS_CSV, P_TRAIN_CSV, P_BPP_CSV, P_TEST_CSV, TEST_CSV

#TODO make a script to download the data


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


# TODO get base pair probabilities of sequences and store them in 
def clean_eterana_bpp_files(P_BPP_CSV):
    pass

if __name__ == "__main__":
    ## Uncomment steps if necessary

    #Step 2: 
    #separate_data()
    separate_data(test_train_flag='test', csv_path=TEST_CSV)

    #Step 3:
    #get_bpp()
