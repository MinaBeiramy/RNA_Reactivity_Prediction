
import re
import pandas as pd
from config import TRAIN_CSV, P_TARGETS_CSV, P_TRAIN_CSV

def separate_data(csv_path:str=TRAIN_CSV):
    ## Get Clean Data For Main train.csv file
    ## Storing Reactivity columns as Targets and the Rests as Train Data
    ## Creating a Separate file for sequences in Train.csv, incase we only want to train on sequences and not the other attributes
    train_data = pd.read_csv(csv_path)
    reactivity_pattern = re.compile('(reactivity_[0-9])')
    reactivity_col_names = [col for col in train_data.columns if(reactivity_pattern.match(col))]
    targets = train_data[reactivity_col_names]
    train_data = train_data.drop(reactivity_col_names)
    train_data.write_csv(P_TRAIN_CSV)
    targets.write_csv(P_TARGETS_CSV)


if __name__ == "__main__":
    separate_data()

