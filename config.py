# To Add Paths and Configs

from torchvision.transforms import ToTensor, Resize

DATA_DIR  = "./DATA/stanford-ribonanza-rna-folding"
SILICO_CSVS = DATA_DIR + '/' + 'supplementary_silico_predictions'
PREPROSECESSED_DATA = "./DATA/preprocessed"
PREPROSECESSED_BPPS = PREPROSECESSED_DATA + '/' + 'bbps'
ETERNA_PKG_BPP = DATA_DIR + '/' +'Ribonanza_bpp_files/extra_data'
SUBMISSIONS = "./DATA/submissions"

TRAIN_CSV = DATA_DIR + '/' + "train_data.csv"
GPN15K_CSV = SILICO_CSVS + '/' + "GPN15k_silico_predictions.csv"
PK90_CSV = SILICO_CSVS + '/' + "PK90_silico_predictions.csv"
PK50_CSV = SILICO_CSVS + '/' + "PK50_silico_predictions.csv"
R1_CSV = SILICO_CSVS + '/' + "R1_silico_predictions.csv"

P_TRAIN_CSV = PREPROSECESSED_DATA + '/' + "p_train_data.csv"
P_TARGETS_CSV = PREPROSECESSED_DATA + '/' + "p_targets_data.csv"
P_BPP_CSV = PREPROSECESSED_DATA + '/' + "p_bpp.csv"
TEST_CSV = PREPROSECESSED_DATA + '/' + "test_sequences.csv"

#BPPs
FORGI_BPP_FILES = PREPROSECESSED_BPPS + '/' + "forgi.pt"


# In case of loading data with parquets, uncomment below:

# TRAIN_PARQUET_FILE = "train_data.parquet"
# TEST_CSV = DATA_DIR + '/' + "test_sequences.csv"  
# PK50_PARQUET_FILE = "pk50.parquet"
# GPN15K_PARQUET_FILE = "gpn15k.parquet"
# PK90_PARQUET_FILE = "pk90.parquet"
# R1_PARQUET_FILE = "r1.parquet"
# TEST_PARQUET_FILE = "test_sequences.parquet"

PRED_CSV = SUBMISSIONS + '/' + "submission.csv"


###### BBP DATA LOADER CONFIG
ETERNA_BBP_SUB_DIRECTORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
TRANSFORM = [ToTensor(), Resize((224, 224))]
