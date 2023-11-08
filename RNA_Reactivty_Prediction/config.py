# To Add Paths and Configs

DATA_DIR  = "/DATA/stanford-ribonanza-rna-folding/"
SILICO_CSVS = DATA_DIR + '/' + 'supplementary_silico_predictions'


TRAIN_CSV = DATA_DIR + '/' + "train_data.csv"
GPN15K_CSV = SILICO_CSVS + '/' + "GPN15k_silico_predictions.csv"
PK90_CSV = SILICO_CSVS + '/' + "PK90_silico_predictions.csv"
PK50_CSV = SILICO_CSVS + '/' + "PK50_silico_predictions.csv"
R1_CSV = SILICO_CSVS + '/' + "R1_silico_predictions.csv"


TRAIN_PARQUET_FILE = "train_data.parquet"
TEST_CSV = DATA_DIR + '/' + "test_sequences.csv"  
PK50_PARQUET_FILE = "pk50.parquet"
GPN15K_PARQUET_FILE = "gpn15k.parquet"
PK90_PARQUET_FILE = "pk90.parquet"
R1_PARQUET_FILE = "r1.parquet"
TEST_PARQUET_FILE = "test_sequences.parquet"

PRED_CSV = "submission.csv"
