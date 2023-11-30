# To Add Paths and Configs

from torchvision.transforms import ToTensor, Resize

DATA_DIR  = "./DATA"
SILICO_CSVS = DATA_DIR + '/' + 'supplementary_silico_predictions'
PREPROSECESSED_DATA = "./DATA/preprocessed"
PREPROSECESSED_BPPS = PREPROSECESSED_DATA + '/' + 'bbps'
ETERNA_PKG_BPP = DATA_DIR + '/' +'Ribonanza_bpp_files/extra_data'
SUBMISSIONS = "./DATA/submissions"

TRAIN_CSV = DATA_DIR + '/' + "train_data.csv"
TEST_CSV = DATA_DIR + '/' + "test_sequences.csv"
GPN15K_CSV = SILICO_CSVS + '/' + "GPN15k_silico_predictions.csv"
PK90_CSV = SILICO_CSVS + '/' + "PK90_silico_predictions.csv"
PK50_CSV = SILICO_CSVS + '/' + "PK50_silico_predictions.csv"
R1_CSV = SILICO_CSVS + '/' + "R1_silico_predictions.csv"

P_TRAIN_CSV = PREPROSECESSED_DATA + '/' + "p_train_data.csv"
P_TARGETS_CSV = PREPROSECESSED_DATA + '/' + "p_targets_data.csv"
P_BPP_CSV = PREPROSECESSED_DATA + '/' + "p_bpp.csv"
P_TEST_CSV = PREPROSECESSED_DATA + '/' + "test_data.csv"

#BPPs
FORGI_BPP_FILES = PREPROSECESSED_BPPS + '/' + "forgi.pt"


# In case of loading data with parquets, uncomment below:


P_TRAIN_PARQUET = PREPROSECESSED_DATA + '/' + "p_train_data.parquet"
P_TARGETS_PARQUET = PREPROSECESSED_DATA + '/' + "p_targets_data.parquet"
P_BPP_PARQUET = PREPROSECESSED_DATA + '/' + "p_bpp.parquet"
P_TEST_PARQUET = PREPROSECESSED_DATA + '/' + "test_data.parquet"

PRED_CSV = SUBMISSIONS + '/' + "submission.csv"


###### BBP DATA LOADER CONFIG
ETERNA_BBP_SUB_DIRECTORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
TRANSFORM = [ToTensor(), Resize((224, 224))]


####### TRAIN CONFIG
# epoch = 15
#     lr = config['lr']
#     show_interval = config['show_interval']
#     valid_interval = config['valid_interval']
#     save_interval = config['save_interval']
#     cpu_workers = config['cpu_workers']
#     reload_checkpoint = config['reload_checkpoint']
#     valid_max_iter = config['valid_max_iter']

#     img_width = config['img_width']
#     img_height = config['img_height']
#     data_dir = config['data_dir']


##### CHECK POINTS AND LOGGERS
CRNN_CHK_PNT = './experiments/crnn'
CRNN_LOG = './experiments/crnn/logs'