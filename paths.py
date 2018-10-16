
# -------------------------------------------------------------------------------------------------
# paths for the new data splits
CHALEARN_TRAIN_SPLIT = '/scratch/users/gabras/data/chalearn20/train_split.txt'
CHALEARN_VAL_SPLIT = '/scratch/users/gabras/data/chalearn20/val_split.txt'
CHALEARN_TEST_SPLIT = '/scratch/users/gabras/data/chalearn20/test_split.txt'

CHALEARN_TRAIN_LABELS_20 = '/scratch/users/gabras/data/chalearn20/train_labels.h5'
CHALEARN_VAL_LABELS_20 = '/scratch/users/gabras/data/chalearn20/val_labels.h5'
CHALEARN_TEST_LABELS_20 = '/scratch/users/gabras/data/chalearn20/test_labels.h5'

CHALEARN_ALL_DATA_20_2 = '/scratch/users/gabras/data/chalearn20/all_data_2'

# unused
# this is not used, will be moved to local soon
CHALEARN_ALL_DATA_20 = '/scratch/users/gabras/data/chalearn20/all_data'

# unused
# these are not used, will be moved to local soon
CHALEARN_TRAIN_DATA_20 = '/scratch/users/gabras/data/chalearn20/train_data.h5'
CHALEARN_VAL_DATA_20 = '/scratch/users/gabras/data/chalearn20/val_data.h5'
CHALEARN_TEST_DATA_20 = '/scratch/users/gabras/data/chalearn20/test_data.h5'

TRAIN_UID_KEYS_MAPPING = '/scratch/users/gabras/data/chalearn20/train_uid_keys_mapping.h5'
TEST_UID_KEYS_MAPPING = '/scratch/users/gabras/data/chalearn20/test_uid_keys_mapping.h5'
VAL_UID_KEYS_MAPPING = '/scratch/users/gabras/data/chalearn20/val_uid_keys_mapping.h5'

NUM_FRAMES = '/scratch/users/gabras/data/chalearn20/id_frames.h5'

# -------------------------------------------------------------------------------------------------
# paths for original data
CHALEARN_TRAIN_LABELS_ORIGINAL = '/scratch/users/gabras/data/chalearn10/train_labels.pkl'
CHALEARN_VAL_LABELS_ORIGINAL = '/scratch/users/gabras/data/chalearn10/val_labels.pkl'
CHALEARN_TEST_LABELS_ORIGINAL = '/scratch/users/gabras/data/chalearn10/test_labels.pkl'

CHALEARN_TEST_GETH_LABELS_ORIGINAL = '/scratch/users/gabras/data/chalearn10/eth_gender_annotations_test.csv'
CHALEARN_DEV_GETH_LABELS_ORIGINAL = '/scratch/users/gabras/data/chalearn10/eth_gender_annotations_dev.csv'

# -------------------------------------------------------------------------------------------------
# stefan data
CHALEARN_FACES_TRAIN_H5 = '/scratch/users/steiac/steiac/deployed-3d/3DDIdata/data_preprocessed/'
CHALEARN_FACES_VAL_H5 = '/scratch/users/steiac/steiac/deployed-3d/3DDIdata/val_preprocessed/'
CHALEARN_FACES_TEST_H5 = '/scratch/users/gabras/data/chalearn10/test_preprocessed'

# -------------------------------------------------------------------------------------------------
# paths for processing the original test split
CHALEARN_TEST_ORIGINAL = '/scratch/users/gabras/data/chalearn10/original_test'
CHALEARN_FACES_TEST_TIGHT = '/scratch/users/gabras/data/chalearn10/test_face_tight_crop'

# -------------------------------------------------------------------------------------------------
# dummy data for performance testing
DUMMY_DATA_JPG = '/scratch/users/gabras/data/trying_out/training80_05'
SETUP1 = '/scratch/users/gabras/data/trying_out/setup1'
SETUP3 = '/scratch/users/gabras/data/trying_out/setup3'
SETUP4 = '/scratch/users/gabras/data/trying_out/setup4'

# -------------------------------------------------------------------------------------------------
# log files
LOG_BASE = '/scratch/users/gabras/data/loss/'
TRAIN_LOG = '/scratch/users/gabras/data/loss/train_64.txt'
VAL_LOG = '/scratch/users/gabras/data/loss/val_64.txt'
##
TEST_LOG = '/scratch/users/gabras/data/loss/test_84.txt'
PREDICTION_LOG = '/scratch/users/gabras/data/loss/pred_84.txt'

# -------------------------------------------------------------------------------------------------
# models
MODELS = '/scratch/users/gabras/data/models'

# -------------------------------------------------------------------------------------------------
# figures
FIGURES = '/home/gabras/deployed/deepimpression2/figures'
PAPER_PLOTS = '/home/gabras/deployed/deepimpression2/paper_plots'

# -------------------------------------------------------------------------------------------------
# original dataset
CHALEARN_TRAIN_ORIGINAL = '/scratch/users/gabras/data/chalearn10/original_train'
CHALEARN_VAL_ORIGINAL = '/scratch/users/gabras/data/chalearn10/original_val'

# -------------------------------------------------------------------------------------------------
# chalearn30
CHALEARN30_ALL_DATA = '/scratch/users/gabras/data/chalearn30/all_data'

# -------------------------------------------------------------------------------------------------
# list of encountered h5 that have zero frames
ZERO_FRAMES = '/scratch/users/gabras/data/misc/zero_frames.txt'