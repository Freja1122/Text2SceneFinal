NEG_INF = -1e9
BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
UNK = "<unk>"
INIT_WORD_DICT = {BOS: 0, EOS: 1, PAD: 2, UNK: 3}
METRIC_SIGN = {"loss": -1, "ppl": -1, "co_set_rate": -1, "real_recall": -1}
HUMAN_PIC_IDX = [len(INIT_WORD_DICT), len(INIT_WORD_DICT) + 1]
HUMAN_IDX_LIST = [2, 3]
GRID_SIZE = 28
N_VOCABULARY = -1
LENGTH_BOUNDARY = 200
DEBUG = False
DPI = 120
'''change'''
VISUALIZE = False
DTYPES = ["train"]
# DTYPES = ["valid"]
# DTYPES = ["test"]
MODEL_DIR = "models20190124.172753"  # bibi
GLOBAL_STEP = 76000  # when USEBESTSTEP is not True
USEBESTSTEP = False
TRAIN_INFO = "att"
GPU_ID = "4"
SHOW_PROFILE = False
MAX_SNAPSHOT_STEP = 1000
TEST_DEL_DUP = False
CLOSE_OBJ = False
'''change over'''
MAX_LEN, N_PLOTS, FONT_SIZE = 40, 1, 8
DATA_NAME = "AS"
CONFIG_NAME = "basic"
ATT_CHANNEL_DIC = {'location': 1, 'z': 3, 'flip': 2, 'pose': 7, 'expression': 5}
GRID_TYPE = 'gridx'
BACKGROUND_IDS = [-1, -1]
# att_one_hot_idx = [1,3,2,7,5]
DIM_CAT = {'grid': -2, 'grid_x': -2, 'grid_y': -2, 'location': -2, 'z': -2, 'flip': -2, 'pose': -2, 'expression': -2,
           'type_logits': -2, 'type_samples': -1, 'type': 0}
