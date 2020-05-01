"""Constants utilized by other utility scripts."""

import os


# Constants related to config settings
MODEL = 'model'
ARCH = 'arch'
ENCODER = 'encoder'
DECODER = 'decoder'
PRETRAINED_PATH = 'pretrained_path'
ROOT_PATH = 'root_path'
LOSS_FN = 'loss_fn'
OUTPUTS = 'outputs'
ACTIVE = 'active'
OUT_CHANNELS = 'out_channels'
LOSS = 'loss'
LOSS_WEIGHT = 'loss_weight'
METRIC = 'metric'
POSTPROC = 'postproc'
TASKS = 'tasks'
TYPE = 'type'
DATA = 'data'
DATASET = 'dataset'
IM_SIZE = 'im_size'
SAVE_LOGS = 'savelogs'
LOGS = 'logs'

PARAMS = 'prams'
EPOCHS = 'epochs'
PATIENCE = 'patience'
EARLY_STOP = 'early_stop'
GPU_ID = 'gpu_id'
MULTIGPU = 'multigpu'
PRINT_INTERVAL = 'print_interval'
RESUME = 'resume'

MODEL_STATE = 'model_state'
OPTIMIZER_STATE = 'optimizer_state'
CRITERION_STATE = 'criterion_state'
EPOCH = 'epoch'
BEST_LOSS = 'best_loss'
OPTIMIZER = 'optimizer'

RESULTS_DIR = os.path.join(os.path.expanduser('~'), 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
EXPERIMENT_NAME = '{encoder_name}-{decoder_name}-{imsize}'
LOSS_STR = "epoch: {epoch} batch: {batch} loss: {loss}"
TASK_LOSS = " {task}_loss: {loss}"
EPOCH_STR = '\n******** Epoch {epoch} *********'
TRAIN_STR = '************* Training ***********'
VAL_STR = '\n************ Validation **********'
METRICS_STR = '\n************* Metrics ************'
TASK_STR = '*********** {task} **********'
VAL_LOSS_STR = "\nepoch: {epoch} validation_loss: {loss}"
PATIENCE_STR = 'Patience of {patience} epochs reached.'
EARLY_STOP_STR = 'Early Stopping after {epoch} epochs: '
PNG = '.png'
PKL = '.pkl'
TRAIN = 'train'
VAL = 'val'

HWC = 'HWC'
CHW = 'CHW'

GT = 'gt'
DET = 'det'

INPUT_IMAGE = 'Input_Image'
IMAGES = 'Images'

INPUT_WRITER_NAME = 'Images/{state}/RGB'
INPUT_SAVE_NAME = '{epoch}_RGB_{state}'
OUTPUT_WRITER_NAME = 'Images/{state}/{data_type}/{task}'
OUTPUT_SAVE_NAME = '{epoch}_{data_type}_{task}_{state}'
MODEL_NAME = '{time_stamp}_best-loss_{best_loss}.pkl'

SEMANTIC = 'semantic'
INSTANCE_CONTOUR = 'instance_contour'
INSTANCE_REGRESSION = 'instance_regression'
INSTANCE_HEATMAP = 'instance_heatmap'
INSTANCE_PROBS = 'instance_probs'

MILLION = 1e7
START_ITER = 0
PLATEAU_COUNT = 0
STATE = None


GPU_STR = 'cuda:{gpu_id}'
CPU = 'cpu'
