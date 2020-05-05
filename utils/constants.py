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
INPUTS = 'inputs'
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
POSTPROCS = 'postprocs'

PARAMS = 'params'
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

RESULTS_DIR = os.path.join(os.getcwd(), 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
EXPERIMENT_NAME = '{encoder_name}-{decoder_name}-{imsize}'
LOSS_STR = "Epoch-{epoch}: Loss: {loss} "
TASK_LOSS = "{}_loss: {} "
EPOCH_STR = 'Epoch {epoch}'
TRAIN_STR = 'Training'
VAL_STR = 'Validation'
METRICS_STR = 'Metrics'
TASK_STR = '{task}'
VAL_LOSS_STR = "Epoch: {epoch} Validation_loss: {loss} "
PATIENCE_STR = 'Patience of {patience} epochs reached.'
EARLY_STOP_STR = 'Early Stopping after {epoch} epochs: '
BATCH_LOG = """Epoch-{}, Batch-{}/{}: Total: {:05.3f}, Loader: {:05.3f}, Forward: {:05.3f}, Backward: {:05.3f}, Loss: {:05.3f}, PostProc: {:05.3f}, Metrics: {:05.3f}, Writer: {:05.3f}"""


PNG = '.png'
PKL = '.pkl'
TRAIN = 'train'
VAL = 'val'

HWC = 'HWC'
CHW = 'CHW'
DOUBLE = 'double'
LONG = 'long'
FLOAT = 'float'

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
INSTANCE = 'instance'
DISPARITY = 'disparity'
PANOPTIC = 'panoptic'
INSTANCE_SEG = 'instance_seg'
INSTANCE_CONTOUR = 'instance_contour'
INSTANCE_REGRESSION = 'instance_regression'
INSTANCE_HEATMAP = 'instance_heatmap'
INSTANCE_PROBS = 'instance_probs'
INSTANCE_IMAGE = 'instance_image'
PANOPTIC_IMAGE = 'panoptic_image'
SEGMENT_INFO = 'segment_info'

ALL_INSTANCE_TASKS = [INSTANCE_CONTOUR, INSTANCE_REGRESSION,
                      INSTANCE_PROBS, INSTANCE_HEATMAP]

OUTPUTS_TO_TASK = {SEMANTIC: SEMANTIC,
                   INSTANCE: ALL_INSTANCE_TASKS,
                   DISPARITY: DISPARITY}


MILLION = 1e7
START_ITER = 0
PLATEAU_COUNT = 0
VOID = 255
STATE = None
LENGTH = 150


GPU_STR = 'cuda:{gpu_id}'
CPU = 'cpu'

ARGMAX = 'argmax'
