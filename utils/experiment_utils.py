"""Helper functions for train/validation loop."""
import pdb
import io
import torch
import tensorflow as tf
import matplotlib
import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.misc

from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from texttable import Texttable


from utils.encoder_decoder import get_encoder_decoder
from utils.optimizers import get_optimizer
from utils.metrics import metrics
from utils.dataloader import get_dataloaders
from utils.loss_utils import loss_meters, MultiTaskLoss, averageMeter
from utils.im_utils import cat_labels, prob_labels, inst_labels, labels
from utils.writer_utils import add_images_to_writer
from utils.data_utils import get_labels, get_weights, get_predictions, \
    post_process_predictions
from utils.constants import TRAIN, VAL, EXPERIMENT_NAME, BEST_LOSS, MILLION, \
    START_ITER, PLATEAU_COUNT, STATE, GPU_STR, GPU_ID, CPU, PARAMS, MODEL, \
    ENCODER, DECODER, DATA, IM_SIZE, PRETRAINED_PATH, ROOT_PATH, RESULTS_DIR, \
    MULTIGPU, TASKS, MODEL_NAME, LOGS, SAVE_LOGS, PKL, EPOCHS, EARLY_STOP, \
    PATIENCE, PRINT_INTERVAL, RESUME, MODEL_STATE, OPTIMIZER_STATE, LOSS_FN, \
    CRITERION_STATE, EPOCH, EPOCH_STR, OUTPUTS, OPTIMIZER, LOSS_STR, \
    TASK_LOSS, TRAIN_STR, VAL_STR, VAL_LOSS_STR, METRICS_STR, PATIENCE_STR, \
    EARLY_STOP_STR, TASK_STR, SEMANTIC, INSTANCE_PROBS, INSTANCE_CONTOUR, \
    POSTPROC, POSTPROCS, PANOPTIC


class ExperimentLoop():
    """Loops experiment in train/val modes."""

    def __init__(self, cfg, mode=TRAIN):
        """Initialize ExperimentLoop class.

        Arguments:
            cfg {[dict]} -- [configuration settings for the experiment]

        Keyword Arguments:
            mode {[str]} -- [train/val mode to run experiment]
            (default: {TRAIN})
        """
        self.cfg = cfg
        self.mode = mode
        self.setup_experiment()

    def setup_experiment(self):
        """Set up required parameters, variables etc for experiment."""
        # Define Configuration parameters
        self.get_config_params()

        # Get device and writer
        self.device = get_device(self.cfg)
        self.writer = get_writer(self.cfg)

        # Get dataloaders, model, weights, optimizers and metrics
        self.dataloaders = get_dataloaders(self.cfg)
        self.model = get_model(self.cfg, self.device)  # TODO: print(model)
        self.weights = get_weights(self.cfg, self.device)
        self.criterion = self.get_criterion()
        self.optimizer = self.init_optimizer()
        self.get_losses_and_metrics()

        # Load Checkpoint
        self.load_checkpoint()

    def get_config_params(self):
        """Get configuration parameters for the experiment."""
        cfg = self.cfg
        params = cfg[PARAMS]
        self.params = params
        self.epochs = params[EPOCHS]
        self.patience = params[PATIENCE]
        self.early_stop = params[EARLY_STOP]
        self.print_interval = params[PRINT_INTERVAL]
        self.resume_training = params[RESUME]
        self.exp_dir, _, _ = get_exp_dir(cfg)
        self.plateau_count = 0

    def get_criterion(self):
        """Get criterion for the current experiment."""
        cfg = self.cfg
        loss_fn = cfg[MODEL][LOSS_FN]
        criterion = MultiTaskLoss(cfg[MODEL][OUTPUTS],
                                  self.weights, loss_fn)
        criterion = criterion.to(self.device)
        return criterion

    def init_optimizer(self):
        """Initialize the optimizer."""
        optimizer_cls = get_optimizer(self.params[TRAIN])
        optimizer_params = {k: v for k, v in
                            self.params[TRAIN][OPTIMIZER].items()
                            if k != "name"}
        criterion_params = list(self.criterion.parameters())
        if criterion_params is not None:
            model_params = list(self.model.parameters()) + criterion_params
        else:
            model_params = self.model.parameters()
        optimizer = optimizer_cls(model_params, **optimizer_params)

        return optimizer

    def get_losses_and_metrics(self):
        """Get loss and metric meters to log them during experiments."""
        out_model_cfg = self.cfg[MODEL][OUTPUTS]
        post_proc_cfg = self.cfg[POSTPROCS]

        self.train_loss_meters = loss_meters(out_model_cfg)
        self.val_loss_meters = loss_meters(out_model_cfg)
        self.val_metrics = metrics(out_model_cfg)
        self.post_proc_metrics = metrics(post_proc_cfg)

    def load_checkpoint(self):
        """Load a checkpoint from experiments dir."""
        self.check_point_exists = if_checkpoint_exists(self.exp_dir)
        if self.check_point_exists and self.resume_training:
            checkpoint_name, checkpoint = get_checkpoint(self.exp_dir)
            self.model.load_state_dict(checkpoint[MODEL_STATE])
            self.optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE])
            self.criterion.load_state_dict(checkpoint[CRITERION_STATE])
            self.start_iter = checkpoint[EPOCH]
            self.best_loss = checkpoint[BEST_LOSS]
            print("Loaded checkpoint '{}' from epoch {} with loss {}".format(
                checkpoint_name, self.start_iter, self.best_loss))
        else:
            if self.mode == TRAIN:
                print("Begining Training from Scratch")
            else:
                raise ValueError('Cannot load checkpoint')

    def train(self):
        """Train a model on a entire train data for several epochs."""
        for epoch in range(self.epochs):
            print(EPOCH_STR.format(epoch=epoch+1))
            print(TRAIN_STR)
            self.n_batches = len(self.dataloaders[TRAIN])
            self.train_running_loss = averageMeter()
            for batch_id, data in tqdm(enumerate(self.dataloaders[TRAIN])):
                self.train_step(data, epoch, batch_id)
                break
            self.validation_step(epoch)
            self.train_loss_meters.reset()
            self.save_model(epoch)
            if self.stop_training(epoch):
                self.writer.close()
                break

        self.writer.close()

    def validation(self):
        """Validate a model on a entire validation data."""
        self.validation_step(epoch=1)

    def train_step(self, data, epoch, batch):
        """Train a model on a batch of data.

        Arguments:
            data {tuple} -- [inputs and groundtruth for the network]
            epoch {[int]} -- [Current epoch number]
            batch {[int]} -- [current batch number]

        """
        self.optimizer.zero_grad()

        inputs, targets = data
        logits = self.model(inputs.to(self.device))
        labels = get_labels(targets, self.cfg, self.device)
        losses, loss = self.criterion(logits, labels)
        self.train_running_loss.update(loss)
        self.train_loss_meters.update(losses)
        loss.backward()
        self.optimizer.step()

        if (batch % self.print_interval == 0 or batch == self.n_batches-1):
            loss_weights = [p.data for p in self.criterion.parameters()]
            print('\nLoss weights {}'.format(loss_weights))

            self.writer.add_scalar('Loss/train', self.train_running_loss.avg,
                                   epoch*self.n_batches + batch)
            loss_str = LOSS_STR.format(epoch=epoch+1, batch=batch,
                                       loss=self.train_running_loss.avg)
            predictions = get_predictions(logits, self.cfg[MODEL][OUTPUTS],
                                          labels)

            for task, loss in self.train_loss_meters.meters.items():
                loss_str += TASK_LOSS.format(task=task, loss=loss.avg)
                self.writer.add_scalar('Loss/train_{}'.format(task),
                                       loss.avg,
                                       epoch*self.n_batches + batch)
            self.send_predictions_to_writer(inputs, predictions, labels,
                                            self.train_loss_meters, epoch,
                                            save_to_disk=False)
            print(loss_str)
        self.train_running_loss.reset()

    def validation_step(self, epoch):
        """Validate a model on a batch of data.

        Arguments:
            epoch {[int]} -- [Current epoch number]
        """
        print(VAL_STR)
        running_val_loss = averageMeter()
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.dataloaders[VAL])):
                inputs, targets = data
                logits = self.model(inputs.to(self.device))
                labels = get_labels(targets, self.cfg, self.device,
                                    get_postprocs=True)
                val_losses, val_loss = self.criterion(logits, labels)
                self.val_loss_meters.update(val_losses)
                running_val_loss.update(val_loss)
                predictions = get_predictions(logits, self.cfg[MODEL][OUTPUTS],
                                              labels)
                outputs = post_process_predictions(predictions,
                                                   self.cfg[POSTPROCS])
                self.val_metrics.update(labels, predictions)
                self.post_proc_metrics.update(labels, outputs)
                if self.mode == VAL:
                    self.send_predictions_to_writer(inputs, predictions,
                                                    labels,
                                                    self.val_loss_meters, i,
                                                    save_to_disk=True)
                    self.send_outputs_to_writer(outputs, labels,
                                                self.post_proc_metrics, i,
                                                save_to_disk=True)
        if self.mode != VAL:
            print(VAL_LOSS_STR.format(epoch=epoch + 1,
                                      loss=running_val_loss.avg))
            self.writer.add_scalar('Loss/Val', running_val_loss.avg, epoch)
            for task, loss in self.val_loss_meters.meters.items():
                print(TASK_LOSS.format(task=task, loss=loss.avg))
                self.writer.add_scalar('Loss/Validation_{}'.format(task),
                                       loss.avg, epoch)
                self.send_predictions_to_writer(inputs, predictions,
                                                labels,
                                                self.val_loss_meters, i,
                                                save_to_disk=True)
                self.send_outputs_to_writer(outputs, labels,
                                            self.post_proc_metrics, i,
                                            save_to_disk=True)

        print_metrics(self.val_metrics)
        print_metrics(self.post_proc_metrics)
        self.val_loss_meters.reset()
        self.val_metrics.reset()
        self.post_proc_metrics.reset()
        self.val_loss = running_val_loss.avg.data
        running_val_loss.reset()

    def send_predictions_to_writer(self, inputs, predictions, labels, meters,
                                   epoch=1, save_to_disk=False):
        for task in meters.meters.keys():
            add_images_to_writer(inputs, predictions[task], labels[task],
                                 self.writer, epoch, task, state=self.mode,
                                 save_to_disk=True)

    def send_outputs_to_writer(self, outputs, labels, metrics, epoch=1,
                               save_to_disk=False):
        for task in metrics.metrics.keys():
            add_images_to_writer(None, outputs[task], labels[task],
                                 self.writer, epoch, task, state=self.mode,
                                 save_to_disk=True)

    def save_model(self, epoch):
        """Save model to checkpoint.

        Arguments:
            epoch {[int]} -- [Current epoch number]
        """
        if self.val_loss <= self.best_loss:
            self.best_loss = self.val_loss
            self.state = {"epoch": self.start_iter + epoch + 1,
                          "model_state": self.model.state_dict(),
                          "criterion_state": self.criterion.state_dict(),
                          "optimizer_state": self.optimizer.state_dict(),
                          "best_loss": self.best_loss}
            save_path, _, _, time_stamp = get_save_path(self.cfg,
                                                        self.best_loss)
            torch.save(self.state, save_path)
            deleted_old = delete_old_checkpoint(save_path)
            model_name = MODEL_NAME.format(time_stamp=time_stamp,
                                           best_loss=self.best_loss)
            print("Saving checkpoint {} at epoch {}"
                  .format(model_name, epoch+1))
            if deleted_old:
                print('Deleted old checkpoints')
                self.plateau_count = 0
            else:
                self.plateau_count += 1

    def stop_training(self, epoch):
        """Stop training if stop criteria is met.

        Arguments:
            epoch {[int]} -- [Current epoch number]

        Returns:
            [bool] -- [true/false to stop training]
        """
        if self.plateau_count == self.patience and self.early_stop:
            early_stop_str = EARLY_STOP_STR.format(epochs=epoch+1)
            patience_str = PATIENCE_STR.format(patience=self.plateau_count)
            print(early_stop_str+patience_str)
        return True


def get_device(cfg):
    """Get device to place data on.

    Arguments:
        cfg {[dict]} -- [configuration settings for the experiment]

    Returns:
        [device] -- [device to place data.]
    """
    if not cfg[PARAMS][MULTIGPU]:
        # Select device specified in cfg.
        device_str = GPU_STR.format(gpu_id=cfg[PARAMS][GPU_ID])
    else:
        # Chose the first available device for multi-gpu training
        device_str = GPU_STR.format(gpu_id=0)
    device = torch.device(device_str if torch.cuda.is_available() else CPU)
    return device


def get_exp_dir(cfg):
    """Get the name of experiments directory.

    Arguments:
        cfg {[dict]} -- [configuration settings for the experiment]

    Returns:
        exp_dir_path{[str]} -- [path to experiment directory]
        network_name{[str]} -- [Name of the network that includes encoder_name
        and decoder_name]
        exp_name{[str]} -- [Name of the experiment that includes all tasks
        learnt by network]
    """
    encoder_name = cfg[MODEL][ENCODER]
    decoder_name = cfg[MODEL][DECODER]
    imsize = cfg[DATA][IM_SIZE]
    base_dir = cfg[MODEL][PRETRAINED_PATH]

    # create results directory if it's not specified in cfg.
    if base_dir is None:
        base_dir = RESULTS_DIR

    network_name = EXPERIMENT_NAME.format(encoder_name=encoder_name,
                                          decoder_name=decoder_name,
                                          imsize=imsize)
    exp_name = '_'.join(cfg[TASKS].keys())

    # create experiment directory based on tasks and network type.
    exp_dir_path = os.path.join(base_dir, exp_name, network_name)
    if not os.path.exists(exp_dir_path):
        os.makedirs(exp_dir_path)

    return exp_dir_path, exp_name, network_name


def get_save_path(cfg, best_loss=None):
    """Get path to save best model.

    Arguments:
        cfg {[dict]} -- [configuration settings for the experiment]

    Keyword Arguments:
        best_loss {[float]} -- [best loss of network] (default: {None})

    Returns:
        path[str] -- path to save model
        network_name{[str]} -- [Name of the network that includes encoder_name
        and decoder_name]
        exp_name{[str]} -- [Name of the experiment that includes all tasks
        learnt by network]
        time_stamp[str] -- [Timestamp in YYYY-MM-DD-HH:MM:SS]
    """
    exp_dir_path, exp_name, network_name = get_exp_dir(cfg)
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    path = os.path.join(exp_dir_path, MODEL_NAME.format(time_stamp=time_stamp,
                                                        best_loss=best_loss))

    return path, network_name, exp_name, time_stamp


def get_writer(cfg):
    """Get tf.writer to log experiments data to Tensorboard.

    Arguments:
        cfg {[dict]} -- [configuration settings for the experiment]

    Returns:
        [tf.SummaryWriter] -- [Object to log experiments data to Tensorboard]
    """
    base_dir = cfg[MODEL][PRETRAINED_PATH]
    savelogs = cfg[PARAMS][SAVE_LOGS]
    _, exp_name, task_name, time_stamp = get_save_path(cfg)
    log_dir = os.path.join(base_dir, LOGS, task_name,
                           exp_name, "{}".format(time_stamp))
    if not os.path.exists(log_dir) and savelogs:
        os.makedirs(log_dir)
        return SummaryWriter(log_dir=log_dir)
    else:
        return None


def get_best_model(list_of_models):
    """Get best model from given list of models.

    Arguments:
        list_of_models {[list]} -- [List of trained models]

    Returns:
        [.pkl file] -- [Neural Network model with best loss]
    """
    best_loss = MILLION
    best_model = None
    for model in list_of_models:
        checkpoint_name = model.split('/')[-1].split(PKL)[0]
        loss = float(checkpoint_name.split('_')[-1])
        if loss < best_loss:
            best_loss = loss
            best_model = model
    return best_model


def get_checkpoint_list(exp_dir):
    """Get list of checkpoint model files.

    Arguments:
        exp_dir {[str]} -- [Path to current experiment directory]

    Returns:
        [list] -- [List of trained models]
    """
    return glob.glob(os.path.join(exp_dir, '*.pkl'))


def get_checkpoint(exp_dir):
    """Get checkpoint model file.

    Arguments:
        exp_dir {[str]} -- [Path to current experiment directory]

    Returns:
        checkpoint_name[str] -- [Name of the checkpoint file with best_loss]
        model(torch.nn.Module) -- Neural network module with best_loss
    """
    list_of_models = get_checkpoint_list(exp_dir)
    checkpoint_name = get_best_model(list_of_models)
    model = torch.load(checkpoint_name)
    return checkpoint_name, model


def if_checkpoint_exists(exp_dir):
    """Check if a checkpoint exists in given path.

    Arguments:
        exp_dir {[str]} -- [Path to current experiment directory]


    Returns:
        [bool] -- [True/False based on checkpoint existence]
    """
    list_of_models = get_checkpoint_list(exp_dir)

    return True if len(list_of_models) != 0 else False


def get_model(cfg, device):
    """Get the model and place it on devices specified in config.

    Arguments:
        cfg {[dict]} -- [configuration settings for the experiment]
        device {[torch.device]} -- [device to place data.]

    Returns:
        [model(torch.nn.Module) -- Neural network module defined for the
        experiment
    """
    gpus = list(range(torch.cuda.device_count()))
    model = get_encoder_decoder(cfg)
    model = model.to(device)
    multigpu = cfg[PARAMS][MULTIGPU]
    if len(gpus) > 1 and multigpu:
        model = nn.DataParallel(model, device_ids=gpus, dim=0)
    return model


def delete_old_checkpoint(path_to_checkpoint):
    """Delete an old checkpoint in given path.

    Arguments:
        path_to_checkpoint[str] -- path to save model

    Returns:
        [bool] -- [True/False]
    """
    removed = False
    path = '/'.join(path_to_checkpoint.split('/')[:-1])
    list_of_models = get_checkpoint_list(path)
    for model in list_of_models:
        if model not in path_to_checkpoint:
            os.remove(model)
            removed = True
    return removed


def print_metrics(metrics):
    """Print metrics.

    Arguments:
        metrics {dict} -- [metrics to be printed]
    """
    print(METRICS_STR)
    for task in metrics.metrics.keys():
        if metrics.metrics[task] is not None:
            score, class_iou = metrics.metrics[task].get_scores()
            if isinstance(score, dict):
                header = score.keys()
                t = PrettyTable(header)
                t.title(task)
                t.add_row(['{:05.3f}'.format(score[k] * 100)
                           for k in score.keys()])
                print(t)
                if task == SEMANTIC:
                    print_table(class_iou, cat_labels)
                elif task == INSTANCE_PROBS:
                    print_table(class_iou, prob_labels)
                elif task == INSTANCE_CONTOUR:
                    print_table(class_iou, inst_labels)
                elif task == PANOPTIC:
                    t = PrettyTable(['Category']+list(class_iou.keys()))
                    for label in labels:
                        if label.ignoreInEval:
                            continue
                        row = ([label.name] + ['{:05.3f}'.format(
                            class_iou[metric][label.id]*100)
                            for metric in class_iou.keys()])
                        t.add_row(row)
                    print(t)


def print_table(data, labels):
    t = PrettyTable([labels[k].name for k in data.keys()])
    t.add_row(['{:05.3f}'.format(v*100) for v in data.values()])
    print(t)
